# ============================================================
# Catastrophic Forgetting Probe — TPU Colab Edition
# Run this in a Colab notebook with Runtime → TPU
#
# Scales:
#   Colab TPU v2-8  (8 cores × 8GB = 64GB) → 3B, 7B
#   TRC v3-8 / v4   (approved)              → 14B, 32B
#
# Instructions:
#   1. Runtime → Change runtime type → TPU
#   2. Run all cells (or: !python forgetting_tpu_colab.py)
# ============================================================

# ── 0. Install / imports ──────────────────────────────────────────────────────
import subprocess, sys

def pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

pip("transformers", "datasets", "accelerate")

import json, math, random, time, os
from itertools import permutations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ── 1. Detect backend: TPU (XLA) or GPU/CPU ──────────────────────────────────
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    USE_XLA = True
    DEVICE  = xm.xla_device()
    print(f"Backend: TPU/XLA  device={DEVICE}")
except ImportError:
    USE_XLA = False
    DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Backend: {'GPU' if 'cuda' in str(DEVICE) else 'CPU'}  device={DEVICE}")

def step_optimizer(optimizer):
    """Optimizer step + XLA mark_step if on TPU."""
    if USE_XLA:
        xm.optimizer_step(optimizer)
    else:
        optimizer.step()

# ── 2. Config ─────────────────────────────────────────────────────────────────
# Change MODEL_ID to scale up:
#   "Qwen/Qwen2.5-3B"   → Colab TPU v2-8 (fits, ~6GB weights)
#   "Qwen/Qwen2.5-7B"   → Colab TPU v2-8 (fits, ~14GB weights, use all 8 cores)
#   "Qwen/Qwen2.5-14B"  → TRC v3-8 / v4 (28GB weights, needs sharding)
#   "Qwen/Qwen2.5-32B"  → TRC v4-32+
MODEL_ID      = "Qwen/Qwen2.5-3B"
TOKEN_BUDGET  = 50_000
MAX_LEN       = 256
TRAIN_CHUNKS  = 175    # 175 × 256 = 44,800 training tokens
EVAL_CHUNKS   = 19     # 19  × 256 = 4,864  eval tokens
TRAIN_STEPS   = 200
BATCH_SIZE    = 8      # TPU prefers power-of-2; increase if HBM allows
LR            = 2e-5
SEED          = 42
DTYPE         = torch.bfloat16   # native on TPU; also good on A100/H100
RESULTS_FILE  = f"results_{MODEL_ID.split('/')[-1].lower()}.json"
PREV_RESULTS  = "results_controlled.json"  # 0.5B baseline for comparison

random.seed(SEED)
torch.manual_seed(SEED)
print(f"Model: {MODEL_ID}  dtype={DTYPE}  batch={BATCH_SIZE}")
print(f"Token budget: {TOKEN_BUDGET:,}/domain  Steps: {TRAIN_STEPS}  Results: {RESULTS_FILE}")


# ── 3. Dataset loaders ────────────────────────────────────────────────────────

def load_math_texts():
    print("  Loading math (gsm8k)...")
    try:
        ds = load_dataset("gsm8k", "main", split="train")
        texts = [f"Question: {r['question']}\nAnswer: {r['answer']}" for r in ds]
        print(f"  gsm8k: {len(texts)} samples"); return texts
    except Exception as e: print(f"  gsm8k failed: {e}")
    try:
        ds = load_dataset("math_qa", split="train")
        texts = [f"Problem: {r['Problem']}\nRationale: {r['Rationale']}" for r in ds]
        print(f"  math_qa: {len(texts)} samples"); return texts
    except Exception as e: print(f"  math_qa failed: {e}")
    ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
    return [r["text"] for r in ds if len(r["text"].strip()) > 80]


def load_code_texts():
    print("  Loading code (code_search_net python)...")
    try:
        ds = load_dataset("code_search_net", "python", split="train")
        texts = [r["whole_func_string"] for r in ds]
        print(f"  code_search_net: {len(texts)} samples"); return texts
    except Exception as e: print(f"  code_search_net failed: {e}")
    try:
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
        texts = [f"# {r['text']}\n{r['code']}" for r in ds]
        print(f"  mbpp: {len(texts)} samples"); return texts
    except Exception as e: print(f"  mbpp failed: {e}")
    ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
    return [r["text"] for r in ds if len(r["text"].strip()) > 80]


def load_literature_texts():
    print("  Loading literature (wikitext-103-raw-v1)...")
    try:
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        texts = [r["text"] for r in ds if len(r["text"].strip()) > 80]
        print(f"  wikitext-103-raw-v1: {len(texts)} samples"); return texts
    except Exception as e: print(f"  failed: {e}")
    ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
    return [r["text"] for r in ds if len(r["text"].strip()) > 80]


# ── 4. Tokenisation ───────────────────────────────────────────────────────────

def build_chunks(texts, tokenizer, budget, max_len):
    ids = []
    while len(ids) < budget:
        for t in texts:
            ids.extend(tokenizer(t, add_special_tokens=False)["input_ids"])
            if len(ids) >= budget: break
    ids = ids[:budget]
    chunks = [torch.tensor(ids[i:i+max_len], dtype=torch.long)
              for i in range(0, len(ids)-max_len+1, max_len)]
    print(f"  {len(chunks)} chunks × {max_len} = {len(chunks)*max_len:,} tokens")
    return chunks


def make_batches(chunks, bs, shuffle=True):
    ch = list(chunks)
    if shuffle: random.shuffle(ch)
    # TPU: pad last batch to full size so XLA doesn't recompile for partial batches
    while len(ch) % bs != 0:
        ch.append(ch[-1])
    return [torch.stack(ch[i:i+bs]) for i in range(0, len(ch), bs)]


# ── 5. Training & evaluation ──────────────────────────────────────────────────

def finetune(model, batches, steps, lr, label=""):
    model.train()
    # Standard AdamW — bitsandbytes not available on TPU
    # TPU HBM is large enough for full fp32 optimizer states at these sizes
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    step, total_loss = 0, 0.0

    while step < steps:
        for batch in batches:
            if step >= steps: break
            batch = batch.to(DEVICE)
            out   = model(input_ids=batch, labels=batch)
            loss  = out.loss
            if torch.isnan(loss):
                optimizer.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            step_optimizer(optimizer)   # XLA-aware step
            optimizer.zero_grad()
            total_loss += loss.item()
            step += 1
            if step % 50 == 0:
                print(f"    {label} step {step}/{steps}  loss={total_loss/step:.4f}")

    if USE_XLA:
        xm.mark_step()   # flush pending XLA computations


@torch.no_grad()
def perplexity(model, batches):
    model.eval()
    nll, ntok = 0.0, 0
    for batch in batches:
        batch = batch.to(DEVICE)
        out   = model(input_ids=batch, labels=batch)
        if torch.isnan(out.loss): continue
        nll  += out.loss.item() * batch.numel()
        ntok += batch.numel()
    if USE_XLA: xm.mark_step()
    return math.exp(nll / ntok) if ntok else float("inf")


def fresh_model(tokenizer):
    """Load model fresh from HuggingFace each pair (ensures no gradient leakage)."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=DTYPE,
        # No device_map — move to XLA device explicitly
    )
    model.config.pad_token_id = tokenizer.eos_token_id
    model = model.to(DEVICE)
    if USE_XLA: xm.mark_step()
    return model


# ── 6. Main experiment ────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n── Loading datasets ──")
    raw = {
        "math":       load_math_texts(),
        "code":       load_code_texts(),
        "literature": load_literature_texts(),
    }

    print("\n── Tokenizing (50k tokens each) ──")
    domain_data = {}
    for name, texts in raw.items():
        random.shuffle(texts)
        chunks = build_chunks(texts, tokenizer, TOKEN_BUDGET, MAX_LEN)
        random.shuffle(chunks)
        domain_data[name] = {
            "train": chunks[:TRAIN_CHUNKS],
            "eval":  chunks[TRAIN_CHUNKS : TRAIN_CHUNKS + EVAL_CHUNKS],
        }
        print(f"  {name}: {len(domain_data[name]['train'])} train, "
              f"{len(domain_data[name]['eval'])} eval chunks")

    for name, d in domain_data.items():
        assert len(d["eval"]) > 0, f"Domain '{name}' has 0 eval chunks"

    domains  = list(domain_data.keys())
    results  = {}

    print(f"\n{'='*62}")
    print(f"Running 6 ordered-pair experiments  [{MODEL_ID}]")
    print(f"{'='*62}")

    for domain_a, domain_b in permutations(domains, 2):
        pair_key = f"{domain_a}->{domain_b}"
        print(f"\n[{pair_key}]")

        train_a = make_batches(domain_data[domain_a]["train"], BATCH_SIZE)
        train_b = make_batches(domain_data[domain_b]["train"], BATCH_SIZE)
        eval_a  = make_batches(domain_data[domain_a]["eval"],  BATCH_SIZE, shuffle=False)

        print("  Loading fresh model...")
        model = fresh_model(tokenizer)

        finetune(model, train_a, TRAIN_STEPS, LR, label=f"A={domain_a}")
        ppl_before = perplexity(model, eval_a)
        print(f"  PPL({domain_a}) before B: {ppl_before:.3f}")

        finetune(model, train_b, TRAIN_STEPS, LR, label=f"B={domain_b}")
        ppl_after = perplexity(model, eval_a)
        print(f"  PPL({domain_a}) after  B: {ppl_after:.3f}")

        fg = (ppl_after - ppl_before) / ppl_before * 100
        print(f"  Forgetting: {fg:+.2f}%")

        results[pair_key] = {
            "domain_a": domain_a, "domain_b": domain_b,
            "ppl_before":     round(ppl_before, 4),
            "ppl_after":      round(ppl_after,  4),
            "forgetting_pct": round(fg, 4),
        }

        del model
        if USE_XLA:
            xm.mark_step()
        else:
            torch.cuda.empty_cache()

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    elapsed      = (time.time() - t0) / 60
    sorted_pairs = sorted(results.items(), key=lambda x: x[1]["forgetting_pct"])
    fg_vals      = [r["forgetting_pct"] for r in results.values()]
    min_pair     = min(results, key=lambda k: results[k]["forgetting_pct"])
    max_pair     = max(results, key=lambda k: results[k]["forgetting_pct"])
    spread       = max(fg_vals) - min(fg_vals)

    print(f"\nTotal time: {elapsed:.1f} min\n")
    print(f"{'='*64}")
    print(f"RESULTS  [{MODEL_ID}]  (50k tokens/domain)")
    print(f"{'='*64}")
    print(f"{'Pair':<22} {'PPL_before':>10} {'PPL_after':>10} {'Forgetting':>12}")
    print(f"{'-'*64}")
    for k, r in sorted_pairs:
        print(f"{k:<22} {r['ppl_before']:>10.2f} {r['ppl_after']:>10.2f} "
              f"{r['forgetting_pct']:>+11.2f}%")
    print(f"{'='*64}")
    print(f"Spread: {spread:.2f} pp  |  Lowest: {min_pair}  |  Highest: {max_pair}")

    if spread > 10:
        verdict = "REAL signal — >10pp spread at this scale."
    elif spread > 3:
        verdict = "WEAK signal — 3-10pp spread."
    else:
        verdict = "NOISE — <3pp spread."
    print(f"Verdict: {verdict}")

    # ── Cross-compare with 0.5B baseline ──────────────────────────────────────
    try:
        with open(PREV_RESULTS) as f:
            prev = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        prev_fg      = [v["forgetting_pct"] for v in prev.values()]
        prev_spread  = max(prev_fg) - min(prev_fg)
        prev_ranking = sorted(prev,    key=lambda k: prev[k]["forgetting_pct"])
        curr_ranking = sorted(results, key=lambda k: results[k]["forgetting_pct"])
        spread_dir   = ("grew"   if spread > prev_spread + 1 else
                        "shrank" if spread < prev_spread - 1 else "stable")

        print(f"\n{'='*64}")
        print(f"vs 0.5B baseline ({PREV_RESULTS})")
        print(f"{'='*64}")
        print(f"{'Pair':<22} {'0.5B ctrl':>12} {MODEL_ID.split('/')[-1]:>10}  Rank?")
        print(f"{'-'*58}")
        for k in sorted(set(list(prev.keys()) + list(results.keys()))):
            p  = prev.get(k, {}).get("forgetting_pct", float("nan"))
            c  = results.get(k, {}).get("forgetting_pct", float("nan"))
            pr = prev_ranking.index(k) + 1 if k in prev_ranking else "?"
            cr = curr_ranking.index(k) + 1 if k in curr_ranking else "?"
            match = "OK" if pr == cr else "NO"
            print(f"{k:<22} {p:>+11.2f}% {c:>+9.2f}%  #{pr}->{cr} {match}")

        cm_0 = prev.get("code->math", {}).get("forgetting_pct")
        mc_0 = prev.get("math->code", {}).get("forgetting_pct")
        cm_n = results.get("code->math", {}).get("forgetting_pct")
        mc_n = results.get("math->code", {}).get("forgetting_pct")

        print(f"\n0.5B spread: {prev_spread:.2f} pp  -> current: {spread:.2f} pp  ({spread_dir})")
        print(f"Asymmetry code->math < math->code  0.5B: {'YES' if cm_0 and mc_0 and cm_0 < mc_0 else 'NO'}")
        print(f"Asymmetry code->math < math->code  now:  {'YES' if cm_n and mc_n and cm_n < mc_n else 'NO'}")
    except FileNotFoundError:
        print(f"\n(No {PREV_RESULTS} found — run 0.5B experiment first for cross-model comparison)")

    results["_summary"] = {
        "model": MODEL_ID, "spread_pct": round(spread, 4),
        "min_pair": min_pair, "max_pair": max_pair,
        "verdict": verdict, "elapsed_min": round(elapsed, 1),
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {RESULTS_FILE}")


# ── 7. Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
