"""
Catastrophic Forgetting Probe — Scale Verification (Qwen2.5-1.5B)
Exactly 50,000 tokens per domain. Checks if signal is a small-model artifact.
Compares with 0.5B controlled results (results_controlled.json).
"""

import json
import math
import random
import time
from itertools import permutations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Adafactor
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID      = "Qwen/Qwen2.5-1.5B"
TOKEN_BUDGET  = 50_000
MAX_LEN       = 256
TRAIN_CHUNKS  = 175
EVAL_CHUNKS   = 19
TRAIN_STEPS   = 200
BATCH_SIZE    = 4
LR            = 2e-5
SEED          = 42
DTYPE         = torch.bfloat16
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
VRAM_WARN_GB  = 14.0
RESULTS_FILE  = "results_1p5b.json"
PREV_RESULTS  = "results_controlled.json"   # 0.5B controlled baseline

random.seed(SEED)
torch.manual_seed(SEED)

print(f"Device: {DEVICE}  ({torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'})")
print(f"Model:  {MODEL_ID}  dtype={DTYPE}")
print(f"Token budget per domain: {TOKEN_BUDGET:,}  ({TRAIN_CHUNKS} train + {EVAL_CHUNKS} eval chunks)\n")


# ── VRAM monitoring ───────────────────────────────────────────────────────────

def check_vram(label=""):
    if not torch.cuda.is_available():
        return 0.0
    used  = torch.cuda.memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    tag = f"  [{label}] " if label else "  "
    msg = f"{tag}VRAM: {used:.2f} / {total:.1f} GB"
    if used > VRAM_WARN_GB:
        print(f"  *** WARNING: {msg} — exceeds {VRAM_WARN_GB}GB threshold! ***")
    else:
        print(msg)
    return used


# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_math_texts():
    print("  Loading math (gsm8k)...")
    try:
        ds = load_dataset("gsm8k", "main", split="train")
        texts = [f"Question: {r['question']}\nAnswer: {r['answer']}" for r in ds]
        print(f"  gsm8k: {len(texts)} samples")
        return texts
    except Exception as e:
        print(f"  gsm8k failed ({e}), trying math_qa...")
    try:
        ds = load_dataset("math_qa", split="train")
        texts = [f"Problem: {r['Problem']}\nRationale: {r['Rationale']}" for r in ds]
        print(f"  math_qa: {len(texts)} samples")
        return texts
    except Exception as e:
        print(f"  math_qa failed ({e}), using wikitext fallback...")
    ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
    return [r["text"] for r in ds if len(r["text"].strip()) > 80]


def load_code_texts():
    print("  Loading code (code_search_net python)...")
    try:
        ds = load_dataset("code_search_net", "python", split="train")
        texts = [r["whole_func_string"] for r in ds]
        print(f"  code_search_net: {len(texts)} samples")
        return texts
    except Exception as e:
        print(f"  code_search_net failed ({e}), trying mbpp...")
    try:
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
        texts = [f"# {r['text']}\n{r['code']}" for r in ds]
        print(f"  mbpp: {len(texts)} samples")
        return texts
    except Exception as e:
        print(f"  mbpp failed ({e}), using wikitext fallback...")
    ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
    return [r["text"] for r in ds if len(r["text"].strip()) > 80]


def load_literature_texts():
    print("  Loading literature (wikitext-103-raw-v1)...")
    try:
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        texts = [r["text"] for r in ds if len(r["text"].strip()) > 80]
        print(f"  wikitext-103-raw-v1: {len(texts)} samples")
        return texts
    except Exception as e:
        print(f"  wikitext-103-raw-v1 failed ({e}), trying wikitext-103-v1...")
    try:
        ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
        texts = [r["text"] for r in ds if len(r["text"].strip()) > 80]
        print(f"  wikitext-103-v1: {len(texts)} samples")
        return texts
    except Exception as e:
        print(f"  wikitext failed ({e})")
    ds = load_dataset("bookcorpus", split="train")
    return [r["text"] for r in ds]


# ── Token-controlled chunking ──────────────────────────────────────────────────

def build_token_controlled_chunks(texts, tokenizer, token_budget, max_len):
    all_ids = []
    while len(all_ids) < token_budget:
        for text in texts:
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            all_ids.extend(ids)
            if len(all_ids) >= token_budget:
                break
    all_ids = all_ids[:token_budget]
    chunks = []
    for i in range(0, len(all_ids) - max_len + 1, max_len):
        chunks.append(torch.tensor(all_ids[i : i + max_len], dtype=torch.long))
    print(f"  {len(chunks)} chunks × {max_len} = {len(chunks)*max_len:,} tokens from {token_budget:,} budget")
    return chunks


def make_batches(chunks, batch_size, shuffle=True):
    ch = list(chunks)
    if shuffle:
        random.shuffle(ch)
    return [
        torch.stack(ch[i : i + batch_size])
        for i in range(0, len(ch) - batch_size + 1, batch_size)
    ]


# ── Training & eval ────────────────────────────────────────────────────────────

def finetune(model, batches, steps, lr, label=""):
    model.train()
    # Adafactor: factored second moments (~100MB) vs AdamW fp32 (~12GB).
    # VRAM budget: model(3GB) + grads(3GB) + Adafactor(~0.1GB) ≈ 6.1GB < 8GB
    # relative_step=False, scale_parameter=False → use explicit lr like AdamW
    optimizer = Adafactor(
        model.parameters(),
        lr=lr,
        relative_step=False,
        scale_parameter=False,
        weight_decay=0.01,
    )
    step = 0
    total_loss = 0.0
    while step < steps:
        for batch in batches:
            if step >= steps:
                break
            batch = batch.to(DEVICE)
            out = model(input_ids=batch, labels=batch)
            loss = out.loss
            if torch.isnan(loss):
                optimizer.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            step += 1
            if step % 50 == 0:
                vram = torch.cuda.memory_allocated() / 1024**3 if DEVICE == "cuda" else 0
                vram_tag = f"  VRAM={vram:.1f}GB{'  *** WARN ***' if vram > VRAM_WARN_GB else ''}"
                print(f"    {label} step {step}/{steps}  loss={total_loss/step:.4f}{vram_tag}")


@torch.no_grad()
def perplexity(model, batches):
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for batch in batches:
        batch = batch.to(DEVICE)
        out = model(input_ids=batch, labels=batch)
        if torch.isnan(out.loss):
            continue
        total_nll += out.loss.item() * batch.numel()
        total_tokens += batch.numel()
    return math.exp(total_nll / total_tokens) if total_tokens > 0 else float("inf")


def fresh_model(tokenizer):
    # Load to CPU first, then move to GPU — avoids device_map dispatch overhead
    # that can OOM mid-load on tight VRAM budgets.
    print("  Loading weights to CPU...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE)
    model.config.pad_token_id = tokenizer.eos_token_id
    # Gradient checkpointing: recompute activations during backward instead of
    # caching them. Saves ~1-2GB VRAM, ~30% slower backward — worth it here.
    model.gradient_checkpointing_enable()
    print("  Moving model to GPU...")
    model = model.to(DEVICE)
    torch.cuda.synchronize()
    print("  Model on GPU.")
    return model


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    # Pre-flight VRAM check
    if DEVICE == "cuda":
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Total VRAM: {total_vram:.1f} GB  (warn threshold: {VRAM_WARN_GB} GB)")
        # Adafactor (~0.1GB) + gradient checkpointing (recomputes activations)
        # VRAM: model(3GB) + grads(3GB) + Adafactor(~0.1GB) + activations(~0.5GB) ≈ 6.6GB
        print(f"Estimated peak VRAM: ~6.6 GB (model+grads+Adafactor+checkpointing)\n")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n── Loading datasets ──")
    raw = {
        "math":       load_math_texts(),
        "code":       load_code_texts(),
        "literature": load_literature_texts(),
    }

    print("\n── Building token-controlled chunks (50k tokens each) ──")
    domain_data = {}
    for name, texts in raw.items():
        random.shuffle(texts)
        all_chunks = build_token_controlled_chunks(texts, tokenizer, TOKEN_BUDGET, MAX_LEN)
        random.shuffle(all_chunks)
        domain_data[name] = {
            "train": all_chunks[:TRAIN_CHUNKS],
            "eval":  all_chunks[TRAIN_CHUNKS : TRAIN_CHUNKS + EVAL_CHUNKS],
        }
        print(f"  {name}: {len(domain_data[name]['train'])} train, {len(domain_data[name]['eval'])} eval chunks")

    for name, d in domain_data.items():
        if len(d["eval"]) == 0:
            raise RuntimeError(f"Domain '{name}' has 0 eval chunks")

    domains = list(domain_data.keys())
    results = {}

    print(f"\n{'='*62}")
    print(f"Running 6 ordered-pair experiments ({MODEL_ID})")
    print(f"{'='*62}")

    for domain_a, domain_b in permutations(domains, 2):
        pair_key = f"{domain_a}→{domain_b}"
        print(f"\n[{pair_key}]")

        train_a = make_batches(domain_data[domain_a]["train"], BATCH_SIZE)
        train_b = make_batches(domain_data[domain_b]["train"], BATCH_SIZE)
        eval_a  = make_batches(domain_data[domain_a]["eval"],  BATCH_SIZE, shuffle=False)

        print("  Loading fresh model...")
        model = fresh_model(tokenizer)
        check_vram("after model load")

        finetune(model, train_a, TRAIN_STEPS, LR, label=f"A={domain_a}")
        check_vram("after A finetune")

        ppl_before = perplexity(model, eval_a)
        print(f"  PPL({domain_a}) before B: {ppl_before:.3f}")

        finetune(model, train_b, TRAIN_STEPS, LR, label=f"B={domain_b}")
        check_vram("after B finetune")

        ppl_after = perplexity(model, eval_a)
        print(f"  PPL({domain_a}) after  B: {ppl_after:.3f}")

        fg = (ppl_after - ppl_before) / ppl_before * 100
        print(f"  Forgetting: {fg:+.2f}%")

        results[pair_key] = {
            "domain_a":       domain_a,
            "domain_b":       domain_b,
            "ppl_before":     round(ppl_before, 4),
            "ppl_after":      round(ppl_after, 4),
            "forgetting_pct": round(fg, 4),
        }

        del model
        torch.cuda.empty_cache()
        check_vram("after model del")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    elapsed = (time.time() - t0) / 60
    sorted_pairs = sorted(results.items(), key=lambda x: x[1]["forgetting_pct"])
    fg_vals  = [r["forgetting_pct"] for r in results.values()]
    min_pair = min(results, key=lambda k: results[k]["forgetting_pct"])
    max_pair = max(results, key=lambda k: results[k]["forgetting_pct"])
    spread   = max(fg_vals) - min(fg_vals)

    print(f"\nTotal time: {elapsed:.1f} min\n")
    print(f"{'='*64}")
    print(f"1.5B RESULTS (50k tokens/domain)")
    print(f"{'='*64}")
    print(f"{'Pair':<22} {'PPL_before':>10} {'PPL_after':>10} {'Forgetting':>12}")
    print(f"{'-'*64}")
    for k, r in sorted_pairs:
        print(f"{k:<22} {r['ppl_before']:>10.2f} {r['ppl_after']:>10.2f} {r['forgetting_pct']:>+11.2f}%")
    print(f"{'='*64}")
    print(f"Spread: {spread:.2f} pp  |  Lowest: {min_pair}  |  Highest: {max_pair}")

    # ── Cross-compare with 0.5B controlled baseline ────────────────────────────
    print(f"\n{'='*64}")
    print("COMPARISON: 0.5B Controlled  vs  1.5B")
    print(f"{'='*64}")
    try:
        with open(PREV_RESULTS) as f:
            prev = {k: v for k, v in json.load(f).items() if not k.startswith("_")}

        prev_fg      = [v["forgetting_pct"] for v in prev.values()]
        prev_ranking = sorted(prev,    key=lambda k: prev[k]["forgetting_pct"])
        curr_ranking = sorted(results, key=lambda k: results[k]["forgetting_pct"])
        prev_spread  = max(prev_fg) - min(prev_fg)

        print(f"\n{'Pair':<22} {'0.5B ctrl':>12} {'1.5B':>10}  Rank match?")
        print(f"{'-'*58}")
        all_pairs = sorted(set(list(prev.keys()) + list(results.keys())))
        for k in all_pairs:
            p  = prev.get(k, {}).get("forgetting_pct", float("nan"))
            c  = results.get(k, {}).get("forgetting_pct", float("nan"))
            pr = prev_ranking.index(k) + 1 if k in prev_ranking else "?"
            cr = curr_ranking.index(k) + 1 if k in curr_ranking else "?"
            match = "OK" if pr == cr else "NO"
            print(f"{k:<22} {p:>+11.2f}% {c:>+9.2f}%  #{pr}->{cr} {match}")

        ranking_stable   = prev_ranking == curr_ranking
        spread_direction = ("grew" if spread > prev_spread + 1
                            else "shrank" if spread < prev_spread - 1
                            else "stable")

        # Check code→math < math→code asymmetry
        cm_0p5 = prev.get("code→math", {}).get("forgetting_pct", None)
        mc_0p5 = prev.get("math→code", {}).get("forgetting_pct", None)
        cm_1p5 = results.get("code→math", {}).get("forgetting_pct", None)
        mc_1p5 = results.get("math→code", {}).get("forgetting_pct", None)
        asym_0p5 = (cm_0p5 < mc_0p5) if (cm_0p5 is not None and mc_0p5 is not None) else None
        asym_1p5 = (cm_1p5 < mc_1p5) if (cm_1p5 is not None and mc_1p5 is not None) else None

        math_lit_key = "math→literature"
        still_highest = (max_pair == math_lit_key)

        print(f"\n0.5B spread: {prev_spread:.2f} pp  →  1.5B spread: {spread:.2f} pp  ({spread_direction})")
        print(f"Ranking stable:          {'YES' if ranking_stable else 'NO'}")
        print(f"code→math < math→code asymmetry (0.5B): {'YES' if asym_0p5 else 'NO' if asym_0p5 is not None else 'N/A'}")
        print(f"code→math < math→code asymmetry (1.5B): {'YES' if asym_1p5 else 'NO' if asym_1p5 is not None else 'N/A'}")
        print(f"math→literature still highest (1.5B):   {'YES' if still_highest else 'NO'}")

        if spread > 10 and ranking_stable:
            verdict = f"STRUCTURAL: spread {spread_direction} ({prev_spread:.1f}pp → {spread:.1f}pp) and ranking stable. Signal is not a small-model artifact."
        elif spread > 10 and not ranking_stable:
            verdict = f"PARTIALLY STRUCTURAL: spread {spread_direction} but ranking shifted. Core signal holds but details are noisy."
        elif spread <= 10:
            verdict = f"WEAKENED: spread {spread_direction} to {spread:.1f}pp (<10pp). Signal may be size-dependent."
        else:
            verdict = "INCONCLUSIVE."

        print(f"\nVerdict: {verdict}")

        results["_summary"] = {
            "model":            MODEL_ID,
            "min_pair":         min_pair,
            "max_pair":         max_pair,
            "spread_pct":       round(spread, 4),
            "prev_spread_pct":  round(prev_spread, 4),
            "spread_direction": spread_direction,
            "ranking_stable":   ranking_stable,
            "asym_held_1p5b":   asym_1p5,
            "math_lit_highest": still_highest,
            "verdict":          verdict,
            "elapsed_min":      round(elapsed, 1),
        }

    except FileNotFoundError:
        print(f"  {PREV_RESULTS} not found — skipping cross-comparison")
        results["_summary"] = {
            "model": MODEL_ID, "min_pair": min_pair, "max_pair": max_pair,
            "spread_pct": round(spread, 4), "elapsed_min": round(elapsed, 1),
        }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
