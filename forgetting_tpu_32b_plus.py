"""
Forgetting Probe — Cloud TPU 32B+ Edition
==========================================
Runs the 6 ordered-pair catastrophic-forgetting experiment at 32B / 72B scale
using PyTorch/XLA SPMD for model sharding across all TPU chips.

Exactly matches the prior controlled protocol:
  - 50k tokens/domain, 200 steps, lr=2e-5, batch=4, seed=42, max_len=256
  - domains: math (gsm8k), code (code_search_net), literature (wikitext-103)
  - 6 ordered pairs from permutations(domains, 2)

Recommended topologies:
  32B (64 GB weights)  → v4-8  (256 GB), v5e-8 (128 GB), v6e-4 (128 GB)
  72B (144 GB weights) → v4-32 (1 TB),   v5e-16 (256 GB), v6e-8 (256 GB)

SINGLE-HOST (v4-8 / v5e-8 / v6e-8):
  MODEL_ID=Qwen/Qwen2.5-32B python forgetting_tpu_32b_plus.py

MULTI-HOST (v4-32 — run on ALL workers simultaneously):
  gcloud compute tpus tpu-vm ssh MY_TPU --zone=us-central2-b --worker=all \
    --command="MODEL_ID=Qwen/Qwen2.5-32B python forgetting_tpu_32b_plus.py"

For 72B on v6e-64 (spot):
  gcloud compute tpus tpu-vm ssh MY_TPU --zone=us-east1-d --worker=all \
    --command="MODEL_ID=Qwen/Qwen2.5-72B python forgetting_tpu_32b_plus.py"

Spot preemption: results are written after every pair, so a restart can
resume from the last completed pair (set RESUME=1 env var to skip done pairs).
"""

# ── 0. SPMD must be enabled BEFORE any other torch_xla import ────────────────
import os
os.environ.setdefault("PJRT_DEVICE", "TPU")

import torch_xla.runtime as xr
xr.use_spmd()

import numpy as np
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs

# ── 1. Detect topology ────────────────────────────────────────────────────────
num_chips = xr.global_runtime_device_count()
DEVICE    = xm.xla_device()

mesh = xs.Mesh(np.array(range(num_chips)), (num_chips,), ("model",))
xs.set_global_mesh(mesh)

print(f"TPU chips: {num_chips}  |  device: {DEVICE}", flush=True)

# ── 2. Standard imports ───────────────────────────────────────────────────────
import json, math, random, time, sys
from itertools import permutations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ── 3. Config ─────────────────────────────────────────────────────────────────
MODEL_ID     = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-32B")
TOKEN_BUDGET = 50_000
MAX_LEN      = 256
TRAIN_CHUNKS = 175
EVAL_CHUNKS  = 19
TRAIN_STEPS  = 200
BATCH_SIZE   = 4       # matches prior controlled scale sweep
LR           = 2e-5
SEED         = 42
DTYPE        = torch.bfloat16
MODEL_SLUG   = MODEL_ID.split("/")[-1].lower()
RESULTS_FILE = f"results_{MODEL_SLUG}_tpu.json"
PREV_RESULTS = "results_controlled.json"   # 0.5B baseline for cross-comparison
RESUME       = bool(int(os.environ.get("RESUME", "0")))
GCS_BUCKET   = "gs://forgettingprobe-results"

random.seed(SEED)
torch.manual_seed(SEED)
print(f"Model: {MODEL_ID}  batch={BATCH_SIZE}  steps={TRAIN_STEPS}  "
      f"chips={num_chips}  resume={RESUME}", flush=True)


# ── 4. SPMD sharding ──────────────────────────────────────────────────────────
# Megatron-style column/row-parallel split for transformer weight matrices.
# After model.to(DEVICE) (lazy in XLA), we annotate sharding BEFORE mark_step()
# so the compiler materialises each chip's shard directly — never a full replica.
#
# Column-parallel (shard output dim = first dim):
#   q/k/v projections, gate/up in MLP, embedding table, lm_head
# Row-parallel (shard input dim = second dim):
#   o projection, down projection in MLP
# Replicated (1D or tiny):
#   layernorm weights/biases, attention biases

_COL_PARALLEL_KEYS = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj",
                      "embed_tokens", "lm_head")
_ROW_PARALLEL_KEYS = ("o_proj", "down_proj")


def apply_sharding(model):
    sharded = 0
    for name, param in model.named_parameters():
        if param.dim() < 2 or param.numel() < 65_536:
            # 1-D params (layer-norm, biases) and tiny tensors: replicate
            xs.mark_sharding(param, mesh, (None,) * param.dim())
        elif any(k in name for k in _COL_PARALLEL_KEYS):
            xs.mark_sharding(param, mesh, ("model", None))
            sharded += 1
        elif any(k in name for k in _ROW_PARALLEL_KEYS):
            xs.mark_sharding(param, mesh, (None, "model"))
            sharded += 1
        else:
            xs.mark_sharding(param, mesh, (None,) * param.dim())
    print(f"  Sharded {sharded} weight matrices across {num_chips} chips",
          flush=True)


def load_model(tokenizer):
    """
    Load model onto CPU (low_cpu_mem_usage avoids peak double-RAM),
    move to XLA lazily, annotate SPMD sharding, then materialise shards.
    Peak HBM per chip = model_size / num_chips + optimizer states / num_chips.
    """
    print(f"  Loading {MODEL_ID} on CPU (low_cpu_mem_usage=True)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tokenizer.eos_token_id
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  {n_params:.1f}B params  |  moving to XLA (lazy)...", flush=True)

    model = model.to(DEVICE)   # lazy — no HBM allocation yet
    apply_sharding(model)      # annotate before mark_step materialises
    xm.mark_step()             # compiler shards weights onto chips

    print(f"  Model ready: {n_params:.1f}B params across {num_chips} chips",
          flush=True)
    return model


# ── 5. Dataset loaders ────────────────────────────────────────────────────────
def gcs_backup(local_file):
    """Copy results file to GCS bucket for safe retrieval if VM is preempted."""
    import subprocess
    try:
        subprocess.run(
            ["gsutil", "cp", local_file, f"{GCS_BUCKET}/{local_file}"],
            check=True, capture_output=True)
        print(f"  Backed up -> {GCS_BUCKET}/{local_file}", flush=True)
    except Exception as e:
        print(f"  GCS backup failed (non-fatal): {e}", flush=True)


def load_math_texts():
    print("  Loading math...", flush=True)
    try:
        ds = load_dataset("gsm8k", "main", split="train")
        t = [f"Question: {r['question']}\nAnswer: {r['answer']}" for r in ds]
        print(f"  gsm8k: {len(t)} samples"); return t
    except Exception as e:
        print(f"  gsm8k failed ({e}), trying math_qa...", flush=True)
    try:
        ds = load_dataset("math_qa", split="train")
        t = [f"Problem: {r['Problem']}\nRationale: {r['Rationale']}" for r in ds]
        print(f"  math_qa: {len(t)} samples"); return t
    except Exception as e:
        print(f"  math_qa failed ({e}), falling back to wikitext", flush=True)
    ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
    return [r["text"] for r in ds if len(r["text"].strip()) > 80]


def load_code_texts():
    print("  Loading code...", flush=True)
    try:
        ds = load_dataset("code_search_net", "python", split="train")
        t = [r["whole_func_string"] for r in ds]
        print(f"  code_search_net: {len(t)} samples"); return t
    except Exception as e:
        print(f"  code_search_net failed ({e}), trying mbpp...", flush=True)
    try:
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
        t = [f"# {r['text']}\n{r['code']}" for r in ds]
        print(f"  mbpp: {len(t)} samples"); return t
    except Exception as e:
        print(f"  mbpp failed ({e}), falling back to wikitext", flush=True)
    ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
    return [r["text"] for r in ds if len(r["text"].strip()) > 80]


def load_literature_texts():
    print("  Loading literature...", flush=True)
    try:
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        t = [r["text"] for r in ds if len(r["text"].strip()) > 80]
        print(f"  wikitext-103-raw-v1: {len(t)} samples"); return t
    except Exception as e:
        print(f"  wikitext-103-raw-v1 failed ({e})", flush=True)
    ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
    return [r["text"] for r in ds if len(r["text"].strip()) > 80]


# ── 6. Tokenisation ───────────────────────────────────────────────────────────
def build_chunks(texts, tokenizer, budget, max_len):
    ids = []
    while len(ids) < budget:
        for t in texts:
            ids.extend(tokenizer(t, add_special_tokens=False)["input_ids"])
            if len(ids) >= budget:
                break
    ids = ids[:budget]
    chunks = [torch.tensor(ids[i:i + max_len], dtype=torch.long)
              for i in range(0, len(ids) - max_len + 1, max_len)]
    print(f"  {len(chunks)} chunks × {max_len} = {len(chunks) * max_len:,} tokens")
    return chunks


def make_batches(chunks, bs, shuffle=True):
    ch = list(chunks)
    if shuffle:
        random.shuffle(ch)
    # Drop incomplete tail — matches forgetting_controlled.py / batchscale_2d.py.
    # Batch shape is always [bs, MAX_LEN], so XLA never recompiles.
    return [torch.stack(ch[i:i + bs]) for i in range(0, len(ch) - bs + 1, bs)]


# ── 7. Training ───────────────────────────────────────────────────────────────
def finetune(model, batches, steps, lr, label=""):
    model.train()
    # Standard AdamW — bitsandbytes is not available on TPU.
    # With SPMD, optimizer states are also sharded so fp32 states fit easily.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    step, total_loss = 0, 0.0

    while step < steps:
        for batch in batches:
            if step >= steps:
                break
            batch = batch.to(DEVICE)
            # Model-parallel (not data-parallel): replicate input across chips.
            xs.mark_sharding(batch, mesh, (None, None))

            out  = model(input_ids=batch, labels=batch)
            loss = out.loss
            if torch.isnan(loss):
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            xm.optimizer_step(optimizer)   # XLA-aware step + implicit mark_step
            optimizer.zero_grad()

            total_loss += loss.item()
            step += 1
            if step % 50 == 0:
                print(f"    {label} step {step}/{steps}  "
                      f"loss={total_loss / step:.4f}", flush=True)

    xm.mark_step()


# ── 8. Evaluation ─────────────────────────────────────────────────────────────
@torch.no_grad()
def perplexity(model, batches):
    model.eval()
    nll, ntok = 0.0, 0
    for batch in batches:
        batch = batch.to(DEVICE)
        xs.mark_sharding(batch, mesh, (None, None))
        out  = model(input_ids=batch, labels=batch)
        if torch.isnan(out.loss):
            continue
        nll  += out.loss.item() * batch.numel()
        ntok += batch.numel()
    xm.mark_step()
    return math.exp(nll / ntok) if ntok else float("inf")


# ── 9. Main experiment ────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    # Load any previously saved results (spot preemption resume)
    results = {}
    if RESUME and os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = {k: v for k, v in json.load(f).items()
                       if not k.startswith("_")}
        print(f"Resuming: {len(results)} pairs already done: "
              f"{list(results.keys())}", flush=True)

    print("\nLoading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n── Datasets ──", flush=True)
    raw = {
        "math":       load_math_texts(),
        "code":       load_code_texts(),
        "literature": load_literature_texts(),
    }

    print("\n── Tokenizing (50k tokens each) ──", flush=True)
    domain_data = {}
    for name, texts in raw.items():
        random.shuffle(texts)
        chunks = build_chunks(texts, tokenizer, TOKEN_BUDGET, MAX_LEN)
        random.shuffle(chunks)
        domain_data[name] = {
            "train": chunks[:TRAIN_CHUNKS],
            "eval":  chunks[TRAIN_CHUNKS: TRAIN_CHUNKS + EVAL_CHUNKS],
        }
        print(f"  {name}: {len(domain_data[name]['train'])} train, "
              f"{len(domain_data[name]['eval'])} eval chunks")

    for name, d in domain_data.items():
        assert len(d["eval"]) > 0, f"Domain '{name}' has 0 eval chunks"

    domains = list(domain_data.keys())

    print(f"\n{'='*62}")
    print(f"6 ordered-pair experiments  [{MODEL_ID}  chips={num_chips}]")
    print(f"{'='*62}")

    for domain_a, domain_b in permutations(domains, 2):
        pair_key = f"{domain_a}→{domain_b}"

        if pair_key in results:
            print(f"\n[{pair_key}] already done — skipping", flush=True)
            continue

        print(f"\n[{pair_key}]", flush=True)

        train_a = make_batches(domain_data[domain_a]["train"], BATCH_SIZE)
        train_b = make_batches(domain_data[domain_b]["train"], BATCH_SIZE)
        eval_a  = make_batches(domain_data[domain_a]["eval"],  BATCH_SIZE,
                               shuffle=False)

        print("  Loading fresh model...", flush=True)
        model = load_model(tokenizer)

        finetune(model, train_a, TRAIN_STEPS, LR, label=f"A={domain_a}")
        ppl_before = perplexity(model, eval_a)
        print(f"  PPL({domain_a}) before B: {ppl_before:.3f}", flush=True)

        finetune(model, train_b, TRAIN_STEPS, LR, label=f"B={domain_b}")
        ppl_after = perplexity(model, eval_a)
        print(f"  PPL({domain_a}) after  B: {ppl_after:.3f}", flush=True)

        fg = (ppl_after - ppl_before) / ppl_before * 100
        print(f"  Forgetting: {fg:+.2f}%", flush=True)

        results[pair_key] = {
            "domain_a":      domain_a,
            "domain_b":      domain_b,
            "ppl_before":    round(ppl_before, 4),
            "ppl_after":     round(ppl_after,  4),
            "forgetting_pct": round(fg, 4),
        }

        del model
        xm.mark_step()

        # Save after every pair — spot instances can be preempted anytime
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved -> {RESULTS_FILE}", flush=True)
        gcs_backup(RESULTS_FILE)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed      = (time.time() - t0) / 60
    fg_vals      = [r["forgetting_pct"] for r in results.values()]
    sorted_pairs = sorted(results.items(), key=lambda x: x[1]["forgetting_pct"])
    spread       = max(fg_vals) - min(fg_vals)
    min_pair     = min(results, key=lambda k: results[k]["forgetting_pct"])
    max_pair     = max(results, key=lambda k: results[k]["forgetting_pct"])

    if spread > 10:
        verdict = "REAL signal — >10pp spread at this scale."
    elif spread > 3:
        verdict = "WEAK signal — 3-10pp spread."
    else:
        verdict = "NOISE — <3pp spread."

    print(f"\nTotal time: {elapsed:.1f} min")
    print(f"\n{'='*64}")
    print(f"RESULTS  [{MODEL_ID}]  (50k tokens/domain, batch={BATCH_SIZE})")
    print(f"{'='*64}")
    print(f"{'Pair':<22} {'PPL_before':>10} {'PPL_after':>10} {'Forgetting':>12}")
    print(f"{'-'*64}")
    for k, r in sorted_pairs:
        print(f"{k:<22} {r['ppl_before']:>10.2f} {r['ppl_after']:>10.2f} "
              f"{r['forgetting_pct']:>+11.2f}%")
    print(f"{'='*64}")
    print(f"Spread: {spread:.2f} pp  |  Lowest: {min_pair}  |  Highest: {max_pair}")
    print(f"Verdict: {verdict}")

    # ── Cross-compare with prior scales ───────────────────────────────────────
    try:
        with open(PREV_RESULTS) as f:
            prev = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        prev_fg      = [v["forgetting_pct"] for v in prev.values()]
        prev_spread  = max(prev_fg) - min(prev_fg)
        spread_dir   = ("grew"   if spread > prev_spread + 1 else
                        "shrank" if spread < prev_spread - 1 else "stable")
        prev_ranking = sorted(prev,    key=lambda k: prev[k]["forgetting_pct"])
        curr_ranking = sorted(results, key=lambda k: results[k]["forgetting_pct"])

        print(f"\n{'='*64}")
        print(f"vs 0.5B baseline ({PREV_RESULTS})")
        print(f"{'='*64}")
        print(f"{'Pair':<22} {'0.5B ctrl':>12} {MODEL_SLUG:>10}  Rank?")
        print("-" * 60)
        for k in sorted(set(list(prev.keys()) + list(results.keys()))):
            p   = prev.get(k, {}).get("forgetting_pct", float("nan"))
            c   = results.get(k, {}).get("forgetting_pct", float("nan"))
            pr  = prev_ranking.index(k) + 1 if k in prev_ranking else "?"
            cr  = curr_ranking.index(k) + 1 if k in curr_ranking else "?"
            match = "OK" if pr == cr else "NO"
            print(f"{k:<22} {p:>+11.2f}% {c:>+9.2f}%  #{pr}->{cr} {match}")

        cm_0 = prev.get("code->math",  {}).get("forgetting_pct")
        mc_0 = prev.get("math->code",  {}).get("forgetting_pct")
        cm_n = results.get("code->math",  {}).get("forgetting_pct")
        mc_n = results.get("math->code",  {}).get("forgetting_pct")

        print(f"\n0.5B spread: {prev_spread:.2f} pp  "
              f"-> current: {spread:.2f} pp  ({spread_dir})")
        print(f"Asymmetry code->math < math->code  "
              f"0.5B: {'YES' if cm_0 and mc_0 and cm_0 < mc_0 else 'NO'}")
        print(f"Asymmetry code->math < math->code  "
              f"now:  {'YES' if cm_n and mc_n and cm_n < mc_n else 'NO'}")
    except FileNotFoundError:
        print(f"\n(No {PREV_RESULTS} — copy from prior experiments for comparison)")

    results["_summary"] = {
        "model":       MODEL_ID,
        "num_chips":   num_chips,
        "batch":       BATCH_SIZE,
        "steps":       TRAIN_STEPS,
        "lr":          LR,
        "spread_pct":  round(spread, 4),
        "min_pair":    min_pair,
        "max_pair":    max_pair,
        "verdict":     verdict,
        "elapsed_min": round(elapsed, 1),
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {RESULTS_FILE}")
    gcs_backup(RESULTS_FILE)


if __name__ == "__main__":
    main()
