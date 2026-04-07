"""
TPU Batch-Size Sweep for 32B / 72B — extends the 2D forgetting surface.
========================================================================
Matches the protocol of batchscale_2d.py / server_batchsweep_7b.py but
runs on Cloud TPU with SPMD sharding for large models.

Sweeps batch sizes [1, 2, 4, 8, 16] for each model in MODEL_IDS.
Domain pair: math -> literature (same as the controlled scale sweep).

Usage (v4-32, single or multi-host):
  python tpu_batchsweep_32b.py

  # Multi-host:
  gcloud compute tpus tpu-vm ssh MY_TPU --zone=us-central2-b --worker=all \
    --command="python tpu_batchsweep_32b.py"

Results saved to results_{model_slug}_batchsweep_tpu.json after each batch.
Set RESUME=1 to skip already-completed batch sizes.
"""

# ── 0. SPMD before any other torch_xla import ────────────────────────────────
import os
os.environ.setdefault("PJRT_DEVICE", "TPU")

import torch_xla.runtime as xr
xr.use_spmd()

import numpy as np
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs

num_chips = xr.global_runtime_device_count()
DEVICE    = xm.xla_device()

mesh = xs.Mesh(np.array(range(num_chips)), (num_chips,), ("model",))
xs.set_global_mesh(mesh)
print(f"TPU chips: {num_chips}  |  device: {DEVICE}", flush=True)

# ── 1. Imports ────────────────────────────────────────────────────────────────
import json, math, random, time, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ── 2. Config ─────────────────────────────────────────────────────────────────
# Models to sweep — comment out ones you don't need.
# Script runs them in order; skip to a specific model with MODEL_FILTER env var.
MODEL_IDS = [
    "Qwen/Qwen2.5-32B",
    "Qwen/Qwen2.5-72B",
]
# Override with MODEL_FILTER=Qwen2.5-32B to run only that model
_filter = os.environ.get("MODEL_FILTER", "")
if _filter:
    MODEL_IDS = [m for m in MODEL_IDS if _filter in m]
    print(f"Model filter: {_filter} -> {MODEL_IDS}", flush=True)

BATCH_SIZES  = [1, 2, 4, 8, 16, 32, 64]
TOKEN_BUDGET = 50_000
MAX_LEN      = 256
TRAIN_N      = 175
EVAL_N       = 19
STEPS        = 200
LR           = 2e-5
SEED         = 42
DTYPE        = torch.bfloat16
RESUME       = bool(int(os.environ.get("RESUME", "0")))

random.seed(SEED)
torch.manual_seed(SEED)

GCS_BUCKET = "gs://forgettingprobe-results"


def gcs_backup(local_file):
    import subprocess
    try:
        subprocess.run(
            ["gsutil", "cp", local_file, f"{GCS_BUCKET}/{local_file}"],
            check=True, capture_output=True)
        print(f"  Backed up -> {GCS_BUCKET}/{local_file}", flush=True)
    except Exception as e:
        print(f"  GCS backup failed (non-fatal): {e}", flush=True)


# ── 3. SPMD sharding ──────────────────────────────────────────────────────────
_COL_PARALLEL = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj",
                 "embed_tokens", "lm_head")
_ROW_PARALLEL = ("o_proj", "down_proj")


def apply_sharding(model):
    sharded = 0
    for name, param in model.named_parameters():
        if param.dim() < 2 or param.numel() < 65_536:
            xs.mark_sharding(param, mesh, (None,) * param.dim())
        elif any(k in name for k in _COL_PARALLEL):
            xs.mark_sharding(param, mesh, ("model", None))
            sharded += 1
        elif any(k in name for k in _ROW_PARALLEL):
            xs.mark_sharding(param, mesh, (None, "model"))
            sharded += 1
        else:
            xs.mark_sharding(param, mesh, (None,) * param.dim())
    print(f"  Sharded {sharded} matrices across {num_chips} chips", flush=True)


def load_model(model_id, tokenizer):
    print(f"  Loading {model_id} on CPU...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=DTYPE, low_cpu_mem_usage=True)
    model.config.pad_token_id = tokenizer.eos_token_id
    model = model.to(DEVICE)
    apply_sharding(model)
    xm.mark_step()
    return model


# ── 4. Data utils ─────────────────────────────────────────────────────────────
def build_chunks(texts, tok):
    ids = []
    while len(ids) < TOKEN_BUDGET:
        for t in texts:
            ids.extend(tok(t, add_special_tokens=False)["input_ids"])
            if len(ids) >= TOKEN_BUDGET:
                break
    ids = ids[:TOKEN_BUDGET]
    return [torch.tensor(ids[i:i + MAX_LEN], dtype=torch.long)
            for i in range(0, len(ids) - MAX_LEN + 1, MAX_LEN)]


def make_batches(chunks, bs, shuffle=True):
    ch = list(chunks)
    if shuffle:
        random.shuffle(ch)
    # Drop incomplete tail — matches batchscale_2d.py exactly.
    return [torch.stack(ch[i:i + bs]) for i in range(0, len(ch) - bs + 1, bs)]


# ── 5. Training & eval ────────────────────────────────────────────────────────
def finetune(model, batches, label):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    step, total_loss = 0, 0.0
    while step < STEPS:
        for batch in batches:
            if step >= STEPS:
                break
            batch = batch.to(DEVICE)
            xs.mark_sharding(batch, mesh, (None, None))
            out  = model(input_ids=batch, labels=batch)
            loss = out.loss
            if torch.isnan(loss):
                optimizer.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            xm.optimizer_step(optimizer)
            optimizer.zero_grad()
            total_loss += loss.item()
            step += 1
            if step % 50 == 0:
                print(f"    {label} step {step}/{STEPS}  "
                      f"loss={total_loss / step:.4f}", flush=True)
    xm.mark_step()


@torch.no_grad()
def ppl(model, batches):
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


# ── 6. Main sweep ─────────────────────────────────────────────────────────────
def run_model(model_id, math_chunks, lit_chunks):
    slug         = model_id.split("/")[-1].lower()
    results_file = f"results_{slug}_batchsweep_tpu.json"
    results      = {}

    # Resume: load any previously saved results
    if RESUME and os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)
        done = [k for k in results if not k.startswith("_") and "error" not in results[k]]
        print(f"  Resuming: {len(done)} batch sizes already done: {done}", flush=True)

    # Load tokenizer from the first model in the run (shared across Qwen2.5 family)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\n{'#'*60}")
    print(f"  {model_id}  ({num_chips} chips)")
    print(f"{'#'*60}")

    for bs in BATCH_SIZES:
        key = f"batch_{bs}"
        if key in results and "error" not in results[key]:
            print(f"\n  batch={bs} already done — skipping", flush=True)
            continue

        bpe    = TRAIN_N // bs
        epochs = STEPS / bpe if bpe > 0 else float("inf")
        print(f"\n  --- batch={bs}  epochs~{epochs:.2f} ---", flush=True)

        random.seed(SEED)
        torch.manual_seed(SEED)

        math_train = math_chunks[:TRAIN_N]
        math_eval  = math_chunks[TRAIN_N:TRAIN_N + EVAL_N]
        lit_train  = lit_chunks[:TRAIN_N]

        ta = make_batches(math_train, bs, shuffle=True)
        tb = make_batches(lit_train,  bs, shuffle=True)
        ea = make_batches(math_eval,  bs, shuffle=False)

        try:
            t0    = time.time()
            model = load_model(model_id, tokenizer)

            finetune(model, ta, f"A=math  bs={bs}")
            pb = ppl(model, ea)
            print(f"  ppl(math) before lit: {pb:.4f}", flush=True)

            finetune(model, tb, f"B=lit   bs={bs}")
            pa = ppl(model, ea)
            print(f"  ppl(math) after  lit: {pa:.4f}", flush=True)

            fg      = (pa - pb) / pb * 100
            gap     = pa - pb
            elapsed = (time.time() - t0) / 60

            neg_flag = " *** NEGATIVE ***" if fg < 0 else ""
            print(f"  forgetting: {fg:+.2f}%  abs_gap: {gap:+.4f}"
                  f"{neg_flag}  [{elapsed:.1f} min]", flush=True)

            results[key] = {
                "batch_size":    bs,
                "epochs":        round(epochs, 2),
                "ppl_before":    round(pb, 4),
                "ppl_after":     round(pa, 4),
                "forgetting_pct": round(fg, 4),
                "abs_gap":       round(gap, 4),
                "elapsed_min":   round(elapsed, 1),
            }

        except Exception as e:
            import traceback
            print(f"  ERROR at batch={bs}: {e}", flush=True)
            traceback.print_exc(file=sys.stdout)
            results[key] = {"batch_size": bs, "error": str(e)}

        finally:
            try:
                del model
            except NameError:
                pass
            xm.mark_step()

        # Save after each batch size — spot instances may be preempted
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved -> {results_file}", flush=True)
        gcs_backup(results_file)

    return results, results_file


def print_summary(model_id, results):
    slug = model_id.split("/")[-1]
    valid = {k: v for k, v in results.items()
             if not k.startswith("_") and "error" not in v}
    if not valid:
        print(f"  {slug}: no valid results"); return

    print(f"\n{'='*75}")
    print(f"BATCH SWEEP  [{model_id}]  (math->literature)")
    print(f"{'='*75}")
    print(f"{'Batch':>7} {'Epochs':>8} {'PPL_before':>12} "
          f"{'PPL_after':>12} {'Forgetting':>12} {'Gap':>10}")
    print("-" * 75)
    for bs in BATCH_SIZES:
        r = valid.get(f"batch_{bs}")
        if r:
            neg = " *" if r["forgetting_pct"] < 0 else ""
            print(f"{bs:>7} {r['epochs']:>8.2f} {r['ppl_before']:>12.4f} "
                  f"{r['ppl_after']:>12.4f} {r['forgetting_pct']:>+11.2f}%"
                  f"{r['abs_gap']:>+10.4f}{neg}")
        else:
            print(f"{bs:>7} {'—':>8} {'—':>12} {'—':>12} {'OOM/ERR':>12}")
    print(f"{'='*75}")

    fg_vals = [v["forgetting_pct"] for v in valid.values()]
    spread  = max(fg_vals) - min(fg_vals)
    peak_bs = max(valid, key=lambda k: valid[k]["forgetting_pct"])
    neg_bs  = [v["batch_size"] for v in valid.values() if v["forgetting_pct"] < 0]

    print(f"Spread: {spread:.2f} pp  |  Peak at: {peak_bs}")
    if neg_bs:
        print(f"NEGATIVE forgetting at batch sizes: {neg_bs}")

    # V-shape detection
    fg_seq = [valid.get(f"batch_{bs}", {}).get("forgetting_pct")
              for bs in BATCH_SIZES]
    fg_seq = [x for x in fg_seq if x is not None]
    if len(fg_seq) >= 3:
        peak_idx = fg_seq.index(max(fg_seq))
        if 0 < peak_idx < len(fg_seq) - 1:
            print(f"V-shape: YES — peak at bs index {peak_idx} "
                  f"(bs={BATCH_SIZES[peak_idx]})")
        elif peak_idx == 0:
            print("Shape: monotonically decreasing (peak at smallest bs)")
        else:
            print("Shape: monotonically increasing (no V-shape)")


def main():
    t0 = time.time()

    # ── Data (shared across models — same Qwen2.5 tokenizer family) ──────────
    print("\nLoading tokenizer (Qwen2.5-32B)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS[0])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading datasets...", flush=True)
    ds_math = load_dataset("gsm8k", "main", split="train")
    math_texts = [f"Question: {r['question']}\nAnswer: {r['answer']}"
                  for r in ds_math]
    ds_lit = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    lit_texts = [r["text"] for r in ds_lit if len(r["text"].strip()) > 80]
    print(f"  math: {len(math_texts)}  lit: {len(lit_texts)}", flush=True)

    random.seed(SEED)
    random.shuffle(math_texts)
    random.shuffle(lit_texts)

    print("Tokenizing (50k tokens each)...", flush=True)
    math_chunks = build_chunks(math_texts, tokenizer)
    random.shuffle(math_chunks)
    lit_chunks = build_chunks(lit_texts, tokenizer)
    random.shuffle(lit_chunks)
    print(f"  math: {len(math_chunks)} chunks  lit: {len(lit_chunks)} chunks",
          flush=True)

    # ── Sweep each model ─────────────────────────────────────────────────────
    all_results = {}
    for model_id in MODEL_IDS:
        results, results_file = run_model(model_id, math_chunks, lit_chunks)
        all_results[model_id] = results

    # ── Cross-model table ─────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print("RELATIVE FORGETTING %  (math->literature, 200 steps, AdamW fp32)")
    print(f"{'='*75}")
    print(f"{'Scale':<12}" + "".join(f"{'bs='+str(b):>11}" for b in BATCH_SIZES))
    print("-" * 75)
    for model_id in MODEL_IDS:
        slug = model_id.split("/")[-1]
        row  = f"{slug:<12}"
        for bs in BATCH_SIZES:
            r  = all_results.get(model_id, {}).get(f"batch_{bs}", {})
            fg = r.get("forgetting_pct")
            row += f"{fg:>+10.2f}%" if fg is not None else f"{'ERR':>11}"
        print(row)
    print(f"{'='*75}")

    # ── Compare with prior GPU data ───────────────────────────────────────────
    prior_files = {
        "14B": "results_14b_batchsweep.json",
        "7B":  "results_7b_batchsweep.json",
    }
    prior = {}
    for label, fname in prior_files.items():
        if os.path.exists(fname):
            with open(fname) as f:
                prior[label] = json.load(f)

    if prior:
        print(f"\n{'='*75}")
        print("SCALE TREND  (batch=4, math->literature)")
        print(f"{'='*75}")
        all_scales = list(prior.keys()) + [m.split("/")[-1] for m in MODEL_IDS]
        print(f"{'Scale':<12} {'Forgetting @ bs=4':>20}")
        print("-" * 35)
        for label, data in prior.items():
            fg = data.get("batch_4", {}).get("forgetting_pct")
            if fg is not None:
                print(f"{label:<12} {fg:>+19.2f}%")
        for model_id in MODEL_IDS:
            slug = model_id.split("/")[-1]
            fg = all_results.get(model_id, {}).get("batch_4", {}).get("forgetting_pct")
            if fg is not None:
                print(f"{slug:<12} {fg:>+19.2f}%  (TPU)")

    # ── Per-model detailed summaries ──────────────────────────────────────────
    for model_id in MODEL_IDS:
        print_summary(model_id, all_results[model_id])

    elapsed = (time.time() - t0) / 60
    print(f"\nTotal wall time: {elapsed:.1f} min")
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
