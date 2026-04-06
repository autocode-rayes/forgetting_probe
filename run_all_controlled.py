import os, sys, json, math, random, time, shutil

os.environ["HF_ENDPOINT"]            = "https://hf-mirror.com"
os.environ["HF_HOME"]                = "/root/autodl-tmp/hf_cache"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCHDYNAMO_DISABLE"]    = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from huggingface_hub import snapshot_download

# ── Identical protocol for ALL models ─────────────────────────────────────────
TOKEN_BUDGET = 50_000
MAX_LEN      = 256
TRAIN_N      = 175
EVAL_N       = 19
STEPS        = 200
BATCH        = 4
LR           = 2e-5
SEED         = 42
DTYPE        = torch.bfloat16
OUT_DIR      = "/root/autodl-tmp"

MODELS = [
    ("0.5B", "Qwen/Qwen2.5-0.5B", f"{OUT_DIR}/Qwen2.5-0.5B"),
    ("1.5B", "Qwen/Qwen2.5-1.5B", f"{OUT_DIR}/Qwen2.5-1.5B"),
    ("3B",   "Qwen/Qwen2.5-3B",   f"{OUT_DIR}/Qwen2.5-3B"),
    ("7B",   "Qwen/Qwen2.5-7B",   f"{OUT_DIR}/Qwen2.5-7B"),
    ("14B",  "Qwen/Qwen2.5-14B",  f"{OUT_DIR}/Qwen2.5-14B"),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def ensure_model(hf_id, local_path):
    safetensors = [f for f in os.listdir(local_path)
                   if f.endswith(".safetensors")] if os.path.isdir(local_path) else []
    if safetensors:
        print(f"  Model already at {local_path} ({len(safetensors)} shards)", flush=True)
        return
    print(f"  Downloading {hf_id} ...", flush=True)
    snapshot_download(hf_id, local_dir=local_path,
                      ignore_patterns=["*.msgpack", "*.h5", "flax*"])
    print(f"  Download complete", flush=True)


def build_chunks(texts, tok):
    ids = []
    for t in texts:
        ids.extend(tok(t, add_special_tokens=False)["input_ids"])
        if len(ids) >= TOKEN_BUDGET:
            break
    ids = ids[:TOKEN_BUDGET]
    return [torch.tensor(ids[i:i+MAX_LEN], dtype=torch.long)
            for i in range(0, len(ids) - MAX_LEN + 1, MAX_LEN)]


def make_batches(chunks, shuffle=True):
    ch = list(chunks)
    if shuffle:
        random.shuffle(ch)
    return [torch.stack(ch[i:i+BATCH]) for i in range(0, len(ch) - BATCH + 1, BATCH)]


def finetune(model, blist, label, input_dev):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    step = total = 0
    while step < STEPS:
        for b in blist:
            if step >= STEPS:
                break
            out = model(input_ids=b.to(input_dev), labels=b.to(input_dev))
            loss = out.loss
            if torch.isnan(loss):
                opt.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            total += loss.item()
            step += 1
            if step % 50 == 0:
                print(f"    {label} {step}/{STEPS}  loss={total/step:.4f}", flush=True)
    return step


@torch.no_grad()
def ppl(model, blist, input_dev):
    model.eval()
    nll = ntok = 0
    for b in blist:
        out = model(input_ids=b.to(input_dev), labels=b.to(input_dev))
        if torch.isnan(out.loss):
            continue
        nll  += out.loss.item() * b.numel()
        ntok += b.numel()
    return math.exp(nll / ntok) if ntok else float("inf")


# ── Data prep (do once; all Qwen2.5 share identical tokenizer) ────────────────

print("=" * 60, flush=True)
print("Downloading 0.5B for tokenizer + data prep", flush=True)
ensure_model("Qwen/Qwen2.5-0.5B", f"{OUT_DIR}/Qwen2.5-0.5B")

tok = AutoTokenizer.from_pretrained(f"{OUT_DIR}/Qwen2.5-0.5B", local_files_only=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("Loading datasets ...", flush=True)
ds_math = load_dataset("gsm8k", "main", split="train")
math_texts = [f"Question: {r['question']}\nAnswer: {r['answer']}" for r in ds_math]
ds_lit = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
lit_texts = [r["text"] for r in ds_lit if len(r["text"].strip()) > 80]
print(f"  math: {len(math_texts)} samples  lit: {len(lit_texts)} samples", flush=True)

# Fix the random state so every model sees identical data
random.seed(SEED)
torch.manual_seed(SEED)

random.shuffle(math_texts)
random.shuffle(lit_texts)

math_chunks = build_chunks(math_texts, tok)
random.shuffle(math_chunks)
lit_chunks  = build_chunks(lit_texts,  tok)
random.shuffle(lit_chunks)

math_train = math_chunks[:TRAIN_N]
math_eval  = math_chunks[TRAIN_N:TRAIN_N + EVAL_N]
lit_train  = lit_chunks[:TRAIN_N]

# Build batches once — same order for every model
ta = make_batches(math_train, shuffle=True)
tb = make_batches(lit_train,  shuffle=True)
ea = make_batches(math_eval,  shuffle=False)

batches_per_epoch = len(math_train) // BATCH
epochs_run        = STEPS / batches_per_epoch

print(f"\nData ready:", flush=True)
print(f"  math  train: {len(math_train)} chunks = {len(math_train)*MAX_LEN:,} tokens", flush=True)
print(f"  math  eval:  {len(math_eval)}  chunks = {len(math_eval)*MAX_LEN:,} tokens", flush=True)
print(f"  lit   train: {len(lit_train)}  chunks = {len(lit_train)*MAX_LEN:,} tokens", flush=True)
print(f"  batch={BATCH}, batches/epoch={batches_per_epoch}, "
      f"epochs={epochs_run:.2f} (identical for all models)", flush=True)

# ── Main loop ─────────────────────────────────────────────────────────────────

all_results = {}

for size, hf_id, local_path in MODELS:
    print(f"\n{'='*60}", flush=True)
    print(f"  MODEL: {size}  ({hf_id})", flush=True)
    print(f"{'='*60}", flush=True)

    try:
        ensure_model(hf_id, local_path)

        # Small models fit on one GPU; larger models need pipeline parallelism.
        # device_map="balanced" = single process, layers spread across GPUs —
        # NOT DDP (no gradient sync, effective batch is still BATCH=4).
        n_gpus = torch.cuda.device_count()
        if size in ("0.5B", "1.5B"):
            model = AutoModelForCausalLM.from_pretrained(
                local_path, dtype=DTYPE, local_files_only=True)
            model = model.to("cuda:0")
            input_dev = torch.device("cuda:0")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                local_path, dtype=DTYPE, device_map="balanced",
                local_files_only=True)
            input_dev = torch.device("cuda:0")

        model.config.pad_token_id = tok.eos_token_id
        # NO gradient checkpointing — identical to 0.5B baseline

        t0 = time.time()
        torch.manual_seed(SEED)   # reset before training for reproducibility

        steps_a = finetune(model, ta, "A=math", input_dev)
        pb = ppl(model, ea, input_dev)
        print(f"  ppl(math) before lit: {pb:.4f}", flush=True)

        steps_b = finetune(model, tb, "B=lit", input_dev)
        pa = ppl(model, ea, input_dev)
        print(f"  ppl(math) after  lit: {pa:.4f}", flush=True)

        fg      = (pa - pb) / pb * 100
        elapsed = (time.time() - t0) / 60
        print(f"  forgetting: {fg:+.2f}%  elapsed: {elapsed:.1f} min", flush=True)

        result = {
            "math->literature": {
                "domain_a": "math", "domain_b": "literature",
                "ppl_before":     round(pb, 4),
                "ppl_after":      round(pa, 4),
                "forgetting_pct": round(fg, 4),
            },
            "_config": {
                "model": hf_id, "size": size,
                "batch": BATCH, "steps": STEPS, "lr": LR,
                "optimizer": "AdamW_fp32",
                "grad_checkpointing": False,
                "token_budget": TOKEN_BUDGET, "max_len": MAX_LEN,
                "train_chunks": len(math_train),
                "eval_chunks":  len(math_eval),
                "train_tokens": len(math_train) * MAX_LEN,
                "eval_tokens":  len(math_eval)  * MAX_LEN,
                "batches_per_epoch": batches_per_epoch,
                "epochs": round(epochs_run, 2),
                "elapsed_min": round(elapsed, 1),
            },
        }
        fname = f"{OUT_DIR}/results_{size.replace('.','p')}_v2.json"
        with open(fname, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved -> {fname}", flush=True)
        all_results[size] = result

    except torch.cuda.OutOfMemoryError as e:
        print(f"  OOM: {e}", flush=True)
        all_results[size] = {"error": "OOM", "detail": str(e)}
        for i in range(torch.cuda.device_count()):
            torch.cuda.empty_cache()

    except Exception as e:
        import traceback
        print(f"  ERROR: {e}", flush=True)
        traceback.print_exc(file=sys.stdout)
        all_results[size] = {"error": str(e)}

    finally:
        # Free model from memory
        try:
            del model
        except NameError:
            pass
        for i in range(torch.cuda.device_count()):
            torch.cuda.empty_cache()

        # Delete model weights from disk to free space for next download
        # (keep 14B since it's the last and already occupied the space)
        if size != "14B" and os.path.isdir(local_path):
            print(f"  Removing {local_path} to free disk ...", flush=True)
            shutil.rmtree(local_path)

# ── Final comparison table ─────────────────────────────────────────────────────

print(f"\n{'='*75}", flush=True)
print("FINAL COMPARISON — math→literature  (AdamW fp32, batch=4, 200 steps)", flush=True)
print(f"{'='*75}", flush=True)
print(f"{'Scale':<7} {'PPL_before':>11} {'PPL_after':>11} {'Forgetting':>12} "
      f"{'Epochs':>8} {'Minutes':>8}", flush=True)
print(f"{'-'*75}", flush=True)

for size, _, _ in MODELS:
    r = all_results.get(size, {})
    if "error" in r:
        print(f"{size:<7}  ERROR: {r.get('error','?')}", flush=True)
        continue
    ml  = r["math->literature"]
    cfg = r["_config"]
    print(f"{size:<7} {ml['ppl_before']:>11.3f} {ml['ppl_after']:>11.3f} "
          f"{ml['forgetting_pct']:>+11.2f}% "
          f"{cfg['epochs']:>8.2f} {cfg['elapsed_min']:>8.1f}", flush=True)

print(f"\nProtocol confirmed identical across all runs:", flush=True)
print(f"  batch={BATCH}, steps={STEPS}, epochs={epochs_run:.2f}, "
      f"lr={LR}, optimizer=AdamW_fp32, grad_ckpt=False", flush=True)

with open(f"{OUT_DIR}/results_all_v2.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nCombined results -> {OUT_DIR}/results_all_v2.json", flush=True)
