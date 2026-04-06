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

TOKEN_BUDGET = 50_000
MAX_LEN      = 256
TRAIN_N      = 175
EVAL_N       = 19
STEPS        = 200
LR           = 2e-5
SEED         = 42
DTYPE        = torch.bfloat16
OUT_DIR      = "/root/autodl-tmp"
N_GPUS       = torch.cuda.device_count()

BATCH_SIZES = [1, 2, 4, 8, 16]

MODELS = [
    ("14B", "Qwen/Qwen2.5-14B", f"{OUT_DIR}/Qwen2.5-14B"),
    ("32B", "Qwen/Qwen2.5-32B", f"{OUT_DIR}/Qwen2.5-32B"),
]

print(f"GPUs: {N_GPUS}", flush=True)
for i in range(N_GPUS):
    print(f"  GPU{i}: {torch.cuda.get_device_name(i)}  "
          f"{torch.cuda.get_device_properties(i).total_memory // 1024**3}GB", flush=True)


def vram():
    return " | ".join(
        f"G{i}:{torch.cuda.memory_allocated(i)/1024**3:.1f}GB" for i in range(N_GPUS))


def ensure_model(hf_id, local_path):
    safetensors = [f for f in os.listdir(local_path)
                   if f.endswith(".safetensors")] if os.path.isdir(local_path) else []
    if safetensors:
        print(f"  Model at {local_path} ({len(safetensors)} shards)", flush=True)
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


def make_batches(chunks, bs, shuffle=True):
    ch = list(chunks)
    if shuffle:
        random.shuffle(ch)
    return [torch.stack(ch[i:i+bs]) for i in range(0, len(ch) - bs + 1, bs)]


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
                print(f"    {label} {step}/{STEPS}  loss={total/step:.4f}  [{vram()}]",
                      flush=True)
    return step


@torch.no_grad()
def ppl(model, blist, input_dev):
    model.eval()
    nll = ntok = 0
    for b in blist:
        out = model(input_ids=b.to(input_dev), labels=b.to(input_dev))
        if torch.isnan(out.loss):
            continue
        nll += out.loss.item() * b.numel()
        ntok += b.numel()
    return math.exp(nll / ntok) if ntok else float("inf")


# ── Data prep ─────────────────────────────────────────────────────────────────

# Download 14B first for tokenizer (all Qwen2.5 share same tokenizer)
ensure_model("Qwen/Qwen2.5-14B", f"{OUT_DIR}/Qwen2.5-14B")
tok = AutoTokenizer.from_pretrained(f"{OUT_DIR}/Qwen2.5-14B", local_files_only=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("Loading datasets...", flush=True)
ds_math = load_dataset("gsm8k", "main", split="train")
math_texts = [f"Question: {r['question']}\nAnswer: {r['answer']}" for r in ds_math]
ds_lit = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
lit_texts = [r["text"] for r in ds_lit if len(r["text"].strip()) > 80]
print(f"  math: {len(math_texts)}  lit: {len(lit_texts)}", flush=True)

random.seed(SEED)
torch.manual_seed(SEED)
random.shuffle(math_texts)
random.shuffle(lit_texts)

math_chunks = build_chunks(math_texts, tok)
random.shuffle(math_chunks)
lit_chunks = build_chunks(lit_texts, tok)
random.shuffle(lit_chunks)

math_train = math_chunks[:TRAIN_N]
math_eval  = math_chunks[TRAIN_N:TRAIN_N + EVAL_N]
lit_train  = lit_chunks[:TRAIN_N]

print(f"  math train: {len(math_train)}  eval: {len(math_eval)}  lit train: {len(lit_train)}",
      flush=True)

# ── Sweep loop ────────────────────────────────────────────────────────────────

for size, hf_id, local_path in MODELS:
    print(f"\n{'#'*60}", flush=True)
    print(f"  MODEL: {size}  ({hf_id})", flush=True)
    print(f"{'#'*60}", flush=True)

    ensure_model(hf_id, local_path)
    results = {}
    input_dev = torch.device("cuda:0")

    for bs in BATCH_SIZES:
        bpe = len(math_train) // bs
        epochs = STEPS / bpe if bpe > 0 else 0
        key = f"batch_{bs}"

        print(f"\n  --- {size} batch={bs}  epochs={epochs:.2f} ---", flush=True)

        try:
            random.seed(SEED)
            ta = make_batches(math_train, bs, shuffle=True)
            tb = make_batches(lit_train, bs, shuffle=True)
            ea = make_batches(math_eval, bs, shuffle=False)

            model = AutoModelForCausalLM.from_pretrained(
                local_path, dtype=DTYPE, device_map="balanced",
                local_files_only=True)
            model.config.pad_token_id = tok.eos_token_id
            # NO gradient checkpointing — match protocol
            print(f"  loaded  [{vram()}]", flush=True)

            t0 = time.time()
            torch.manual_seed(SEED)

            finetune(model, ta, f"A=math {size} bs={bs}", input_dev)
            pb = ppl(model, ea, input_dev)
            print(f"  ppl(math) before lit: {pb:.4f}", flush=True)

            finetune(model, tb, f"B=lit  {size} bs={bs}", input_dev)
            pa = ppl(model, ea, input_dev)
            print(f"  ppl(math) after  lit: {pa:.4f}", flush=True)

            fg = (pa - pb) / pb * 100
            gap = pa - pb
            elapsed = (time.time() - t0) / 60

            neg = " *** NEGATIVE ***" if fg < 0 else ""
            print(f"  forgetting: {fg:+.2f}%  abs_gap: {gap:+.4f}{neg}  [{elapsed:.1f} min]",
                  flush=True)

            results[key] = {
                "batch_size": bs,
                "epochs": round(epochs, 2),
                "ppl_before": round(pb, 4),
                "ppl_after": round(pa, 4),
                "forgetting_pct": round(fg, 4),
                "abs_gap": round(gap, 4),
                "optimizer": "AdamW_fp32",
                "elapsed_min": round(elapsed, 1),
            }

        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM at {size} batch={bs}: {e}", flush=True)
            results[key] = {"batch_size": bs, "error": "OOM"}

        except Exception as e:
            import traceback
            print(f"  ERROR at {size} batch={bs}: {e}", flush=True)
            traceback.print_exc(file=sys.stdout)
            results[key] = {"batch_size": bs, "error": str(e)}

        finally:
            try:
                del model
            except NameError:
                pass
            for i in range(N_GPUS):
                torch.cuda.empty_cache()

    # Save per-model results
    fname = f"{OUT_DIR}/results_{size.lower()}_batchsweep.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved -> {fname}", flush=True)

    # Keep models on disk (150GB available)

# ── Final table ───────────────────────────────────────────────────────────────

print(f"\n{'='*75}", flush=True)
print("RELATIVE FORGETTING %  (math->literature, 200 steps, AdamW fp32)", flush=True)
print(f"{'='*75}", flush=True)
print(f"{'Scale':<7}" + "".join(f"{'bs='+str(b):>11}" for b in BATCH_SIZES), flush=True)
print("-" * 75, flush=True)

all_model_results = {}
for size, _, _ in MODELS:
    fname = f"{OUT_DIR}/results_{size.lower()}_batchsweep.json"
    try:
        with open(fname) as f:
            all_model_results[size] = json.load(f)
    except FileNotFoundError:
        all_model_results[size] = {}

for size in ["14B", "32B"]:
    row = f"{size:<7}"
    best_bs = None
    best_fg = -999
    for bs in BATCH_SIZES:
        key = f"batch_{bs}"
        r = all_model_results.get(size, {}).get(key, {})
        fg = r.get("forgetting_pct")
        if fg is not None and "error" not in r:
            row += f"{fg:>+10.2f}%"
            if fg > best_fg:
                best_fg = fg
                best_bs = bs
        else:
            row += f"{'OOM':>11}"
    print(row, flush=True)

print(f"\n{'='*75}", flush=True)
print("ABSOLUTE PPL GAP  (ppl_after - ppl_before)", flush=True)
print(f"{'='*75}", flush=True)
print(f"{'Scale':<7}" + "".join(f"{'bs='+str(b):>11}" for b in BATCH_SIZES), flush=True)
print("-" * 75, flush=True)

for size in ["14B", "32B"]:
    row = f"{size:<7}"
    for bs in BATCH_SIZES:
        key = f"batch_{bs}"
        r = all_model_results.get(size, {}).get(key, {})
        gap = r.get("abs_gap")
        if gap is not None:
            row += f"{gap:>+11.4f}"
        else:
            row += f"{'OOM':>11}"
    print(row, flush=True)

# Analysis
print(f"\n{'='*75}", flush=True)
print("ANALYSIS", flush=True)
print(f"{'='*75}", flush=True)

for size in ["14B", "32B"]:
    best_bs = None
    best_fg = -999
    has_neg = []
    for bs in BATCH_SIZES:
        key = f"batch_{bs}"
        r = all_model_results.get(size, {}).get(key, {})
        fg = r.get("forgetting_pct")
        if fg is not None and "error" not in r:
            if fg > best_fg:
                best_fg = fg
                best_bs = bs
            if fg < 0:
                has_neg.append((bs, fg))
    if best_bs:
        print(f"  {size}: peak at batch={best_bs} ({best_fg:+.2f}%)", flush=True)
    if has_neg:
        for bs, fg in has_neg:
            print(f"  {size}: NEGATIVE forgetting at batch={bs} ({fg:+.2f}%)", flush=True)
    else:
        print(f"  {size}: no negative forgetting", flush=True)

print("\nALL DONE", flush=True)
