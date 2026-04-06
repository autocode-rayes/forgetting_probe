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

BATCH_SIZES  = [1, 2, 4, 8, 16]


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
        nll += out.loss.item() * b.numel()
        ntok += b.numel()
    return math.exp(nll / ntok) if ntok else float("inf")


def run_sweep(size, hf_id, local_path, batch_list, use_balanced):
    """Run batch sweep for one model. Returns dict of results."""
    ensure_model(hf_id, local_path)

    # Tokenize with this model's tokenizer
    tok = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Same seed-controlled data prep as all other runs
    random.seed(SEED)
    torch.manual_seed(SEED)

    ds_math = load_dataset("gsm8k", "main", split="train")
    math_texts = [f"Question: {r['question']}\nAnswer: {r['answer']}" for r in ds_math]
    ds_lit = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    lit_texts = [r["text"] for r in ds_lit if len(r["text"].strip()) > 80]

    random.shuffle(math_texts)
    random.shuffle(lit_texts)

    math_chunks = build_chunks(math_texts, tok)
    random.shuffle(math_chunks)
    lit_chunks = build_chunks(lit_texts, tok)
    random.shuffle(lit_chunks)

    math_train = math_chunks[:TRAIN_N]
    math_eval  = math_chunks[TRAIN_N:TRAIN_N + EVAL_N]
    lit_train  = lit_chunks[:TRAIN_N]

    results = {}
    input_dev = torch.device("cuda:0")

    for bs in batch_list:
        bpe = len(math_train) // bs
        epochs = STEPS / bpe if bpe > 0 else 0
        key = f"batch_{bs}"

        print(f"\n  --- {size} batch={bs}  epochs={epochs:.2f} ---", flush=True)

        try:
            random.seed(SEED)
            ta = make_batches(math_train, bs, shuffle=True)
            tb = make_batches(lit_train, bs, shuffle=True)
            ea = make_batches(math_eval, bs, shuffle=False)

            if use_balanced:
                model = AutoModelForCausalLM.from_pretrained(
                    local_path, dtype=DTYPE, device_map="balanced",
                    local_files_only=True)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    local_path, dtype=DTYPE, local_files_only=True)
                model = model.to("cuda:0")

            model.config.pad_token_id = tok.eos_token_id
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
            print(f"  forgetting: {fg:+.2f}%  abs_gap: {gap:+.4f}{neg}  [{elapsed:.1f} min]", flush=True)

            results[key] = {
                "batch_size": bs,
                "epochs": round(epochs, 2),
                "ppl_before": round(pb, 4),
                "ppl_after": round(pa, 4),
                "forgetting_pct": round(fg, 4),
                "abs_gap": round(gap, 4),
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
            for i in range(torch.cuda.device_count()):
                torch.cuda.empty_cache()

    return results


# ── Step 1: 1.5B batch=16 on balanced (multi-GPU) ────────────────────────────

print("=" * 60, flush=True)
print("  1.5B batch=16 rerun (device_map=balanced)", flush=True)
print("=" * 60, flush=True)

ensure_model("Qwen/Qwen2.5-1.5B", f"{OUT_DIR}/Qwen2.5-1.5B")
r_1p5b_16 = run_sweep("1.5B", "Qwen/Qwen2.5-1.5B",
                       f"{OUT_DIR}/Qwen2.5-1.5B",
                       batch_list=[16], use_balanced=True)

# Delete 1.5B to free disk
shutil.rmtree(f"{OUT_DIR}/Qwen2.5-1.5B", ignore_errors=True)
print("  Removed 1.5B weights", flush=True)


# ── Step 2: Full 3B sweep ────────────────────────────────────────────────────

print(f"\n{'='*60}", flush=True)
print("  3B full batch sweep (device_map=balanced)", flush=True)
print("=" * 60, flush=True)

r_3b = run_sweep("3B", "Qwen/Qwen2.5-3B",
                  f"{OUT_DIR}/Qwen2.5-3B",
                  batch_list=BATCH_SIZES, use_balanced=True)


# ── Merge with existing results and save ──────────────────────────────────────

# Load previous 2D results if they exist
prev_path = f"{OUT_DIR}/results_batchscale_2d.json"
try:
    with open(prev_path) as f:
        all_results = json.load(f)
except FileNotFoundError:
    all_results = {}

# Patch 1.5B batch_16
if "1.5B" not in all_results:
    all_results["1.5B"] = {}
all_results["1.5B"]["batch_16"] = r_1p5b_16.get("batch_16", {"error": "failed"})

# Add 3B results
all_results["3B"] = r_3b

with open(prev_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved -> {prev_path}", flush=True)


# ── Print 2D table ───────────────────────────────────────────────────────────

PREV_7B = {
    1:  {"forgetting_pct": 3.7275, "abs_gap": 0.057},
    2:  {"forgetting_pct": 6.8836, "abs_gap": 0.1097},
    4:  {"forgetting_pct": 14.2423, "abs_gap": 0.2325},
    8:  {"forgetting_pct": 10.4405, "abs_gap": 0.1916},
    16: {"forgetting_pct": 8.7885, "abs_gap": 0.1739},
}

sizes = ["0.5B", "1.5B", "3B", "7B"]

print(f"\n{'='*70}", flush=True)
print("RELATIVE FORGETTING %", flush=True)
print(f"{'='*70}", flush=True)
print(f"{'Scale':<7}" + "".join(f"{'bs='+str(b):>11}" for b in BATCH_SIZES), flush=True)
print("-" * 70, flush=True)

for sz in sizes:
    row = f"{sz:<7}"
    for bs in BATCH_SIZES:
        if sz == "7B":
            fg = PREV_7B.get(bs, {}).get("forgetting_pct")
        else:
            r = all_results.get(sz, {}).get(f"batch_{bs}", {})
            fg = r.get("forgetting_pct")
        if fg is not None:
            row += f"{fg:>+10.2f}%"
        else:
            err = all_results.get(sz, {}).get(f"batch_{bs}", {}).get("error", "?")
            row += f"{err:>11}"
    print(row, flush=True)

print(f"\n{'='*70}", flush=True)
print("ABSOLUTE PPL GAP", flush=True)
print(f"{'='*70}", flush=True)
print(f"{'Scale':<7}" + "".join(f"{'bs='+str(b):>11}" for b in BATCH_SIZES), flush=True)
print("-" * 70, flush=True)

for sz in sizes:
    row = f"{sz:<7}"
    for bs in BATCH_SIZES:
        if sz == "7B":
            gap = PREV_7B.get(bs, {}).get("abs_gap")
        else:
            r = all_results.get(sz, {}).get(f"batch_{bs}", {})
            gap = r.get("abs_gap")
        if gap is not None:
            row += f"{gap:>+11.4f}"
        else:
            row += f"{'?':>11}"
    print(row, flush=True)

print("\nDONE", flush=True)
