import os, sys, json, math, random, time, shutil

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def _load_dataset_ms(name, subset=None, split="train"):
    try:
        from modelscope.msdatasets import MsDataset
        kwargs = {"split": split}
        if subset:
            kwargs["subset_name"] = subset
        print(f"  Loading dataset {name} via ModelScope...", flush=True)
        return MsDataset.load(name, **kwargs)
    except Exception as e:
        print(f"  ModelScope dataset failed ({e}), trying HF...", flush=True)
        return load_dataset(name, subset, split=split)


def _resolve_model_path(model_id, local_path):
    if os.path.isdir(local_path) and any(
            f.endswith(".safetensors") for f in os.listdir(local_path)):
        print(f"  Model at {local_path}", flush=True)
        return local_path
    try:
        from modelscope import snapshot_download
        ms_id = model_id.split("/")[0].lower() + "/" + model_id.split("/")[1]
        print(f"  Downloading {ms_id} via ModelScope...", flush=True)
        path = snapshot_download(ms_id, local_dir=local_path)
        print(f"  Download complete -> {path}", flush=True)
        return path
    except Exception as e:
        print(f"  ModelScope failed ({e}), trying HF hub...", flush=True)
        from huggingface_hub import snapshot_download as hf_dl
        hf_dl(model_id, local_dir=local_path,
              ignore_patterns=["*.msgpack", "*.h5", "flax*"])
        return local_path

TOKEN_BUDGET = 50_000
MAX_LEN      = 256
TRAIN_N      = 175
EVAL_N       = 19
STEPS        = 200
LR           = 2e-5
SEED         = 42
DTYPE        = torch.bfloat16
OUT_DIR      = "/root/autodl-tmp"
OUT          = f"{OUT_DIR}/results_batchscale_2d.json"

MODELS = [
    ("0.5B", "Qwen/Qwen2.5-0.5B", f"{OUT_DIR}/Qwen2.5-0.5B"),
    ("1.5B", "Qwen/Qwen2.5-1.5B", f"{OUT_DIR}/Qwen2.5-1.5B"),
    ("3B",   "Qwen/Qwen2.5-3B",   f"{OUT_DIR}/Qwen2.5-3B"),
]

BATCH_SIZES = [32, 64]

# 7B results from previous sweep (to include in final table)
PREV_7B = {
    1:  {"ppl_before": 1.5298, "ppl_after": 1.5868, "forgetting_pct": 3.7275},
    2:  {"ppl_before": 1.5933, "ppl_after": 1.703,  "forgetting_pct": 6.8836},
    4:  {"ppl_before": 1.6319, "ppl_after": 1.8644, "forgetting_pct": 14.2423},
    8:  {"ppl_before": 1.8349, "ppl_after": 2.0265, "forgetting_pct": 10.4405},
    16: {"ppl_before": 1.9788, "ppl_after": 2.1527, "forgetting_pct": 8.7885},
}


def ensure_model(hf_id, local_path):
    return _resolve_model_path(hf_id, local_path)


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


# ── Data prep (once) ──────────────────────────────────────────────────────────

# Download smallest model for tokenizer
tok_path = ensure_model("Qwen/Qwen2.5-0.5B", f"{OUT_DIR}/Qwen2.5-0.5B")
tok = AutoTokenizer.from_pretrained(tok_path)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("Loading datasets...", flush=True)
ds_math = _load_dataset_ms("gsm8k", "main", split="train")
math_texts = [f"Question: {r['question']}\nAnswer: {r['answer']}" for r in ds_math]
ds_lit = _load_dataset_ms("wikitext", "wikitext-103-raw-v1", split="train")
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

print(f"  math train: {len(math_train)}  eval: {len(math_eval)}  lit train: {len(lit_train)}", flush=True)

# ── Main sweep ────────────────────────────────────────────────────────────────

all_results = {}

for size, hf_id, local_path in MODELS:
    print(f"\n{'#'*60}", flush=True)
    print(f"  MODEL: {size}  ({hf_id})", flush=True)
    print(f"{'#'*60}", flush=True)

    local_path = ensure_model(hf_id, local_path)
    model_results = {}

    for bs in BATCH_SIZES:
        bpe = len(math_train) // bs
        epochs = STEPS / bpe if bpe > 0 else 0
        key = f"batch_{bs}"

        print(f"\n  --- {size} batch={bs}  epochs={epochs:.2f} ---", flush=True)

        try:
            random.seed(SEED)
            ta = make_batches(math_train, bs, shuffle=True)
            tb = make_batches(lit_train, bs, shuffle=True)
            ea = make_batches(math_eval, 1, shuffle=False)

            if size in ("0.5B", "1.5B"):
                model = AutoModelForCausalLM.from_pretrained(
                    local_path, dtype=DTYPE)
                model = model.to("cuda:0")
                input_dev = torch.device("cuda:0")
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    local_path, dtype=DTYPE, device_map="balanced")
                input_dev = torch.device("cuda:0")

            model.config.pad_token_id = tok.eos_token_id
            try:
                model = torch.compile(model)
                print(f"  torch.compile OK", flush=True)
            except Exception as ce:
                print(f"  torch.compile skipped: {ce}", flush=True)
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

            model_results[key] = {
                "batch_size": bs,
                "epochs": round(epochs, 2),
                "ppl_before": round(pb, 4),
                "ppl_after": round(pa, 4),
                "forgetting_pct": round(fg, 4),
                "abs_gap": round(gap, 4),
                "elapsed_min": round(elapsed, 1),
            }

        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM at {size} batch={bs}", flush=True)
            model_results[key] = {"batch_size": bs, "error": "OOM"}

        except Exception as e:
            import traceback
            print(f"  ERROR at {size} batch={bs}: {e}", flush=True)
            traceback.print_exc(file=sys.stdout)
            model_results[key] = {"batch_size": bs, "error": str(e)}

        finally:
            try:
                del model
            except NameError:
                pass
            for i in range(torch.cuda.device_count()):
                torch.cuda.empty_cache()

    all_results[size] = model_results

    # Save after each model
    with open(OUT, "w") as f:
        json.dump(all_results, f, indent=2)

    # Delete model to free disk for next
    if size != "3B" and os.path.isdir(local_path):
        print(f"  Removing {local_path} ...", flush=True)
        shutil.rmtree(local_path)

# ── Final 2D tables ──────────────────────────────────────────────────────────

sizes = ["0.5B", "1.5B", "3B", "7B"]

print(f"\n{'='*75}", flush=True)
print("RELATIVE FORGETTING %  (math->literature, 200 steps, AdamW fp32)", flush=True)
print(f"{'='*75}", flush=True)
header = f"{'Scale':<7}" + "".join(f"{'bs='+str(b):>10}" for b in BATCH_SIZES)
print(header, flush=True)
print("-" * 75, flush=True)

peak_shifts = {}
for sz in sizes:
    row = f"{sz:<7}"
    best_bs = None
    best_fg = -999
    for bs in BATCH_SIZES:
        key = f"batch_{bs}"
        if sz == "7B":
            r = PREV_7B.get(bs, {})
            fg = r.get("forgetting_pct")
        else:
            r = all_results.get(sz, {}).get(key, {})
            fg = r.get("forgetting_pct")
        if fg is not None and "error" not in r:
            neg = "*" if fg < 0 else " "
            row += f"{fg:>+9.2f}%"
            if fg > best_fg:
                best_fg = fg
                best_bs = bs
        else:
            row += f"{'OOM':>10}"
    if best_bs:
        peak_shifts[sz] = best_bs
    print(row, flush=True)

print(f"\n{'='*75}", flush=True)
print("ABSOLUTE PPL GAP  (ppl_after - ppl_before)", flush=True)
print(f"{'='*75}", flush=True)
print(header, flush=True)
print("-" * 75, flush=True)

for sz in sizes:
    row = f"{sz:<7}"
    for bs in BATCH_SIZES:
        key = f"batch_{bs}"
        if sz == "7B":
            r = PREV_7B.get(bs, {})
            if "ppl_before" in r and "ppl_after" in r:
                gap = r["ppl_after"] - r["ppl_before"]
                row += f"{gap:>+10.4f}"
            else:
                row += f"{'?':>10}"
        else:
            r = all_results.get(sz, {}).get(key, {})
            gap = r.get("abs_gap")
            if gap is not None:
                row += f"{gap:>+10.4f}"
            else:
                row += f"{'OOM':>10}"
    print(row, flush=True)

print(f"\n{'='*75}", flush=True)
print("ANALYSIS", flush=True)
print(f"{'='*75}", flush=True)

# Peak batch per scale
print("\nPeak forgetting batch size per scale:", flush=True)
for sz, bs in peak_shifts.items():
    print(f"  {sz}: batch={bs}", flush=True)

peak_vals = list(peak_shifts.values())
if len(set(peak_vals)) > 1:
    print("  >>> V-shape peak SHIFTS across scales", flush=True)
else:
    print(f"  >>> Peak stable at batch={peak_vals[0]} across all scales", flush=True)

# Negative forgetting check
neg_cells = []
for sz in ["0.5B", "1.5B", "3B"]:
    for bs in BATCH_SIZES:
        key = f"batch_{bs}"
        r = all_results.get(sz, {}).get(key, {})
        fg = r.get("forgetting_pct")
        if fg is not None and fg < 0:
            neg_cells.append((sz, bs, fg))

if neg_cells:
    print("\nNEGATIVE FORGETTING detected:", flush=True)
    for sz, bs, fg in neg_cells:
        print(f"  {sz} batch={bs}: {fg:+.2f}%", flush=True)
else:
    print("\nNo negative forgetting in any cell.", flush=True)

print(f"\nSaved -> {OUT}", flush=True)
