import os, sys, json, math, random, time

os.environ["HF_ENDPOINT"]            = "https://hf-mirror.com"
os.environ["HF_HOME"]                = "/root/autodl-tmp/hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCHDYNAMO_DISABLE"]    = "1"
os.environ["TRANSFORMERS_OFFLINE"]   = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_PATH   = "/root/autodl-tmp/Qwen2.5-7B"
TOKEN_BUDGET = 50_000
MAX_LEN      = 256
TRAIN_N      = 175
EVAL_N       = 19
STEPS        = 200
LR           = 2e-5
SEED         = 42
DTYPE        = torch.bfloat16
OUT          = "/root/autodl-tmp/results_7b_batchsweep.json"

BATCH_SIZES  = [1, 2, 4, 8, 16, 32]


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


# ── Data prep (once, deterministic) ──────────────────────────────────────────

tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
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

print(f"  math train: {len(math_train)} chunks  eval: {len(math_eval)} chunks", flush=True)
print(f"  lit  train: {len(lit_train)} chunks", flush=True)

# ── Sweep ─────────────────────────────────────────────────────────────────────

results = {}
input_dev = torch.device("cuda:0")

for bs in BATCH_SIZES:
    bpe = len(math_train) // bs
    epochs = STEPS / bpe if bpe > 0 else 0
    key = f"batch_{bs}"

    print(f"\n{'='*60}", flush=True)
    print(f"  BATCH={bs}  batches/epoch={bpe}  epochs={epochs:.2f}", flush=True)
    print(f"{'='*60}", flush=True)

    try:
        # Rebuild batches with this batch size (same underlying chunk order)
        random.seed(SEED)
        ta = make_batches(math_train, bs, shuffle=True)
        tb = make_batches(lit_train, bs, shuffle=True)
        ea = make_batches(math_eval, bs, shuffle=False)

        # Fresh model each run
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, dtype=DTYPE, device_map="balanced", local_files_only=True)
        model.config.pad_token_id = tok.eos_token_id

        t0 = time.time()
        torch.manual_seed(SEED)

        steps_a = finetune(model, ta, f"A=math bs={bs}", input_dev)
        pb = ppl(model, ea, input_dev)
        print(f"  ppl(math) before lit: {pb:.4f}", flush=True)

        steps_b = finetune(model, tb, f"B=lit  bs={bs}", input_dev)
        pa = ppl(model, ea, input_dev)
        print(f"  ppl(math) after  lit: {pa:.4f}", flush=True)

        fg = (pa - pb) / pb * 100
        elapsed = (time.time() - t0) / 60

        flag = " *** NEGATIVE ***" if fg < 0 else ""
        print(f"  forgetting: {fg:+.2f}%{flag}  elapsed: {elapsed:.1f} min", flush=True)

        results[key] = {
            "batch_size": bs,
            "batches_per_epoch": bpe,
            "epochs": round(epochs, 2),
            "ppl_before": round(pb, 4),
            "ppl_after": round(pa, 4),
            "forgetting_pct": round(fg, 4),
            "elapsed_min": round(elapsed, 1),
        }

    except torch.cuda.OutOfMemoryError as e:
        print(f"  OOM at batch={bs}: {e}", flush=True)
        results[key] = {"batch_size": bs, "error": "OOM"}

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
        for i in range(torch.cuda.device_count()):
            torch.cuda.empty_cache()

    # Save after each run
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)

# ── Final table ───────────────────────────────────────────────────────────────

print(f"\n{'='*75}", flush=True)
print("BATCH SIZE SWEEP — math→literature  Qwen2.5-7B  (200 steps, AdamW fp32)", flush=True)
print(f"{'='*75}", flush=True)
print(f"{'Batch':>6} {'Epochs':>8} {'PPL_before':>11} {'PPL_after':>11} {'Forgetting':>12} {'Flag':>10}", flush=True)
print(f"{'-'*75}", flush=True)

for bs in BATCH_SIZES:
    key = f"batch_{bs}"
    r = results.get(key, {})
    if "error" in r:
        print(f"{bs:>6}  ERROR: {r['error']}", flush=True)
        continue
    flag = "NEGATIVE" if r["forgetting_pct"] < 0 else ""
    print(f"{bs:>6} {r['epochs']:>8.2f} {r['ppl_before']:>11.4f} {r['ppl_after']:>11.4f} "
          f"{r['forgetting_pct']:>+11.2f}% {flag:>10}", flush=True)

print(f"\nSaved -> {OUT}", flush=True)
