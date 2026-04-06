import os, sys, json, math, random, time

os.environ["HF_ENDPOINT"]            = "https://hf-mirror.com"
os.environ["HF_HOME"]                = "/root/autodl-tmp/hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCHDYNAMO_DISABLE"]    = "1"
os.environ["TRANSFORMERS_OFFLINE"]   = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

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
    ("7B",  "Qwen/Qwen2.5-7B",  OUT_DIR + "/Qwen2.5-7B"),
    ("14B", "Qwen/Qwen2.5-14B", OUT_DIR + "/Qwen2.5-14B"),
]


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
        nll += out.loss.item() * b.numel()
        ntok += b.numel()
    return math.exp(nll / ntok) if ntok else float("inf")


# Data prep — same seed & tokenizer as 0.5B/1.5B/3B runs
tok = AutoTokenizer.from_pretrained(OUT_DIR + "/Qwen2.5-7B", local_files_only=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("Loading datasets...", flush=True)
ds_math = load_dataset("gsm8k", "main", split="train")
math_texts = [f"Question: {r['question']}\nAnswer: {r['answer']}" for r in ds_math]
ds_lit = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
lit_texts = [r["text"] for r in ds_lit if len(r["text"].strip()) > 80]

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

ta = make_batches(math_train, shuffle=True)
tb = make_batches(lit_train, shuffle=True)
ea = make_batches(math_eval, shuffle=False)

bpe = len(math_train) // BATCH
epochs = STEPS / bpe
print(f"Data: batch={BATCH}, batches/epoch={bpe}, epochs={epochs:.2f}", flush=True)

for size, hf_id, local_path in MODELS:
    print(f"\n{'='*60}", flush=True)
    print(f"  MODEL: {size}  ({hf_id})", flush=True)
    print(f"{'='*60}", flush=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            local_path, dtype=DTYPE, device_map="balanced", local_files_only=True)
        model.config.pad_token_id = tok.eos_token_id
        input_dev = torch.device("cuda:0")

        t0 = time.time()
        torch.manual_seed(SEED)

        steps_a = finetune(model, ta, "A=math", input_dev)
        pb = ppl(model, ea, input_dev)
        print(f"  ppl(math) before lit: {pb:.4f}", flush=True)

        steps_b = finetune(model, tb, "B=lit", input_dev)
        pa = ppl(model, ea, input_dev)
        print(f"  ppl(math) after  lit: {pa:.4f}", flush=True)

        fg = (pa - pb) / pb * 100
        elapsed = (time.time() - t0) / 60
        print(f"  forgetting: {fg:+.2f}%  elapsed: {elapsed:.1f} min", flush=True)

        result = {
            "math->literature": {
                "domain_a": "math", "domain_b": "literature",
                "ppl_before": round(pb, 4), "ppl_after": round(pa, 4),
                "forgetting_pct": round(fg, 4),
            },
            "_config": {
                "model": hf_id, "size": size, "batch": BATCH, "steps": STEPS,
                "lr": LR, "optimizer": "AdamW_fp32", "grad_checkpointing": False,
                "train_tokens": len(math_train) * MAX_LEN,
                "batches_per_epoch": bpe, "epochs": round(epochs, 2),
                "elapsed_min": round(elapsed, 1),
            },
        }
        fname = f"{OUT_DIR}/results_{size.replace('.', 'p')}_v2.json"
        with open(fname, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved -> {fname}", flush=True)

    except torch.cuda.OutOfMemoryError as e:
        print(f"  OOM: {e}", flush=True)
    except Exception as e:
        import traceback
        print(f"  ERROR: {e}", flush=True)
        traceback.print_exc(file=sys.stdout)
    finally:
        try:
            del model
        except NameError:
            pass
        for i in range(torch.cuda.device_count()):
            torch.cuda.empty_cache()

print("\nRESUME DONE", flush=True)
