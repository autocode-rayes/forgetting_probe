import os, sys

os.environ["HF_ENDPOINT"]            = "https://hf-mirror.com"
os.environ["HF_HOME"]                = "/root/autodl-tmp/hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCHDYNAMO_DISABLE"]    = "1"
os.environ["TRANSFORMERS_OFFLINE"]   = "1"

import json, math, random, time
import torch
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_PATH   = "/root/autodl-tmp/Qwen2.5-14B"
TOKEN_BUDGET = 50_000
MAX_LEN      = 256
TRAIN_N      = 175
EVAL_N       = 19
STEPS        = 200
BATCH        = 2
LR           = 2e-5
SEED         = 42
DTYPE        = torch.bfloat16
N_GPUS       = torch.cuda.device_count()
INPUT_DEV    = "cuda:0"
OUT          = "/root/autodl-tmp/results_14b_math_lit.json"

random.seed(SEED)
torch.manual_seed(SEED)
print(f"GPUs: {N_GPUS}")
for i in range(N_GPUS):
    print(f"  GPU{i}: {torch.cuda.get_device_name(i)}  "
          f"{torch.cuda.get_device_properties(i).total_memory // 1024**3}GB")
print(f"Model: {MODEL_PATH}  device_map=balanced  batch={BATCH}")


def vram():
    return " | ".join(
        f"G{i}:{torch.cuda.memory_allocated(i)/1024**3:.1f}GB" for i in range(N_GPUS))


def load_math():
    ds = load_dataset("gsm8k", "main", split="train")
    t = [f"Question: {r['question']}\nAnswer: {r['answer']}" for r in ds]
    print(f"  math: gsm8k {len(t)} samples")
    return t


def load_lit():
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    t = [r["text"] for r in ds if len(r["text"].strip()) > 80]
    print(f"  lit: wikitext-103-raw-v1 {len(t)} samples")
    return t


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


def finetune(model, blist, label):
    model.train()
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=LR)
    step = total = 0
    while step < STEPS:
        for b in blist:
            if step >= STEPS:
                break
            out = model(input_ids=b.to(INPUT_DEV), labels=b.to(INPUT_DEV))
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
                print(f"    {label} {step}/{STEPS}  loss={total/step:.4f}  [{vram()}]")


@torch.no_grad()
def ppl(model, blist):
    model.eval()
    nll = ntok = 0
    for b in blist:
        out = model(input_ids=b.to(INPUT_DEV), labels=b.to(INPUT_DEV))
        if torch.isnan(out.loss):
            continue
        nll += out.loss.item() * b.numel()
        ntok += b.numel()
    return math.exp(nll / ntok) if ntok else float("inf")


tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("Loading datasets...")
math_texts = load_math()
lit_texts = load_lit()
random.shuffle(math_texts)
random.shuffle(lit_texts)

print("\nTokenizing (50k tokens each)...")
math_chunks = build_chunks(math_texts, tok)
random.shuffle(math_chunks)
lit_chunks = build_chunks(lit_texts, tok)
random.shuffle(lit_chunks)

math_train = math_chunks[:TRAIN_N]
math_eval = math_chunks[TRAIN_N:TRAIN_N + EVAL_N]
lit_train = lit_chunks[:TRAIN_N]
print(f"  math: {len(math_train)} train, {len(math_eval)} eval")
print(f"  lit: {len(lit_train)} train")

ta = make_batches(math_train)
tb = make_batches(lit_train)
ea = make_batches(math_eval, shuffle=False)

print("\nLoading model (device_map=balanced)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=DTYPE, device_map="balanced", local_files_only=True)
model.config.pad_token_id = tok.eos_token_id
model.gradient_checkpointing_enable()
print(f"  loaded  [{vram()}]")

t0 = time.time()

# Phase A: fine-tune on math
finetune(model, ta, "A=math")
ppl_before = ppl(model, ea)
print(f"  ppl(math) before lit: {ppl_before:.3f}")

# Phase B: fine-tune on literature
finetune(model, tb, "B=literature")
ppl_after = ppl(model, ea)
print(f"  ppl(math) after lit:  {ppl_after:.3f}")

fg = (ppl_after - ppl_before) / ppl_before * 100
elapsed = (time.time() - t0) / 60
print(f"  forgetting: {fg:+.2f}%")
print(f"  elapsed: {elapsed:.1f} min")

results = {
    "math->literature": {
        "domain_a": "math",
        "domain_b": "literature",
        "ppl_before": round(ppl_before, 4),
        "ppl_after": round(ppl_after, 4),
        "forgetting_pct": round(fg, 4),
    },
    "_summary": {
        "model": MODEL_PATH,
        "n_gpus": N_GPUS,
        "batch": BATCH,
        "steps": STEPS,
        "lr": LR,
        "elapsed_min": round(elapsed, 1),
    },
}
with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved -> {OUT}")
