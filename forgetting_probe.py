"""
Catastrophic Forgetting Probe
Tests whether forgetting magnitude between domains is a real signal
about parametric structure vs. noise.

Setup: Qwen2.5-0.5B, 3 domains, 6 ordered pairs, 200 steps fine-tuning each.
"""

import json
import math
import random
import time
from itertools import permutations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID      = "Qwen/Qwen2.5-0.5B"
TRAIN_STEPS   = 200
TRAIN_CHUNKS  = 500   # chunks for fine-tuning
EVAL_CHUNKS   = 64    # chunks for perplexity eval
BATCH_SIZE    = 4
MAX_LEN       = 256
LR            = 2e-5
SEED          = 42
DTYPE         = torch.bfloat16   # bfloat16 stable on 4060; avoids fp16 NaN
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_FILE  = "results.json"

random.seed(SEED)
torch.manual_seed(SEED)

print(f"Device: {DEVICE}  ({torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'})")
print(f"Model:  {MODEL_ID}  dtype={DTYPE}\n")


# ── Dataset loaders ────────────────────────────────────────────────────────────
# We load many raw texts so the chunker has plenty of token material.
# Target: at least (TRAIN_CHUNKS + EVAL_CHUNKS) * MAX_LEN * 1.2 tokens.

TARGET_TOKENS = int((TRAIN_CHUNKS + EVAL_CHUNKS) * MAX_LEN * 1.5)


def _enough(ids):
    return len(ids) >= TARGET_TOKENS


def load_math_texts():
    """Returns a flat list of strings for the math domain."""
    # 1. gsm8k — grade-school math, always accessible
    print("  Loading math (gsm8k)...")
    try:
        ds = load_dataset("gsm8k", "main", split="train")
        texts = [f"Question: {r['question']}\nAnswer: {r['answer']}" for r in ds]
        print(f"  gsm8k: {len(texts)} samples")
        return texts
    except Exception as e:
        print(f"  gsm8k failed ({e})")

    # 2. math_qa
    print("  Trying math_qa...")
    try:
        ds = load_dataset("math_qa", split="train")
        texts = [f"Problem: {r['Problem']}\nRationale: {r['Rationale']}" for r in ds]
        print(f"  math_qa: {len(texts)} samples")
        return texts
    except Exception as e:
        print(f"  math_qa failed ({e})")

    # 3. aqua_rat
    print("  Trying aqua_rat...")
    try:
        ds = load_dataset("aqua_rat", "raw", split="train")
        texts = [f"{r['question']} {r['rationale']}" for r in ds]
        print(f"  aqua_rat: {len(texts)} samples")
        return texts
    except Exception as e:
        print(f"  aqua_rat failed ({e}), using wikitext-math-like subset as last resort...")

    ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
    texts = [r["text"] for r in ds if len(r["text"].strip()) > 80]
    random.shuffle(texts)
    return texts


def load_code_texts():
    """Returns a flat list of strings for the code domain."""
    # 1. code_search_net python
    print("  Loading code (code_search_net python)...")
    try:
        ds = load_dataset("code_search_net", "python", split="train")
        texts = [r["whole_func_string"] for r in ds]
        print(f"  code_search_net: {len(texts)} samples")
        return texts
    except Exception as e:
        print(f"  code_search_net failed ({e})")

    # 2. codeparrot/apps
    print("  Trying codeparrot/apps...")
    try:
        ds = load_dataset("codeparrot/apps", split="train", difficulties=["introductory"])
        texts = [r["solutions"] for r in ds if r.get("solutions")]
        print(f"  codeparrot/apps: {len(texts)} samples")
        return texts
    except Exception as e:
        print(f"  codeparrot/apps failed ({e})")

    # 3. mbpp
    print("  Trying mbpp...")
    try:
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
        texts = [f"# {r['text']}\n{r['code']}" for r in ds]
        print(f"  mbpp: {len(texts)} samples")
        return texts
    except Exception as e:
        print(f"  mbpp failed ({e}), falling back to wikitext...")

    ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
    texts = [r["text"] for r in ds if len(r["text"].strip()) > 80]
    random.shuffle(texts)
    return texts


def load_literature_texts():
    """Returns a flat list of strings for the literature domain."""
    print("  Loading literature (wikitext-103)...")
    try:
        ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
        texts = [r["text"] for r in ds if len(r["text"].strip()) > 80]
        print(f"  wikitext-103: {len(texts)} samples")
        return texts
    except Exception as e:
        print(f"  wikitext failed ({e})")

    print("  Trying bookcorpus...")
    try:
        ds = load_dataset("bookcorpus", split="train")
        texts = [r["text"] for r in ds]
        print(f"  bookcorpus: {len(texts)} samples")
        return texts
    except Exception as e:
        print(f"  bookcorpus failed ({e}), using pg19...")

    ds = load_dataset("pg19", split="train")
    texts = [r["text"][:5000] for r in ds]
    print(f"  pg19: {len(texts)} samples")
    return texts


# ── Tokenisation ───────────────────────────────────────────────────────────────

def build_chunks(texts, tokenizer, n_chunks, max_len):
    """
    Concatenate tokenized texts and slice into fixed-length blocks.
    Returns a list of (n_chunks) LongTensors of shape (max_len,).
    """
    all_ids = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        all_ids.extend(ids)
        if len(all_ids) >= n_chunks * max_len * 2:
            break

    chunks = []
    for i in range(0, len(all_ids) - max_len + 1, max_len):
        chunk = all_ids[i : i + max_len]
        if len(chunk) == max_len:
            chunks.append(torch.tensor(chunk, dtype=torch.long))
        if len(chunks) >= n_chunks:
            break

    return chunks


def make_batches(chunks, batch_size, shuffle=True):
    """Stack chunks into list of (B, L) tensors."""
    ch = list(chunks)
    if shuffle:
        random.shuffle(ch)
    return [
        torch.stack(ch[i : i + batch_size])
        for i in range(0, len(ch) - batch_size + 1, batch_size)
    ]


# ── Training ───────────────────────────────────────────────────────────────────

def finetune(model, batches, steps, lr, label=""):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
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
                print(f"    WARNING: NaN loss at {label} step {step}, skipping batch")
                optimizer.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            step += 1
            if step % 50 == 0:
                print(f"    {label} step {step}/{steps}  loss={total_loss/step:.4f}")

    return total_loss / max(step, 1)


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
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_nll / total_tokens)


def fresh_model(tokenizer):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=DTYPE,
        device_map=DEVICE,
    )
    model.config.pad_token_id = tokenizer.eos_token_id
    return model


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load raw text for each domain
    print("\n── Loading raw datasets ──")
    raw = {
        "math":       load_math_texts(),
        "code":       load_code_texts(),
        "literature": load_literature_texts(),
    }

    # Shuffle and build chunks
    print("\n── Tokenizing ──")
    need = TRAIN_CHUNKS + EVAL_CHUNKS
    domain_data = {}
    for name, texts in raw.items():
        random.shuffle(texts)
        chunks = build_chunks(texts, tokenizer, need, MAX_LEN)
        if len(chunks) < need:
            print(f"  WARNING: {name} only produced {len(chunks)}/{need} chunks — "
                  f"eval will use whatever is left")
        random.shuffle(chunks)
        domain_data[name] = {
            "train": chunks[:TRAIN_CHUNKS],
            "eval":  chunks[TRAIN_CHUNKS:],
        }
        print(f"  {name}: {len(domain_data[name]['train'])} train, "
              f"{len(domain_data[name]['eval'])} eval chunks")

    # Sanity-check: warn if any domain has 0 eval chunks
    for name, d in domain_data.items():
        if len(d["eval"]) == 0:
            raise RuntimeError(
                f"Domain '{name}' has 0 eval chunks. "
                "Reduce TRAIN_CHUNKS or EVAL_CHUNKS, or load more raw text."
            )

    # ── Run experiments ────────────────────────────────────────────────────────
    domains = list(domain_data.keys())
    results = {}

    print(f"\n{'='*62}")
    print("Running 6 ordered-pair forgetting experiments")
    print(f"{'='*62}")

    for domain_a, domain_b in permutations(domains, 2):
        pair_key = f"{domain_a}→{domain_b}"
        print(f"\n[{pair_key}]")

        train_a = make_batches(domain_data[domain_a]["train"], BATCH_SIZE)
        train_b = make_batches(domain_data[domain_b]["train"], BATCH_SIZE)
        eval_a  = make_batches(domain_data[domain_a]["eval"],  BATCH_SIZE, shuffle=False)

        print("  Loading fresh model...")
        model = fresh_model(tokenizer)

        print(f"  Fine-tune on A={domain_a} ({TRAIN_STEPS} steps)...")
        finetune(model, train_a, TRAIN_STEPS, LR, label=f"A={domain_a}")

        ppl_before = perplexity(model, eval_a)
        print(f"  PPL({domain_a}) before B-tuning: {ppl_before:.3f}")

        print(f"  Fine-tune on B={domain_b} ({TRAIN_STEPS} steps)...")
        finetune(model, train_b, TRAIN_STEPS, LR, label=f"B={domain_b}")

        ppl_after = perplexity(model, eval_a)
        print(f"  PPL({domain_a}) after  B-tuning: {ppl_after:.3f}")

        forgetting_pct = (ppl_after - ppl_before) / ppl_before * 100
        print(f"  Forgetting: {forgetting_pct:+.2f}%")

        results[pair_key] = {
            "domain_a":       domain_a,
            "domain_b":       domain_b,
            "ppl_before":     round(ppl_before, 4),
            "ppl_after":      round(ppl_after, 4),
            "forgetting_pct": round(forgetting_pct, 4),
        }

        del model
        torch.cuda.empty_cache()

    # ── Save raw results ───────────────────────────────────────────────────────
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = (time.time() - t0) / 60
    print(f"\nTotal time: {elapsed:.1f} min\n")

    sorted_pairs = sorted(results.items(), key=lambda x: x[1]["forgetting_pct"])
    fg_vals = [r["forgetting_pct"] for r in results.values()]
    min_pair = min(results, key=lambda k: results[k]["forgetting_pct"])
    max_pair = max(results, key=lambda k: results[k]["forgetting_pct"])
    spread   = max(fg_vals) - min(fg_vals)

    print(f"{'='*64}")
    print(f"{'Pair':<22} {'PPL_before':>10} {'PPL_after':>10} {'Forgetting':>12}")
    print(f"{'-'*64}")
    for pair_key, r in sorted_pairs:
        print(f"{pair_key:<22} {r['ppl_before']:>10.2f} {r['ppl_after']:>10.2f} "
              f"{r['forgetting_pct']:>+11.2f}%")
    print(f"{'='*64}")

    print(f"\nLowest forgetting:  {min_pair}  ({results[min_pair]['forgetting_pct']:+.2f}%)")
    print(f"Highest forgetting: {max_pair}  ({results[max_pair]['forgetting_pct']:+.2f}%)")
    print(f"Spread: {spread:.2f} pp")

    if spread > 10:
        verdict = "REAL signal — >10pp spread; parametric structure meaningfully drives forgetting magnitude."
    elif spread > 3:
        verdict = "WEAK signal — 3–10pp spread; suggestive but more runs needed."
    else:
        verdict = "NOISE — <3pp spread; forgetting magnitude appears domain-agnostic at this scale."

    print(f"\nVerdict: {verdict}")

    results["_summary"] = {
        "min_pair":    min_pair,
        "max_pair":    max_pair,
        "spread_pct":  round(spread, 4),
        "verdict":     verdict,
        "elapsed_min": round(elapsed, 1),
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
