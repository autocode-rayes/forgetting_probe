"""
Catastrophic Forgetting Probe — Token-Controlled (Confound Ruling #1)
Exactly 50,000 tokens per domain, truncate/repeat to hit budget precisely.
Compares results against results.json to check if ranking is a data artifact.
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
TOKEN_BUDGET  = 50_000          # exactly this many tokens per domain
MAX_LEN       = 256             # tokens per chunk; 50000//256 = 195 chunks
TRAIN_CHUNKS  = 175             # 175 * 256 = 44,800 training tokens
EVAL_CHUNKS   = 19              # 19  * 256 = 4,864  eval tokens  (175+19=194 ≤ 195)
TRAIN_STEPS   = 200
BATCH_SIZE    = 4
LR            = 2e-5
SEED          = 42
DTYPE         = torch.bfloat16
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_FILE  = "results_controlled.json"
PREV_RESULTS  = "results.json"

random.seed(SEED)
torch.manual_seed(SEED)

print(f"Device: {DEVICE}  ({torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'})")
print(f"Model:  {MODEL_ID}  dtype={DTYPE}")
print(f"Token budget per domain: {TOKEN_BUDGET:,}  ({TRAIN_CHUNKS} train + {EVAL_CHUNKS} eval chunks)\n")


# ── Dataset loaders (same fallback chain as before) ───────────────────────────

def load_math_texts():
    print("  Loading math (gsm8k primary)...")
    try:
        ds = load_dataset("gsm8k", "main", split="train")
        texts = [f"Question: {r['question']}\nAnswer: {r['answer']}" for r in ds]
        print(f"  gsm8k: {len(texts)} samples")
        return texts
    except Exception as e:
        print(f"  gsm8k failed ({e}), trying math_qa...")
    try:
        ds = load_dataset("math_qa", split="train")
        texts = [f"Problem: {r['Problem']}\nRationale: {r['Rationale']}" for r in ds]
        print(f"  math_qa: {len(texts)} samples")
        return texts
    except Exception as e:
        print(f"  math_qa failed ({e}), using wikitext fallback...")
    ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
    texts = [r["text"] for r in ds if len(r["text"].strip()) > 80]
    return texts


def load_code_texts():
    print("  Loading code (code_search_net python primary)...")
    try:
        ds = load_dataset("code_search_net", "python", split="train")
        texts = [r["whole_func_string"] for r in ds]
        print(f"  code_search_net: {len(texts)} samples")
        return texts
    except Exception as e:
        print(f"  code_search_net failed ({e}), trying mbpp...")
    try:
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
        texts = [f"# {r['text']}\n{r['code']}" for r in ds]
        print(f"  mbpp: {len(texts)} samples")
        return texts
    except Exception as e:
        print(f"  mbpp failed ({e}), using wikitext fallback...")
    ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
    texts = [r["text"] for r in ds if len(r["text"].strip()) > 80]
    return texts


def load_literature_texts():
    print("  Loading literature (wikitext-103-raw-v1 primary)...")
    try:
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        texts = [r["text"] for r in ds if len(r["text"].strip()) > 80]
        print(f"  wikitext-103-raw-v1: {len(texts)} samples")
        return texts
    except Exception as e:
        print(f"  wikitext-103-raw-v1 failed ({e}), trying wikitext-103-v1...")
    try:
        ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
        texts = [r["text"] for r in ds if len(r["text"].strip()) > 80]
        print(f"  wikitext-103-v1: {len(texts)} samples")
        return texts
    except Exception as e:
        print(f"  wikitext failed ({e}), using bookcorpus...")
    ds = load_dataset("bookcorpus", split="train")
    texts = [r["text"] for r in ds]
    return texts


# ── Token-controlled chunking ──────────────────────────────────────────────────

def build_token_controlled_chunks(texts, tokenizer, token_budget, max_len):
    """
    Concatenate tokenized texts, repeat if needed, truncate to exactly
    token_budget tokens, then slice into fixed max_len blocks.
    """
    all_ids = []
    while len(all_ids) < token_budget:
        for text in texts:
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            all_ids.extend(ids)
            if len(all_ids) >= token_budget:
                break
    all_ids = all_ids[:token_budget]   # exactly token_budget tokens

    chunks = []
    for i in range(0, len(all_ids) - max_len + 1, max_len):
        chunks.append(torch.tensor(all_ids[i : i + max_len], dtype=torch.long))

    actual_tokens = len(chunks) * max_len
    print(f"  Token budget used: {token_budget:,} → {len(chunks)} chunks × {max_len} = {actual_tokens:,} tokens")
    return chunks


def make_batches(chunks, batch_size, shuffle=True):
    ch = list(chunks)
    if shuffle:
        random.shuffle(ch)
    return [
        torch.stack(ch[i : i + batch_size])
        for i in range(0, len(ch) - batch_size + 1, batch_size)
    ]


# ── Training & eval ────────────────────────────────────────────────────────────

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
    return math.exp(total_nll / total_tokens) if total_tokens > 0 else float("inf")


def fresh_model(tokenizer):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=DTYPE, device_map=DEVICE
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

    print("\n── Loading datasets ──")
    raw = {
        "math":       load_math_texts(),
        "code":       load_code_texts(),
        "literature": load_literature_texts(),
    }

    print("\n── Building token-controlled chunks (50k tokens each) ──")
    domain_data = {}
    for name, texts in raw.items():
        random.shuffle(texts)
        all_chunks = build_token_controlled_chunks(texts, tokenizer, TOKEN_BUDGET, MAX_LEN)
        random.shuffle(all_chunks)
        domain_data[name] = {
            "train": all_chunks[:TRAIN_CHUNKS],
            "eval":  all_chunks[TRAIN_CHUNKS : TRAIN_CHUNKS + EVAL_CHUNKS],
        }
        print(f"  {name}: {len(domain_data[name]['train'])} train chunks, "
              f"{len(domain_data[name]['eval'])} eval chunks")

    for name, d in domain_data.items():
        if len(d["eval"]) == 0:
            raise RuntimeError(f"Domain '{name}' has 0 eval chunks — increase TOKEN_BUDGET")

    domains = list(domain_data.keys())
    results = {}

    print(f"\n{'='*62}")
    print("Running 6 ordered-pair experiments (token-controlled)")
    print(f"{'='*62}")

    for domain_a, domain_b in permutations(domains, 2):
        pair_key = f"{domain_a}→{domain_b}"
        print(f"\n[{pair_key}]")

        train_a = make_batches(domain_data[domain_a]["train"], BATCH_SIZE)
        train_b = make_batches(domain_data[domain_b]["train"], BATCH_SIZE)
        eval_a  = make_batches(domain_data[domain_a]["eval"],  BATCH_SIZE, shuffle=False)

        print("  Loading fresh model...")
        model = fresh_model(tokenizer)

        finetune(model, train_a, TRAIN_STEPS, LR, label=f"A={domain_a}")
        ppl_before = perplexity(model, eval_a)
        print(f"  PPL({domain_a}) before B: {ppl_before:.3f}")

        finetune(model, train_b, TRAIN_STEPS, LR, label=f"B={domain_b}")
        ppl_after = perplexity(model, eval_a)
        print(f"  PPL({domain_a}) after  B: {ppl_after:.3f}")

        fg = (ppl_after - ppl_before) / ppl_before * 100
        print(f"  Forgetting: {fg:+.2f}%")

        results[pair_key] = {
            "domain_a":       domain_a,
            "domain_b":       domain_b,
            "ppl_before":     round(ppl_before, 4),
            "ppl_after":      round(ppl_after, 4),
            "forgetting_pct": round(fg, 4),
        }

        del model
        torch.cuda.empty_cache()

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    elapsed = (time.time() - t0) / 60

    # ── Print controlled results table ────────────────────────────────────────
    sorted_pairs = sorted(results.items(), key=lambda x: x[1]["forgetting_pct"])
    fg_vals  = [r["forgetting_pct"] for r in results.values()]
    min_pair = min(results, key=lambda k: results[k]["forgetting_pct"])
    max_pair = max(results, key=lambda k: results[k]["forgetting_pct"])
    spread   = max(fg_vals) - min(fg_vals)

    print(f"\nTotal time: {elapsed:.1f} min\n")
    print(f"{'='*64}")
    print(f"CONTROLLED RESULTS (50k tokens/domain)")
    print(f"{'='*64}")
    print(f"{'Pair':<22} {'PPL_before':>10} {'PPL_after':>10} {'Forgetting':>12}")
    print(f"{'-'*64}")
    for k, r in sorted_pairs:
        print(f"{k:<22} {r['ppl_before']:>10.2f} {r['ppl_after']:>10.2f} {r['forgetting_pct']:>+11.2f}%")
    print(f"{'='*64}")
    print(f"Spread: {spread:.2f} pp  |  Lowest: {min_pair}  |  Highest: {max_pair}")

    # ── Cross-compare with previous uncontrolled results ──────────────────────
    print(f"\n{'='*64}")
    print("COMPARISON: Uncontrolled vs Token-Controlled")
    print(f"{'='*64}")
    try:
        with open(PREV_RESULTS) as f:
            prev = json.load(f)
        prev = {k: v for k, v in prev.items() if not k.startswith("_")}
        prev_fg = [v["forgetting_pct"] for v in prev.values()]
        ctrl_fg = [v["forgetting_pct"] for v in results.values()]

        prev_ranking = sorted(prev, key=lambda k: prev[k]["forgetting_pct"])
        ctrl_ranking = sorted(results, key=lambda k: results[k]["forgetting_pct"])

        print(f"\n{'Pair':<22} {'Uncontrolled':>14} {'Controlled':>12}  Rank match?")
        print(f"{'-'*64}")
        all_pairs = sorted(set(list(prev.keys()) + list(results.keys())))
        for k in all_pairs:
            p = prev.get(k, {}).get("forgetting_pct", float("nan"))
            c = results.get(k, {}).get("forgetting_pct", float("nan"))
            pr = prev_ranking.index(k) + 1 if k in prev_ranking else "?"
            cr = ctrl_ranking.index(k) + 1 if k in ctrl_ranking else "?"
            match = "OK" if pr == cr else "NO"
            print(f"{k:<22} {p:>+13.2f}% {c:>+11.2f}%  #{pr}->{cr} {match}")

        ranking_stable = prev_ranking == ctrl_ranking
        ctrl_spread    = max(ctrl_fg) - min(ctrl_fg)
        prev_spread    = max(prev_fg) - min(prev_fg)

        print(f"\nPrev spread:  {prev_spread:.2f} pp")
        print(f"Ctrl spread:  {ctrl_spread:.2f} pp")
        print(f"Ranking stable: {'YES' if ranking_stable else 'NO — some reordering'}")

        math_lit_key = "math→literature"
        still_highest = (max_pair == math_lit_key)
        above_10pp    = ctrl_spread > 10

        print(f"\nmathematics→literature still highest forgetting: {'YES' if still_highest else 'NO'}")
        print(f"Spread still >10pp: {'YES' if above_10pp else 'NO'}")

        if ranking_stable and above_10pp:
            verdict = "CONFIRMED: ranking is stable and spread >10pp. The signal is real, not a data-volume artifact."
        elif above_10pp and not ranking_stable:
            verdict = "MIXED: spread >10pp but ranking shifted. Signal is real but magnitude is noisy."
        elif ranking_stable and not above_10pp:
            verdict = "WEAK: ranking held but spread dropped <10pp. May be partially a data artifact."
        else:
            verdict = "INCONCLUSIVE: both ranking and spread changed. Likely confounded."

        print(f"\nVerdict: {verdict}")

        results["_summary"] = {
            "min_pair":       min_pair,
            "max_pair":       max_pair,
            "spread_pct":     round(spread, 4),
            "ranking_stable": ranking_stable,
            "verdict":        verdict,
            "elapsed_min":    round(elapsed, 1),
        }

    except FileNotFoundError:
        print(f"  {PREV_RESULTS} not found — skipping cross-comparison")
        results["_summary"] = {
            "min_pair":    min_pair,
            "max_pair":    max_pair,
            "spread_pct":  round(spread, 4),
            "elapsed_min": round(elapsed, 1),
        }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
