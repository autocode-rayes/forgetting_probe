import os, sys

os.environ["HF_ENDPOINT"]            = "https://hf-mirror.com"
os.environ["HF_HOME"]                = "/root/autodl-tmp/hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCHDYNAMO_DISABLE"]    = "1"

import json, math, random, time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_PATH    = "/root/autodl-tmp/Qwen2.5-1.5B"
TOKEN_BUDGET  = 50_000
MAX_LEN       = 256
TRAIN_N       = 175
EVAL_N        = 19
STEPS         = 200
BATCH_PER_GPU = 4
LR            = 2e-5
SEED          = 42
DTYPE         = torch.bfloat16
OUT           = "/root/autodl-tmp/results_1p5b_math_lit.json"


def setup():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def is_main():
    return dist.get_rank() == 0


class ChunkDataset(Dataset):
    def __init__(self, chunks):
        self.chunks = chunks
    def __len__(self):
        return len(self.chunks)
    def __getitem__(self, idx):
        return self.chunks[idx]


def load_math():
    ds = load_dataset("gsm8k", "main", split="train")
    return [f"Question: {r['question']}\nAnswer: {r['answer']}" for r in ds]


def load_lit():
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    return [r["text"] for r in ds if len(r["text"].strip()) > 80]


def build_chunks(texts, tok):
    ids = []
    for t in texts:
        ids.extend(tok(t, add_special_tokens=False)["input_ids"])
        if len(ids) >= TOKEN_BUDGET:
            break
    ids = ids[:TOKEN_BUDGET]
    return [torch.tensor(ids[i:i+MAX_LEN], dtype=torch.long)
            for i in range(0, len(ids) - MAX_LEN + 1, MAX_LEN)]


def finetune(model, dataloader, local_rank, label):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    step = total = 0
    epoch = 0
    while step < STEPS:
        dataloader.sampler.set_epoch(epoch)
        epoch += 1
        for batch in dataloader:
            if step >= STEPS:
                break
            batch = batch.to(local_rank)
            out = model(input_ids=batch, labels=batch)
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
            if step % 50 == 0 and is_main():
                vram = torch.cuda.memory_allocated() / 1024**3
                print(f"    {label} {step}/{STEPS}  loss={total/step:.4f}  vram={vram:.1f}GB")


@torch.no_grad()
def ppl(model, dataloader, local_rank):
    model.eval()
    nll = ntok = 0
    for batch in dataloader:
        batch = batch.to(local_rank)
        out = model(input_ids=batch, labels=batch)
        if torch.isnan(out.loss):
            continue
        nll += out.loss.item() * batch.numel()
        ntok += batch.numel()
    # all-reduce across GPUs
    nll_t = torch.tensor([nll], device=local_rank, dtype=torch.float64)
    ntok_t = torch.tensor([ntok], device=local_rank, dtype=torch.float64)
    dist.all_reduce(nll_t)
    dist.all_reduce(ntok_t)
    return math.exp(nll_t.item() / ntok_t.item()) if ntok_t.item() > 0 else float("inf")


def main():
    local_rank = setup()
    random.seed(SEED)
    torch.manual_seed(SEED + dist.get_rank())

    if is_main():
        print(f"Model: {MODEL_PATH}")
        print(f"DDP: 4 GPUs, batch={BATCH_PER_GPU}/GPU, effective_batch={BATCH_PER_GPU*4}")
        print(f"Steps: {STEPS}, LR: {LR}, MaxLen: {MAX_LEN}")

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if is_main():
        print("Loading datasets...")
    math_texts = load_math()
    lit_texts = load_lit()
    random.shuffle(math_texts)
    random.shuffle(lit_texts)

    if is_main():
        print(f"  math: {len(math_texts)} samples")
        print(f"  lit: {len(lit_texts)} samples")

    math_chunks = build_chunks(math_texts, tok)
    random.shuffle(math_chunks)
    lit_chunks = build_chunks(lit_texts, tok)
    random.shuffle(lit_chunks)

    math_train = math_chunks[:TRAIN_N]
    math_eval = math_chunks[TRAIN_N:TRAIN_N + EVAL_N]
    lit_train = lit_chunks[:TRAIN_N]

    if is_main():
        print(f"  math: {len(math_train)} train, {len(math_eval)} eval chunks")
        print(f"  lit: {len(lit_train)} train chunks")

    train_math_ds = ChunkDataset(math_train)
    train_lit_ds = ChunkDataset(lit_train)
    eval_math_ds = ChunkDataset(math_eval)

    train_math_loader = DataLoader(
        train_math_ds, batch_size=BATCH_PER_GPU,
        sampler=DistributedSampler(train_math_ds, shuffle=True),
        pin_memory=True, num_workers=4, drop_last=True)
    train_lit_loader = DataLoader(
        train_lit_ds, batch_size=BATCH_PER_GPU,
        sampler=DistributedSampler(train_lit_ds, shuffle=True),
        pin_memory=True, num_workers=4, drop_last=True)
    eval_math_loader = DataLoader(
        eval_math_ds, batch_size=BATCH_PER_GPU,
        sampler=DistributedSampler(eval_math_ds, shuffle=False),
        pin_memory=True, num_workers=4)

    if is_main():
        print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, local_files_only=True)
    model.config.pad_token_id = tok.eos_token_id
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    model.module.gradient_checkpointing_enable()

    if is_main():
        vram = torch.cuda.memory_allocated() / 1024**3
        print(f"  loaded  vram={vram:.1f}GB")

    dist.barrier()
    t0 = time.time()

    # Phase A: fine-tune on math
    finetune(model, train_math_loader, local_rank, "A=math")
    dist.barrier()
    ppl_before = ppl(model, eval_math_loader, local_rank)
    if is_main():
        print(f"  ppl(math) before lit: {ppl_before:.3f}")

    # Phase B: fine-tune on literature
    finetune(model, train_lit_loader, local_rank, "B=literature")
    dist.barrier()
    ppl_after = ppl(model, eval_math_loader, local_rank)
    if is_main():
        print(f"  ppl(math) after lit:  {ppl_after:.3f}")

    fg = (ppl_after - ppl_before) / ppl_before * 100
    if is_main():
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
                "n_gpus": 4,
                "batch_per_gpu": BATCH_PER_GPU,
                "effective_batch": BATCH_PER_GPU * 4,
                "steps": STEPS,
                "lr": LR,
                "elapsed_min": round(elapsed, 1),
            },
        }
        with open(OUT, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved -> {OUT}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
