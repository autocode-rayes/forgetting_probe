import os, sys
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
os.environ["HF_HOME"]="/root/autodl-tmp/hf_cache"
os.environ["TOKENIZERS_PARALLELISM"]="false"
os.environ["TORCHDYNAMO_DISABLE"]="1"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
import json,math,random,time
from itertools import permutations
import torch
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM,AutoTokenizer
from datasets import load_dataset
try:
    import torch._dynamo
    torch._dynamo.config.disable=True
except:pass
MODEL_PATH="/root/autodl-tmp/Qwen2.5-32B"
TOKEN_BUDGET=50000;MAX_LEN=256;TRAIN_N=175;EVAL_N=19;STEPS=200;BATCH=1;LR=2e-5;SEED=42
DTYPE=torch.bfloat16;N_GPUS=torch.cuda.device_count();INPUT_DEV="cuda:0"
OUT="/root/autodl-tmp/results_32b.json";PREV="/root/autodl-tmp/results_14b.json"
random.seed(SEED);torch.manual_seed(SEED)
print(f"32B on {N_GPUS} GPUs  device_map=balanced",flush=True)
def vram():return " | ".join(f"G{i}:{torch.cuda.memory_allocated(i)/1024**3:.1f}GB" for i in range(N_GPUS))
def load_math():
    try:
        ds=load_dataset("gsm8k","main",split="train")
        return[f"Question: {r['question']}\nAnswer: {r['answer']}" for r in ds]
    except:pass
    ds=load_dataset("wikitext","wikitext-103-v1",split="train")
    return[r["text"] for r in ds if len(r["text"].strip())>80]
def load_code():
    try:
        ds=load_dataset("code_search_net","python",split="train")
        return[r["whole_func_string"] for r in ds]
    except:pass
    ds=load_dataset("wikitext","wikitext-103-v1",split="train")
    return[r["text"] for r in ds if len(r["text"].strip())>80]
def load_lit():
    try:
        ds=load_dataset("wikitext","wikitext-103-raw-v1",split="train")
        return[r["text"] for r in ds if len(r["text"].strip())>80]
    except:pass
    ds=load_dataset("wikitext","wikitext-103-v1",split="train")
    return[r["text"] for r in ds if len(r["text"].strip())>80]
def build_chunks(texts,tok):
    ids=[]
    while len(ids)<TOKEN_BUDGET:
        for t in texts:
            ids.extend(tok(t,add_special_tokens=False)["input_ids"])
            if len(ids)>=TOKEN_BUDGET:break
    ids=ids[:TOKEN_BUDGET]
    return[torch.tensor(ids[i:i+MAX_LEN],dtype=torch.long) for i in range(0,len(ids)-MAX_LEN+1,MAX_LEN)]
def make_batches(chunks,shuffle=True):
    ch=list(chunks)
    if shuffle:random.shuffle(ch)
    return[torch.stack(ch[i:i+BATCH]) for i in range(0,len(ch)-BATCH+1,BATCH)]
def finetune(model,blist,label):
    model.train()
    opt=bnb.optim.AdamW8bit(model.parameters(),lr=LR)
    step=total=0
    while step<STEPS:
        for b in blist:
            if step>=STEPS:break
            out=model(input_ids=b.to(INPUT_DEV),labels=b.to(INPUT_DEV))
            loss=out.loss
            if torch.isnan(loss):opt.zero_grad();continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step();opt.zero_grad()
            total+=loss.item();step+=1
            if step%50==0:print(f"    {label} {step}/{STEPS} loss={total/step:.4f} [{vram()}]",flush=True)
@torch.no_grad()
def ppl(model,blist):
    model.eval()
    nll=ntok=0
    for b in blist:
        out=model(input_ids=b.to(INPUT_DEV),labels=b.to(INPUT_DEV))
        if torch.isnan(out.loss):continue
        nll+=out.loss.item()*b.numel();ntok+=b.numel()
    return math.exp(nll/ntok) if ntok else float("inf")
tok=AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True)
if tok.pad_token is None:tok.pad_token=tok.eos_token
print("Loading datasets...",flush=True)
raw={"math":load_math(),"code":load_code(),"literature":load_lit()}
print("Tokenizing...",flush=True)
dom={}
for name,texts in raw.items():
    random.shuffle(texts)
    ch=build_chunks(texts,tok);random.shuffle(ch)
    dom[name]={"train":ch[:TRAIN_N],"eval":ch[TRAIN_N:TRAIN_N+EVAL_N]}
    print(f"  {name}: {len(dom[name]['train'])} train, {len(dom[name]['eval'])} eval",flush=True)
results={};t0=time.time()
print(f"\n{'='*60}\nRunning 6 pairs [32B]\n{'='*60}",flush=True)
for da,db in permutations(dom.keys(),2):
    key=f"{da}->{db}"
    print(f"\n[{key}]",flush=True)
    ta=make_batches(dom[da]["train"]);tb=make_batches(dom[db]["train"])
    ea=make_batches(dom[da]["eval"],shuffle=False)
    try:
        model=AutoModelForCausalLM.from_pretrained(MODEL_PATH,dtype=DTYPE,device_map="balanced",local_files_only=True)
        model.config.pad_token_id=tok.eos_token_id
        model.gradient_checkpointing_enable()
        print(f"  loaded [{vram()}]",flush=True)
        finetune(model,ta,f"A={da}")
        pb=ppl(model,ea);print(f"  ppl({da}) before: {pb:.3f}",flush=True)
        finetune(model,tb,f"B={db}")
        pa=ppl(model,ea);print(f"  ppl({da}) after:  {pa:.3f}",flush=True)
        fg=(pa-pb)/pb*100;print(f"  forgetting: {fg:+.2f}%",flush=True)
        results[key]={"domain_a":da,"domain_b":db,"ppl_before":round(pb,4),"ppl_after":round(pa,4),"forgetting_pct":round(fg,4)}
        del model
        for i in range(N_GPUS):torch.cuda.empty_cache()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"OOM on pair {key}: {e}",flush=True)
            results[key]={"error":"OOM","message":str(e)}
            for i in range(N_GPUS):torch.cuda.empty_cache()
            break
        raise
    with open(OUT,"w") as f:json.dump(results,f,indent=2)
elapsed=(time.time()-t0)/60
fg_vals=[r["forgetting_pct"] for r in results.values() if "forgetting_pct" in r]
if fg_vals:
    sp=max(fg_vals)-min(fg_vals)
    min_p=min((k for k in results if "forgetting_pct" in results[k]),key=lambda k:results[k]["forgetting_pct"])
    max_p=max((k for k in results if "forgetting_pct" in results[k]),key=lambda k:results[k]["forgetting_pct"])
    print(f"\nDone in {elapsed:.1f} min  Spread={sp:.2f}pp  Low={min_p}  High={max_p}",flush=True)
    results["_summary"]={"model":MODEL_PATH,"spread_pct":round(sp,4),"min_pair":min_p,"max_pair":max_p,"elapsed_min":round(elapsed,1)}
else:
    print("No successful pairs -- OOM hit early",flush=True)
    results["_summary"]={"model":MODEL_PATH,"error":"OOM before any pair completed"}
with open(OUT,"w") as f:json.dump(results,f,indent=2)
print(f"Saved -> {OUT}",flush=True)
