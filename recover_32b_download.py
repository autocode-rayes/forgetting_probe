"""
Recovery script for Qwen2.5-32B download.
Run this if the main dl_32b.py process dies mid-download.
It will skip already-complete shards and resume incomplete ones.
"""
import os, subprocess, time

os.environ["HF_ENDPOINT"]               = "https://hf-mirror.com"
os.environ["HF_HOME"]                   = "/root/autodl-tmp/hf_cache"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import snapshot_download, list_repo_files

LOCAL_DIR = "/root/autodl-tmp/Qwen2.5-32B"
HF_ID     = "Qwen/Qwen2.5-32B"

os.makedirs(LOCAL_DIR, exist_ok=True)

# ── Check current state ───────────────────────────────────────────────────────
all_shards = sorted([
    f for f in list_repo_files(HF_ID)
    if f.endswith(".safetensors")
])
total = len(all_shards)
print(f"Total shards in repo: {total}")

complete = [f for f in os.listdir(LOCAL_DIR) if f.endswith(".safetensors")]
print(f"Already complete: {len(complete)}/{total}")

cache_dir = os.path.join(LOCAL_DIR, ".cache", "huggingface", "download")
incomplete = []
if os.path.isdir(cache_dir):
    incomplete = [f for f in os.listdir(cache_dir) if f.endswith(".incomplete")]
    print(f"Incomplete cache files: {len(incomplete)}")
    for f in incomplete:
        path = os.path.join(cache_dir, f)
        size = os.path.getsize(path) / 1024**3
        print(f"  {f[:30]}...  {size:.2f}GB")

if len(complete) == total:
    print("\nAll shards complete! Nothing to recover.")
else:
    print(f"\nRecovering: resuming download ({total - len(complete)} shards remaining)...")
    snapshot_download(
        HF_ID,
        local_dir=LOCAL_DIR,
        ignore_patterns=["*.msgpack", "*.h5", "flax*"],
    )
    print("DONE — all shards downloaded.")
