import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-3B", local_dir="/root/autodl-tmp/Qwen2.5-3B",
                  ignore_patterns=["*.msgpack", "*.h5", "flax*"])
print("DOWNLOAD_DONE")
