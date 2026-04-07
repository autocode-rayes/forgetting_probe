#!/bin/bash
# TPU VM setup script — run this once after SSHing into the TPU VM.
# Installs dependencies and configures the environment for 32B/72B experiments.
#
# Usage (on the TPU VM):
#   export GOOGLE_API_KEY=<your-key>   # set this before running
#   bash tpu_vm_setup.sh

set -e

echo "=== TPU VM setup for forgetting probe ==="

# ── 1. System packages ────────────────────────────────────────────────────────
sudo apt-get update -qq
sudo apt-get install -y -qq git screen tmux htop

# ── 2. Python packages ────────────────────────────────────────────────────────
# tpu-ubuntu2204-base ships with torch 2.11.0 but no matching torch_xla.
# Install a pinned pair that has a known-good libtpu release.
pip3 install --quiet \
    torch==2.6.0 \
    torch_xla[tpu]==2.6.0 \
    -f https://storage.googleapis.com/libtpu-releases/index.html

pip3 install --upgrade --quiet \
    "transformers>=4.43" \
    "datasets>=2.20" \
    "accelerate>=0.30" \
    "huggingface_hub>=0.24" \
    "modelscope>=1.14" \
    "sentencepiece" \
    "protobuf"

echo "Package versions:"
python -c "import torch; print(f'  torch {torch.__version__}')"
python -c "import torch_xla; print(f'  torch_xla {torch_xla.__version__}')"
python -c "import transformers; print(f'  transformers {transformers.__version__}')"

# ── 3. Environment ────────────────────────────────────────────────────────────
# Persist key + project for this session and future logins
PROFILE_FILE="${HOME}/.bashrc"

{
  echo ""
  echo "# Forgetting probe — TPU environment"
  echo "export PJRT_DEVICE=TPU"
  echo "export GOOGLE_CLOUD_PROJECT=forgettingprobe"
  echo "export HF_HUB_ENABLE_HF_TRANSFER=1"
  # GOOGLE_API_KEY is NOT written here — set it manually each session:
  #   export GOOGLE_API_KEY=<your-key>
  echo "export TOKENIZERS_PARALLELISM=false"
} >> "${PROFILE_FILE}"

source "${PROFILE_FILE}"

# ── 4. Clone / update the repo ────────────────────────────────────────────────
REPO_DIR="${HOME}/forgetting_probe"
if [ -d "${REPO_DIR}" ]; then
    echo "Updating existing repo..."
    git -C "${REPO_DIR}" pull
else
    echo "Cloning repo..."
    git clone https://github.com/autocode-rayes/forgetting_probe.git "${REPO_DIR}"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "To run experiments (use tmux/screen to keep alive on spot VMs):"
echo ""
echo "  # 32B — all 6 domain pairs:"
echo "  MODEL_ID=Qwen/Qwen2.5-32B python ${REPO_DIR}/forgetting_tpu_32b_plus.py"
echo ""
echo "  # 72B — all 6 domain pairs:"
echo "  MODEL_ID=Qwen/Qwen2.5-72B python ${REPO_DIR}/forgetting_tpu_32b_plus.py"
echo ""
echo "  # 32B + 72B batch sweep (extends 2D surface):"
echo "  python ${REPO_DIR}/tpu_batchsweep_32b.py"
echo ""
echo "  # Restart after preemption (skips completed pairs):"
echo "  RESUME=1 MODEL_ID=Qwen/Qwen2.5-32B python ${REPO_DIR}/forgetting_tpu_32b_plus.py"
echo ""
echo "For multi-host TPU (v4-32), run via:"
echo "  gcloud compute tpus tpu-vm ssh MY_TPU --zone=us-central2-b --worker=all \\"
echo "    --command='MODEL_ID=Qwen/Qwen2.5-32B python ~/forgetting_probe/forgetting_tpu_32b_plus.py'"
