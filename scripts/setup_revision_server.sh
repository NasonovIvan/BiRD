#!/usr/bin/env bash
set -euo pipefail

repo_dir=${1:-$PWD}
venv_dir=${2:-$repo_dir/.venv-revision}
python_bin=${PYTHON_BIN:-python3}

cd "$repo_dir"
"$python_bin" -m venv "$venv_dir"
"$venv_dir/bin/python" -m pip install --upgrade pip wheel setuptools

# RTX 5090 (Blackwell, sm_120) needs a CUDA 12.8-capable PyTorch wheel.
"$venv_dir/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  "torch>=2.7"
"$venv_dir/bin/python" -m pip install -r requirements.txt
"$venv_dir/bin/python" -m pip install "matplotlib>=3.9"
# A preinstalled torchvision compiled for another torch/CUDA build can break
# transformers imports even though this text-only pipeline never uses it.
"$venv_dir/bin/python" -m pip uninstall -y torchvision || true

"$venv_dir/bin/python" - <<'PY'
import torch
import transformers

assert torch.cuda.is_available(), "PyTorch cannot see CUDA"
major, minor = torch.cuda.get_device_capability(0)
assert (major, minor) >= (12, 0), (major, minor)
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
print("capability", (major, minor))
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
print("bf16 smoke", float((x @ x).float().mean()))
PY
