#!/usr/bin/env bash
# One-time setup on the instance: Python virtual env, llama-cpp-python with CUDA,
# and the other requirements. Safe to run again.
set -euo pipefail
cd "$(dirname "$0")"

echo "== GPU =="
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found. G6f (fractional GPU) instances need the NVIDIA GRID driver 18.4 or later."
    echo "See README.md, 'Instance prerequisites'."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "== Python env =="
python3 -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
pip install --upgrade pip wheel

echo "== llama-cpp-python (CUDA) =="
# A prebuilt CUDA wheel is used when one matches this Python version. Otherwise pip
# compiles from source; CMAKE_ARGS makes that compile include CUDA. Compiling needs the
# CUDA toolkit (nvcc) and takes 15-40 minutes on 2 vCPUs.
CUDA_WHEEL_INDEX="${CUDA_WHEEL_INDEX:-https://abetlen.github.io/llama-cpp-python/whl/cu124}"
if ! command -v nvcc >/dev/null 2>&1 && [ -x /usr/local/cuda/bin/nvcc ]; then
    export PATH="/usr/local/cuda/bin:$PATH"
fi
CMAKE_ARGS="-DGGML_CUDA=on" pip install --prefer-binary \
    --extra-index-url "$CUDA_WHEEL_INDEX" llama-cpp-python

echo "== CUDA runtime libraries =="
# The prebuilt wheel links against libcudart.so.12 and libcublas.so.12, which the
# GPU driver does not include. Version 12.4 matches the cu124 wheel index above.
pip install "nvidia-cuda-runtime-cu12==12.4.*" "nvidia-cublas-cu12==12.4.*"

echo "== other requirements =="
pip install -r requirements.txt

echo "== check =="
# shellcheck disable=SC1091
. ./cuda_env.sh
python - <<'EOF'
import llama_cpp
gpu = llama_cpp.llama_supports_gpu_offload()
print("llama-cpp-python", llama_cpp.__version__, "| GPU offload:", gpu)
if not gpu:
    print("WARNING: CPU-only build. Install the CUDA toolkit, then run:")
    print("  . .venv/bin/activate && CMAKE_ARGS=-DGGML_CUDA=on pip install --force-reinstall --no-cache-dir --no-binary llama-cpp-python llama-cpp-python")
EOF

echo
echo "Setup done. Run the check with:  ./run.sh"
