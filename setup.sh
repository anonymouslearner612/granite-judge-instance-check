#!/usr/bin/env bash
# One-time setup on the instance: Python virtual env, llama-cpp-python compiled with
# CUDA, and the other requirements. Safe to run again: a working build is kept.
#
# llama-cpp-python is compiled here instead of using a prebuilt wheel because the
# prebuilt CUDA wheels assume CPU instructions (AVX-512) that the AMD CPUs in G6/G6f
# instances do not have; importing them crashes with "Illegal instruction".
set -euo pipefail
cd "$(dirname "$0")"

# GPU architecture to compile CUDA kernels for. 89 = Ada (NVIDIA L4, used by G6/G6f).
# Compiling for one architecture keeps the build time down on 2 vCPUs.
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-89}"
# CUDA version for the compiler packages. It must not be newer than the driver supports
# (the "CUDA Version" in the nvidia-smi header). GRID driver 18.4+ supports 12.8.
CUDA_VERSION_DASHED="${CUDA_VERSION_DASHED:-12-8}"

check_gpu_driver() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi not found. G6f (fractional GPU) instances need the NVIDIA GRID driver 18.4 or later."
        echo "See README.md, 'Instance prerequisites'."
        exit 1
    fi
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    nvidia-smi | grep -o 'CUDA Version: [0-9.]*' || true
}

add_cuda_to_path() {
    if ! command -v nvcc >/dev/null 2>&1 && [ -x /usr/local/cuda/bin/nvcc ]; then
        export PATH="/usr/local/cuda/bin:$PATH"
    fi
}

install_cuda_compiler() {
    add_cuda_to_path
    if command -v nvcc >/dev/null 2>&1; then
        echo "nvcc found: $(command -v nvcc)"
        return
    fi

    if ! grep -q 'amzn' /etc/os-release || ! grep -q 'VERSION_ID="2023"' /etc/os-release; then
        echo "nvcc (CUDA compiler) not found, and this is not Amazon Linux 2023, so it is not installed automatically."
        echo "Install the CUDA $CUDA_VERSION_DASHED compiler, cudart and cuBLAS dev packages, then run ./setup.sh again."
        exit 1
    fi

    # Only what llama.cpp's CUDA build needs, not the full toolkit. cuda-driver-devel
    # is just the link stub for libcuda; the real GRID driver is left untouched.
    echo "Installing the CUDA compiler from NVIDIA's Amazon Linux 2023 repository"
    sudo dnf install -y dnf-plugins-core gcc gcc-c++ make git
    sudo dnf config-manager --add-repo \
        https://developer.download.nvidia.com/compute/cuda/repos/amzn2023/x86_64/cuda-amzn2023.repo
    sudo dnf install -y \
        "cuda-nvcc-$CUDA_VERSION_DASHED" \
        "cuda-cudart-devel-$CUDA_VERSION_DASHED" \
        "cuda-cccl-$CUDA_VERSION_DASHED" \
        "cuda-driver-devel-$CUDA_VERSION_DASHED" \
        "libcublas-devel-$CUDA_VERSION_DASHED"

    local cuda_dir="/usr/local/cuda-${CUDA_VERSION_DASHED/-/.}"
    export PATH="$cuda_dir/bin:$PATH"
    if ! command -v nvcc >/dev/null 2>&1; then
        echo "nvcc still not found after installing. Expected it in $cuda_dir/bin."
        exit 1
    fi
}

# Exit code 0 only if llama_cpp imports without crashing and supports GPU offload.
# Runs in a separate process because a bad build kills the process outright.
llama_cpp_works() {
    python -c 'import sys, llama_cpp; sys.exit(0 if llama_cpp.llama_supports_gpu_offload() else 1)' \
        >/dev/null 2>&1
}

build_llama_cpp() {
    echo "Compiling llama-cpp-python with CUDA. This takes roughly 15-40 minutes on 2 vCPUs."
    echo "Progress is printed below and saved to llama_build.log."
    # --verbose shows compile progress; without it pip looks frozen for the whole build.
    if ! CUDACXX="$(command -v nvcc)" \
        CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)" \
        CMAKE_ARGS="-DGGML_CUDA=on -DGGML_NATIVE=on -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCHITECTURES" \
        pip install --force-reinstall --no-cache-dir --no-binary llama-cpp-python --verbose \
        llama-cpp-python 2>&1 | tee llama_build.log; then
        echo "Build failed. Full log: llama_build.log"
        exit 1
    fi
}

echo "== GPU =="
check_gpu_driver

echo "== Python env =="
python3 -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
pip install --upgrade pip wheel

echo "== CUDA runtime libraries =="
# Loaded at run time from .venv via cuda_env.sh. Version matches the compiler packages.
pip install "nvidia-cuda-runtime-cu12==${CUDA_VERSION_DASHED/-/.}.*" "nvidia-cublas-cu12==${CUDA_VERSION_DASHED/-/.}.*"
# shellcheck disable=SC1091
. ./cuda_env.sh

echo "== llama-cpp-python (CUDA) =="
if llama_cpp_works; then
    echo "Existing build works, keeping it."
else
    install_cuda_compiler
    build_llama_cpp
fi

echo "== other requirements =="
pip install -r requirements.txt

echo "== check =="
python - <<'EOF'
import llama_cpp
gpu = llama_cpp.llama_supports_gpu_offload()
print("llama-cpp-python", llama_cpp.__version__, "| GPU offload:", gpu)
if not gpu:
    print("WARNING: CPU-only build. See llama_build.log.")
EOF

echo
echo "Setup done. Run the check with:  ./run.sh"
