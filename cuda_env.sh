# Sourced by setup.sh and run.sh (not run directly).
#
# The prebuilt CUDA build of llama-cpp-python needs the CUDA 12 runtime libraries
# (libcudart, libcublas). Many AMIs ship only the GPU driver, so setup.sh installs
# NVIDIA's pip packages of those libraries into .venv. This adds their folders to
# LD_LIBRARY_PATH so the loader can find them.

for cuda_lib_dir in .venv/lib*/python3*/site-packages/nvidia/*/lib; do
    if [ -d "$cuda_lib_dir" ]; then
        LD_LIBRARY_PATH="$PWD/$cuda_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
done
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
