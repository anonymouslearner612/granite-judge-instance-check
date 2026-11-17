#!/usr/bin/env bash
# Run the benchmark inside the env created by setup.sh. Arguments are passed through,
# e.g.  ./bench.sh --concurrency 2,4,8
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
    echo "No .venv found. Run ./setup.sh first."
    exit 1
fi
# shellcheck disable=SC1091
. .venv/bin/activate
# shellcheck disable=SC1091
. ./cuda_env.sh
python benchmark.py "$@"
