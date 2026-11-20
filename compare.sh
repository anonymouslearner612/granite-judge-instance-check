#!/usr/bin/env bash
# Compare all candidate judge models inside the env created by setup.sh. Arguments are
# passed through, e.g.  ./compare.sh --models granite-guardian-3.1-2b,granite-4.0-micro
# Also the command the granite-model-compare systemd service runs.
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
python -u model_compare.py "$@"
