#!/usr/bin/env bash
# Run the instance check inside the virtual env created by setup.sh.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
    echo "No .venv found. Run ./setup.sh first."
    exit 1
fi
# shellcheck disable=SC1091
. .venv/bin/activate
python instance_check.py
