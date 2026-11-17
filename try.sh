#!/usr/bin/env bash
# Type your own messages and see the judge's verdicts, inside the env created by setup.sh.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
    echo "No .venv found. Run ./setup.sh first."
    exit 1
fi
# shellcheck disable=SC1091
. .venv/bin/activate
python try_prompt.py
