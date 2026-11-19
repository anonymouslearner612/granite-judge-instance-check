#!/usr/bin/env bash
# Copies the gateway files that gateway_compare.py needs into gateway_snapshot/src, so the
# GPU instance only needs `git pull` and never the gateway repo itself.
#
# Run where the gateway repo is, then commit gateway_snapshot/:
#   ./update_gateway_snapshot.sh /path/to/ai-gateway-source
#
# Only the six scanners gateway_compare.py runs and the modules they import are copied.
# app.core.settings and app.loggers.applicationlogger are NOT copied: the real ones read
# the gateway's server.properties at import time, so gateway_compare.py supplies stand-ins.
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "usage: $0 /path/to/ai-gateway-source"
    exit 2
fi
gateway_source="$1/src"
snapshot="$(cd "$(dirname "$0")" && pwd)/gateway_snapshot/src"

files_to_copy=(
    app/guardrail/constants.py
    app/guardrail/schemas.py
    app/guardrail/reason_text.py
    app/guardrail/redaction.py
    app/guardrail/text_chunking.py
    app/guardrail/scanner_thresholds.py
    app/guardrail/rules/malware_patterns.yar
    app/guardrail/scanners/secret_scanner.py
    app/guardrail/scanners/pii_scanner.py
    app/guardrail/scanners/malware_scanner.py
    app/guardrail/scanners/malicious_entity_scanner.py
    app/guardrail/scanners/prompt_injection_scanner.py
    app/guardrail/scanners/toxicity_scanner.py
    app/schemas/base_model.py
    app/utils/runtime_parsers/runtime_constants.py
)
# Python packages along those paths. Empty __init__.py files, as in the gateway.
package_dirs=(
    app
    app/core
    app/loggers
    app/guardrail
    app/guardrail/scanners
    app/schemas
    app/utils
    app/utils/runtime_parsers
)

rm -rf "$snapshot"
for relative_path in "${files_to_copy[@]}"; do
    mkdir -p "$snapshot/$(dirname "$relative_path")"
    cp "$gateway_source/$relative_path" "$snapshot/$relative_path"
done
for package_dir in "${package_dirs[@]}"; do
    mkdir -p "$snapshot/$package_dir"
    touch "$snapshot/$package_dir/__init__.py"
done

commit="$(git -C "$1" log -1 --format='%h %cd %s' --date=short 2>/dev/null || echo unknown)"
echo "$commit" > "$snapshot/../SOURCE_COMMIT"
echo "copied ${#files_to_copy[@]} files from ai-gateway-source at: $commit"
