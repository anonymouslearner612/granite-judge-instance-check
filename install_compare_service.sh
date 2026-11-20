#!/usr/bin/env bash
# Install model_compare.py as the systemd service granite-model-compare and start it.
# The run keeps going after you log out or SSH disconnects.
#
#   ./install_compare_service.sh                         all models
#   ./install_compare_service.sh --models granite-guardian-3.1-2b,granite-4.0-micro
#   ./install_compare_service.sh --delete-models         save disk: remove each model after its run
#
# Arguments are passed to compare.sh. Running this again replaces the service and starts a
# new run. Follow it with:  journalctl -u granite-model-compare -f
set -euo pipefail
cd "$(dirname "$0")"

SERVICE_NAME="granite-model-compare"
SERVICE_FILE="/etc/systemd/system/$SERVICE_NAME.service"
PROJECT_DIR="$(pwd)"
RUN_AS_USER="$(id -un)"

if [ ! -d .venv ]; then
    echo "No .venv found. Run ./setup.sh first."
    exit 1
fi
chmod +x compare.sh

# Each argument is quoted so systemd passes it through unchanged.
ARGUMENTS=""
for argument in "$@"; do
    ARGUMENTS="$ARGUMENTS \"$argument\""
done

# | as the sed separator because the paths contain /.
sed -e "s|__USER__|$RUN_AS_USER|" \
    -e "s|__PROJECT_DIR__|$PROJECT_DIR|g" \
    -e "s|__ARGUMENTS__|$ARGUMENTS|" \
    "$SERVICE_NAME.service" | sudo tee "$SERVICE_FILE" >/dev/null

sudo systemctl daemon-reload
# --no-block: a oneshot service's start otherwise waits until the whole comparison finishes.
sudo systemctl restart --no-block "$SERVICE_NAME"

echo "Started $SERVICE_NAME as $RUN_AS_USER in $PROJECT_DIR."
echo "  follow the output:  journalctl -u $SERVICE_NAME -f"
echo "  status:             systemctl status $SERVICE_NAME"
echo "  stop:               sudo systemctl stop $SERVICE_NAME"
echo "  results:            $PROJECT_DIR/comparison_<time>/comparison.txt"
