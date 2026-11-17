#!/usr/bin/env bash
# Copy script: gets this repo onto the instance, or updates it if it is already there.
#
#   curl -fsSL https://raw.githubusercontent.com/anonymouslearner612/granite-judge-instance-check/main/get.sh | bash
#
# Optional argument: the folder to use (default ~/granite-judge-instance-check).
set -euo pipefail

REPO_URL="https://github.com/anonymouslearner612/granite-judge-instance-check.git"
TARGET_DIR="${1:-$HOME/granite-judge-instance-check}"

if ! command -v git >/dev/null 2>&1; then
    echo "git is not installed. Install it first (Amazon Linux: sudo dnf install -y git, Ubuntu: sudo apt-get install -y git)."
    exit 1
fi

if [ -d "$TARGET_DIR/.git" ]; then
    echo "Updating $TARGET_DIR"
    git -C "$TARGET_DIR" pull --ff-only
else
    echo "Cloning into $TARGET_DIR"
    git clone "$REPO_URL" "$TARGET_DIR"
fi

chmod +x "$TARGET_DIR"/*.sh
echo
echo "Next:"
echo "  cd $TARGET_DIR"
echo "  ./setup.sh    # first time only"
echo "  ./run.sh"
