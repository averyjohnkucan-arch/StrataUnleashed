#!/usr/bin/env bash
# Download the app into Documents, then select/download/prepare a model interactively.
set -euo pipefail
INSTALL_DIR="${STRATA_INSTALL_DIR:-$HOME/Documents/StrataUnleashed}"
if ! command -v python3 >/dev/null || ! python3 -c 'import sys; assert sys.version_info >= (3,10)' 2>/dev/null; then
    echo 'Python 3.10 or newer is needed. On Ubuntu/Debian: sudo apt install python3 python3-venv'
    exit 1
fi
mkdir -p "$INSTALL_DIR/work/installer" "$INSTALL_DIR/work/tmp"
export TMPDIR="$INSTALL_DIR/work/tmp" PYTHONDONTWRITEBYTECODE=1
curl -fsSL https://raw.githubusercontent.com/averyjohnkucan-arch/StrataUnleashed/unleashed/install_unleashed.py -o "$INSTALL_DIR/work/installer/install_unleashed.py"
exec python3 -B "$INSTALL_DIR/work/installer/install_unleashed.py" --directory "$INSTALL_DIR" "$@"
