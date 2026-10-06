#!/usr/bin/env bash
# Start the LOSVD explorer in the browser.
#
#   ./start_explorer.sh                          # scan the nearest results/ directory
#   ./start_explorer.sh ../../results/NGC0000_GP  # scan a directory
#   ./start_explorer.sh path/to/RUN_results.hdf5 # open one run directly
#
# Run it in the BAYES-LOSVD environment with plotly, ipywidgets, anywidget and voila installed.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -gt 0 ]]; then
  target="$1"
  if [[ ! -e "$target" ]]; then
    echo "Path does not exist: $target" >&2
    exit 1
  fi
  export LOSVD_EXPLORER_PATH="$(cd "$(dirname "$target")" && pwd)/$(basename "$target")"
fi

cd "$here"
if ! command -v voila >/dev/null 2>&1; then
  echo "voila not found – install it with: conda install -c conda-forge voila" >&2
  exit 1
fi
exec voila interactive_losvd_explorer.ipynb
