#!/usr/bin/env bash
# run_combined_best.sh
# Convenience wrapper for: combined_select_best_and_plot.py
# Usage examples:
#   ./run_combined_best.sh
#   ./run_combined_best.sh --models xlmr --techniques core,per_entity
#   ROOT_OVERRIDE=/abs/path/to/outputs/combined ./run_combined_best.sh --outdir summary_best_xlmr

set -euo pipefail

# Resolve paths
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PY_SCRIPT="$SCRIPT_DIR/combined_select_best_and_plot.py"

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "Error: Can't find combined_select_best_and_plot.py next to this script." >&2
  echo "Looked at: $PY_SCRIPT" >&2
  exit 1
fi

# Prefer local virtualenv if present, otherwise fall back to python3 on PATH
if [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
  PY_EXE="$SCRIPT_DIR/.venv/bin/python"
else
  PY_EXE="python3"
fi

# Default root (can override with env var ROOT_OVERRIDE, or by passing --root yourself)
DEFAULT_ROOT="${ROOT_OVERRIDE:-outputs/combined}"

# Helper: does the arg list already contain a given flag?
has_flag() {
  local flag="$1"; shift || true
  for a in "$@"; do
    [[ "$a" == "$flag" ]] && return 0
  done
  return 1
}

# Build final argv: inject --root DEFAULT_ROOT only if user didn't specify --root
ARGV=()
if has_flag "--root" "$@"; then
  ARGV=( "$@" )
else
  ARGV=( --root "$DEFAULT_ROOT" "$@" )
fi

# Run
exec "$PY_EXE" "$PY_SCRIPT" "${ARGV[@]}"
