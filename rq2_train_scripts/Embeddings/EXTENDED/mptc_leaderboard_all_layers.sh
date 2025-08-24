#!/usr/bin/env bash
set -euo pipefail


BASE_DIR="${1:-rq2_train_scripts/Embeddings/EXTENDED/outputs/mptc_extended}"
METRIC_COLUMN="${2:-Mean}"

python3 mptc_leaderboard.py \
  --base-dir "$BASE_DIR" \
  --metric-column "$METRIC_COLUMN" \
  --analyze-all-layers \
  --plot-topk 20
