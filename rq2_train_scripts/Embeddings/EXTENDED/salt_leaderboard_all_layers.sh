#!/usr/bin/env bash
set -euo pipefail


BASE_DIR="${1:-rq2_train_scripts/Embeddings/EXTENDED/outputs/salt_extended}"
METRIC_COLUMN="${2:-Mean}"

python3 salt_leaderboard.py \
  --base-dir "$BASE_DIR" \
  --metric-column "$METRIC_COLUMN" \
  --analyze-all-layers \
  --plot-topk 20
