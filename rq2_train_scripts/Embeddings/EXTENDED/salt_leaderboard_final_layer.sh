#!/usr/bin/env bash
set -euo pipefail


BASE_DIR="${1:-rq2_train_scripts/Embeddings/EXTENDED/outputs/salt_extended}"
METRIC_COLUMN="${2:-Mean}"
OUT_CSV="${3:-rq2_train_scripts/Embeddings/EXTENDED/outputs/salt_extended_leaderboard_final.csv}"

python3 rq2_train_scripts/Embeddings/EXTENDED/salt_leaderboard.py \
  --base-dir "$BASE_DIR" \
  --metric-column "$METRIC_COLUMN" \
  --out-csv "$OUT_CSV" \
  --plot-topk 20 \
  --plot-per-model
