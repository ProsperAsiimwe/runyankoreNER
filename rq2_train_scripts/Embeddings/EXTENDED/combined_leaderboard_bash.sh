#!/usr/bin/env bash
set -euo pipefail


BASE_DIR="${1:-rq2_train_scripts/Embeddings/EXTENDED/outputs/combined_extended}"
METRIC_COLUMN="${2:-Mean}"
OUT_CSV="${3:-rq2_train_scripts/Embeddings/EXTENDED/outputs/combined_extended_leaderboard.csv}"

python3 combined_leaderboard.py \
  --base-dir "$BASE_DIR" \
  --metric-column "$METRIC_COLUMN" \
  --out-csv "$OUT_CSV"
