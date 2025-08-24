#!/usr/bin/env bash
set -euo pipefail


BASE_DIR="${1:-rq2_train_scripts/Embeddings/EXTENDED/outputs/mptc_extended}"
METRIC_COLUMN="${2:-Mean}"
OUT_CSV="${3:-rq2_train_scripts/Embeddings/EXTENDED/outputs/mptc_extended_leaderboard.csv}"

python3 mptc_leaderboard.py \
  --base-dir "$BASE_DIR" \
  --metric-column "$METRIC_COLUMN" \
  --out-csv "$OUT_CSV" \
  --plot-topk 20 \
  --plot-out rq2_train_scripts/Embeddings/EXTENDED/outputs/mptc_extended_leaderboard.png \
  --plot-per-model
