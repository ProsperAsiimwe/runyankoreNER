#!/bin/bash
set -euo pipefail

# =========================
# GLOBAL SETTINGS
# =========================
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export SEED=${SEED:-42}

# Path to your updated Python script (the one we just wrote)
# Change this if your file lives elsewhere.
PY_SCRIPT="${PY_SCRIPT:-rq2_train_scripts/Embeddings/EXTENDED/salt.py}"

# Base output directory (the Python script will create model/config subfolders)
BASE_OUTDIR="${BASE_OUTDIR:-rq2_train_scripts/Embeddings/EXTENDED/outputs/salt_cosine}"

# Models: set RUN_BOTH=true to evaluate both; otherwise set MODEL to one of: xlmr | mbert
RUN_BOTH="${RUN_BOTH:-true}"
MODEL="${MODEL:-xlmr}"

# =========================
# SWEEP SETTINGS
# =========================
# These arrays define the grid search values. Edit as desired.
# (They will be converted into comma-separated lists for the Python --grid_* flags.)

# Whether to use CLS pooling (0/1)
USE_CLS_LIST=(${USE_CLS_LIST:-0 1})

# Whether to include context around the entity (0/1)
USE_CONTEXT_LIST=(${USE_CONTEXT_LIST:-0 1})

# Context window sizes
CONTEXT_WINDOWS=(${CONTEXT_WINDOWS:-0 2 4})

# Max entity tokens per tag
TOKEN_LIMITS=(${TOKEN_LIMITS:-250 500})

# Batch sizes
BATCH_SIZES=(${BATCH_SIZES:-32 64})

# Single-run defaults (used by the Python script if a grid is empty;
# we’ll pass both the defaults AND the grid below, so defaults are mostly informational here)
DEFAULT_CONTEXT_WINDOW="${DEFAULT_CONTEXT_WINDOW:-2}"
DEFAULT_MAX_TOKENS_PER_TYPE="${DEFAULT_MAX_TOKENS_PER_TYPE:-500}"
DEFAULT_BATCH_SIZE="${DEFAULT_BATCH_SIZE:-64}"
DEFAULT_USE_CLS="${DEFAULT_USE_CLS:-0}"        # 0/1
DEFAULT_USE_CONTEXT="${DEFAULT_USE_CONTEXT:-0}" # 0/1

# =========================
# HELPERS
# =========================
join_by_comma () {
  local IFS=','; echo "$*"
}

to_bool () {
  # Normalize to 0/1 for printing logs only
  case "$1" in
    1|true|TRUE|True|yes|y|Y) echo 1 ;;
    *) echo 0 ;;
  esac
}

# Convert arrays to comma-separated lists for the Python --grid_* flags
GRID_USE_CLS="$(join_by_comma "${USE_CLS_LIST[@]}")"
GRID_USE_CONTEXT="$(join_by_comma "${USE_CONTEXT_LIST[@]}")"
GRID_CONTEXT_WINDOW="$(join_by_comma "${CONTEXT_WINDOWS[@]}")"
GRID_MAX_TOKENS_PER_TYPE="$(join_by_comma "${TOKEN_LIMITS[@]}")"
GRID_BATCH_SIZE="$(join_by_comma "${BATCH_SIZES[@]}")"

# Print config
echo "================ Python sweep config ================"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "SEED=${SEED}"
echo "PY_SCRIPT=${PY_SCRIPT}"
echo "BASE_OUTDIR=${BASE_OUTDIR}"
if [[ "${RUN_BOTH}" == "true" ]]; then
  echo "MODELS=xlmr,mbert (RUN_BOTH=true)"
else
  echo "MODEL=${MODEL} (RUN_BOTH=false)"
fi
echo "Grid:"
echo "  grid_use_cls            = ${GRID_USE_CLS}"
echo "  grid_use_context        = ${GRID_USE_CONTEXT}"
echo "  grid_context_window     = ${GRID_CONTEXT_WINDOW}"
echo "  grid_max_tokens_per_type= ${GRID_MAX_TOKENS_PER_TYPE}"
echo "  grid_batch_size         = ${GRID_BATCH_SIZE}"
echo "Defaults (informational):"
echo "  use_cls=${DEFAULT_USE_CLS} use_context=${DEFAULT_USE_CONTEXT} context_window=${DEFAULT_CONTEXT_WINDOW}"
echo "  max_tokens_per_type=${DEFAULT_MAX_TOKENS_PER_TYPE} batch_size=${DEFAULT_BATCH_SIZE}"
echo "====================================================="
echo

mkdir -p "${BASE_OUTDIR}"

# =========================
# RUN
# =========================
if [[ "${RUN_BOTH}" == "true" ]]; then
  echo "[RUN] Both models (xlmr + mbert) with a single sweep call"
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python3 "${PY_SCRIPT}" \
    --run_both \
    --seed "${SEED}" \
    --output_dir "${BASE_OUTDIR}" \
    --grid_use_cls "${GRID_USE_CLS}" \
    --grid_use_context "${GRID_USE_CONTEXT}" \
    --grid_context_window "${GRID_CONTEXT_WINDOW}" \
    --grid_max_tokens_per_type "${GRID_MAX_TOKENS_PER_TYPE}" \
    --grid_batch_size "${GRID_BATCH_SIZE}" \
    $( [ "$(to_bool "${DEFAULT_USE_CLS}")" -eq 1 ] && echo "--use_cls" ) \
    $( [ "$(to_bool "${DEFAULT_USE_CONTEXT}")" -eq 1 ] && echo "--use_context" ) \
    --context_window "${DEFAULT_CONTEXT_WINDOW}" \
    --max_tokens_per_type "${DEFAULT_MAX_TOKENS_PER_TYPE}" \
    --batch_size "${DEFAULT_BATCH_SIZE}"
else
  echo "[RUN] Single model: ${MODEL}"
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python3 "${PY_SCRIPT}" \
    --model_type "${MODEL}" \
    --seed "${SEED}" \
    --output_dir "${BASE_OUTDIR}" \
    --grid_use_cls "${GRID_USE_CLS}" \
    --grid_use_context "${GRID_USE_CONTEXT}" \
    --grid_context_window "${GRID_CONTEXT_WINDOW}" \
    --grid_max_tokens_per_type "${GRID_MAX_TOKENS_PER_TYPE}" \
    --grid_batch_size "${GRID_BATCH_SIZE}" \
    $( [ "$(to_bool "${DEFAULT_USE_CLS}")" -eq 1 ] && echo "--use_cls" ) \
    $( [ "$(to_bool "${DEFAULT_USE_CONTEXT}")" -eq 1 ] && echo "--use_context" ) \
    --context_window "${DEFAULT_CONTEXT_WINDOW}" \
    --max_tokens_per_type "${DEFAULT_MAX_TOKENS_PER_TYPE}" \
    --batch_size "${DEFAULT_BATCH_SIZE}"
fi

# =========================
# OUTPUTS YOU’LL FIND
# =========================
# For each model (xlmr/mbert), the Python script writes:
#   ${BASE_OUTDIR}/<model>/
#     aggregate_results_<model>.csv                 # per-config, per-layer means + ALL row (overall mean over layers)
#     best_config_<model>.csv                       # single-row CSV: the best config (max overall mean)
#     best_config_layers_<model>__<cfg_tag>.csv     # layer-by-layer means for the winning config
#
# And for each config:
#   ${BASE_OUTDIR}/<model>/<cfg_tag>/
#     cosine_to_run_layerL__<cfg_tag>.csv           # per-layer result CSVs (for ALL layers, L=1..num_layers)
#     heatmap_cosine_layers_<model>__<cfg_tag>.png
#     Cosine similarities vs run (<model>) — Layer X__<cfg_tag>.png (bar charts)
#     summary_top_worst_cosine_<model>__<cfg_tag>.csv

echo
echo "[DONE] Sweep finished. See per-model aggregation in:"
if [[ "${RUN_BOTH}" == "true" ]]; then
  echo "  ${BASE_OUTDIR}/xlmr/aggregate_results_xlmr.csv"
  echo "  ${BASE_OUTDIR}/mbert/aggregate_results_mbert.csv"
else
  echo "  ${BASE_OUTDIR}/${MODEL}/aggregate_results_${MODEL}.csv"
fi
