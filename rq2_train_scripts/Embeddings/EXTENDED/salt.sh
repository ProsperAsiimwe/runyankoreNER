#!/bin/bash
set -euo pipefail

# =========================
# GLOBAL SETTINGS
# =========================
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export SEED=${SEED:-42}

# Path to your updated Python script
PY_SCRIPT="${PY_SCRIPT:-rq2_train_scripts/Embeddings/EXTENDED/salt.py}"

# Base output directory
BASE_OUTDIR="${BASE_OUTDIR:-rq2_train_scripts/Embeddings/EXTENDED/outputs/salt_cosine}"

# Models
RUN_BOTH="${RUN_BOTH:-true}"
MODEL="${MODEL:-xlmr}"

# (NEW) Optional bootstrap iterations for rho stability (0 disables)
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-0}"

# =========================
# SWEEP SETTINGS
# =========================
USE_CLS_LIST=(${USE_CLS_LIST:-0 1})
USE_CONTEXT_LIST=(${USE_CONTEXT_LIST:-0 1})
CONTEXT_WINDOWS=(${CONTEXT_WINDOWS:-0 2 4})
TOKEN_LIMITS=(${TOKEN_LIMITS:-250 500})
BATCH_SIZES=(${BATCH_SIZES:-32 64})

DEFAULT_CONTEXT_WINDOW="${DEFAULT_CONTEXT_WINDOW:-2}"
DEFAULT_MAX_TOKENS_PER_TYPE="${DEFAULT_MAX_TOKENS_PER_TYPE:-500}"
DEFAULT_BATCH_SIZE="${DEFAULT_BATCH_SIZE:-64}"
DEFAULT_USE_CLS="${DEFAULT_USE_CLS:-0}"
DEFAULT_USE_CONTEXT="${DEFAULT_USE_CONTEXT:-0}"

# =========================
# HELPERS
# =========================
join_by_comma () { local IFS=','; echo "$*"; }
to_bool () {
  case "$1" in 1|true|TRUE|True|yes|y|Y) echo 1 ;; *) echo 0 ;; esac
}

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
echo "grid_use_cls=${GRID_USE_CLS}"
echo "grid_use_context=${GRID_USE_CONTEXT}"
echo "grid_context_window=${GRID_CONTEXT_WINDOW}"
echo "grid_max_tokens_per_type=${GRID_MAX_TOKENS_PER_TYPE}"
echo "grid_batch_size=${GRID_BATCH_SIZE}"
echo "====================================================="
echo

mkdir -p "${BASE_OUTDIR}"

# =========================
# RUN
# =========================
if [[ "${RUN_BOTH}" == "true" ]]; then
  echo "[RUN] Both models (xlmr + mbert)"
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
# OUTPUTS YOU’LL FIND (updated)
# =========================
# Per model (xlmr/mbert):
#   ${BASE_OUTDIR}/<model>/best_layers_by_overlap_<model>.csv
#     - one row per config: best_layer_by_overlap, best_layer_rho
#
# Per config:
#   ${BASE_OUTDIR}/<model>/<cfg_tag>/
#     cosine_to_run_layerL__<cfg_tag>.csv         # weighted cosine per layer
#     heatmap_cosine_layers_<model>__<cfg_tag>.png
#     Cosine similarities vs run (<model>) — Layer X__<cfg_tag>.png
#     summary_top_worst_cosine_<model>__<cfg_tag>.csv
#     entity_overlap_vs_run__<cfg_tag>.csv        # entity overlap vs target
#     layer_rhos_vs_overlap__<cfg_tag>.csv        # Spearman rho per layer
#     layer_rhos_bootstrap__<cfg_tag>.csv         # (present if BOOTSTRAP_ITERS>0)

echo
echo "[DONE] Sweep finished. See per-model summary in:"
if [[ "${RUN_BOTH}" == "true" ]]; then
  echo "  ${BASE_OUTDIR}/xlmr/best_layers_by_overlap_xlmr.csv"
  echo "  ${BASE_OUTDIR}/mbert/best_layers_by_overlap_mbert.csv"
else
  echo "  ${BASE_OUTDIR}/${MODEL}/best_layers_by_overlap_${MODEL}.csv"
fi
