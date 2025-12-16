#!/bin/bash
set -euo pipefail

# =========================
# Config
# =========================
export CUDA_VISIBLE_DEVICES=0

# Runtime knobs
export USE_FP16=true
export MAX_LENGTH=96

# Sweep values (edit as needed)
declare -a MODELS=("afroxlmr" "xlmr" "mbert")
declare -a TOKEN_LIMITS=(200 300 400 500)
declare -a CONTEXT_WINDOWS=(0)

# CLS / Context combos (kept for directory compatibility)
declare -a configs=(
  "false false"
)

# Path to SWD script
PY_SWD="rq2_train_scripts/Embeddings/CLEAN/combined_distribution_distance.py"

# =========================
# Main
# =========================

for MODEL in "${MODELS[@]}"; do
  for i in "${!configs[@]}"; do
    IFS=' ' read -r USE_CLS USE_CONTEXT <<< "${configs[$i]}"

    for CONTEXT_WINDOW in "${CONTEXT_WINDOWS[@]}"; do
      for MAX_TOKENS_PER_TYPE in "${TOKEN_LIMITS[@]}"; do

        ROOT="rq2_train_scripts/Embeddings/CLEAN/outputs/combined/${MODEL}/config_${i}_cls${USE_CLS}_ctx${USE_CONTEXT}/tokens_${MAX_TOKENS_PER_TYPE}_ctx${CONTEXT_WINDOW}"
        OUTDIR_SWD="${ROOT}/swd"
        mkdir -p "$OUTDIR_SWD"

        echo ""
        echo "=========================================================="
        echo "[SWD] MODEL=$MODEL | TOKENS=$MAX_TOKENS_PER_TYPE | CTX=$CONTEXT_WINDOW"
        echo "=========================================================="

        FP16_FLAG=""
        [[ "$USE_FP16" == "true" ]] && FP16_FLAG="--fp16"

        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 "$PY_SWD" \
          --model_type "$MODEL" \
          $FP16_FLAG \
          --span_mode bio \
          --pool mean \
          --max_tokens_per_type "$MAX_TOKENS_PER_TYPE" \
          --max_length "$MAX_LENGTH" \
          --samples_per_tag 200 \
          --num_projections 256 \
          --output_dir "$OUTDIR_SWD"

      done
    done
  done
done

echo ""
echo "All SWD runs completed successfully."
