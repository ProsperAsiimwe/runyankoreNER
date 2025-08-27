#!/bin/bash
set -euo pipefail

# =========================
# Config
# =========================
export CUDA_VISIBLE_DEVICES=0
export EARLY_STOPPING_PATIENCE=2        # stop after N consecutive declines (post warm-up)
export METRIC_COLUMN="Mean"
export USE_FP16=true                    # set to false to disable AMP
export MAX_LENGTH=96                    # token budget for spans + context
export WARMUP_STEPS=2                   # try at least N token limits before allowing early stop
export TOP_K_AGG=3                      # aggregate = average of top-K layer Means

# Tuning ranges
declare -a TOKEN_LIMITS=(200 300 400 500)
declare -a CONTEXT_WINDOWS=(0 1 2 3)

# CLS / Context combos (hybrid removed)
declare -a configs=(
  "false false"
  "true false"
  "false true"
  "true true"
)

# Path to your rewritten Python driver (adjust if needed)
PY="rq2_train_scripts/Embeddings/CLEAN/combined.py"

# Optional: override layers explicitly (comma list) e.g. "1,6,12"
# If empty, we auto-detect from the model config (all layers).
EXPLICIT_LAYERS=""

# =========================
# Helpers
# =========================

# Return a comma-separated list of all hidden layer indices (1..L) for a model key
get_layers_csv() {
  local model_key="$1"  # "xlmr" or "mbert"
  if [[ -n "$EXPLICIT_LAYERS" ]]; then
    echo "$EXPLICIT_LAYERS"
    return
  fi
  python3 - "$model_key" <<'PY'
import sys
from transformers import AutoModel
MAP={"xlmr":"xlm-roberta-base","mbert":"bert-base-multilingual-cased"}
m=AutoModel.from_pretrained(MAP[sys.argv[1]])
L=m.config.num_hidden_layers
print(",".join(str(i) for i in range(1, L+1)))
PY
}

# Read the 'Mean' value from a layer CSV (cosine_to_run_layer{L}.csv)
read_layer_mean() {
  local csv_file="$1"
  local col="$2"
  if [[ ! -f "$csv_file" ]]; then
    echo "NA"
    return
  fi
  # Find the 'Mean' column from header (NR==1) then output the first data row (NR==2)
  awk -F',' -v COL="$col" '
    NR==1 { for (i=1;i<=NF;i++) if ($i==COL) c=i; next }
    NR==2 { if (c>0) print $c; else print "NA"; exit }
  ' "$csv_file"
}

# Compare two floats; treat NA as very small
lt_float() {
  python3 - "$@" <<'PY'
import sys, math
def f(x, d=-1e9):
    try:
        v=float(x)
        return v if not math.isnan(v) else d
    except: return d
a=f(sys.argv[1]); b=f(sys.argv[2])
print(0 if a < b else 1)
PY
}

# =========================
# Main grid
# =========================

for MODEL in "xlmr" "mbert"; do
  LAYERS_CSV="$(get_layers_csv "$MODEL")"
  IFS=',' read -r -a LAYERS <<< "$LAYERS_CSV"

  for i in "${!configs[@]}"; do
    IFS=' ' read -r USE_CLS USE_CONTEXT <<< "${configs[$i]}"

    # Reset aggregate early-stop counters per (MODEL, config, context)
    for CONTEXT_WINDOW in "${CONTEXT_WINDOWS[@]}"; do
      # These track across TOKEN_LIMITS for the same MODEL/config/context
      KEY="agg_${MODEL}_CFG${i}_CTX${CONTEXT_WINDOW}"
      export "BEST_$KEY"="0"
      export "DECLINE_$KEY"="0"
      export "STEPS_$KEY"="0"

      for MAX_TOKENS_PER_TYPE in "${TOKEN_LIMITS[@]}"; do

        OUTDIR="rq2_train_scripts/Embeddings/CLEAN/outputs/combined/${MODEL}/config_${i}_cls${USE_CLS}_ctx${USE_CONTEXT}/tokens_${MAX_TOKENS_PER_TYPE}_ctx${CONTEXT_WINDOW}"
        mkdir -p "$OUTDIR"

        echo ""
        echo "=========================================================="
        echo "[INFO] MODEL=$MODEL | CONFIG=$i | TOKENS=$MAX_TOKENS_PER_TYPE | CTX=$CONTEXT_WINDOW"
        echo "USE_CLS=$USE_CLS, USE_CONTEXT=$USE_CONTEXT"
        echo "Layers: ${LAYERS[*]}"
        echo "Saving to $OUTDIR"
        echo "=========================================================="

        # Optional flags
        CLS_FLAG=$([ "$USE_CLS" = true ] && echo "--use_cls" || echo "")
        CTX_FLAG=$([ "$USE_CONTEXT" = true ] && echo "--use_context" || echo "")
        FP16_FLAG=$([ "$USE_FP16" = true ] && echo "--fp16" || echo "")

        # Title suffix helps disambiguate plots/files later
        TITLE_SUFFIX="L=all, tokens=${MAX_TOKENS_PER_TYPE}, ctx=${CONTEXT_WINDOW}, cls=${USE_CLS}, fp16=${USE_FP16}"

        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 "$PY" \
          --model_type "$MODEL" \
          $CLS_FLAG \
          $CTX_FLAG \
          $FP16_FLAG \
          --span_mode bio \
          --pool mean \
          --context_window "$CONTEXT_WINDOW" \
          --max_tokens_per_type "$MAX_TOKENS_PER_TYPE" \
          --max_length "$MAX_LENGTH" \
          --title_suffix "$TITLE_SUFFIX" \
          --output_dir "$OUTDIR"

        # -------- Aggregate Early Stopping over layers (avg top-K) --------
        SCORES=()
        for L in "${LAYERS[@]}"; do
          CSV_FILE="$OUTDIR/${MODEL}/cosine_to_run_layer${L}.csv"
          SCORE="$(read_layer_mean "$CSV_FILE" "$METRIC_COLUMN")"
          [[ -n "$SCORE" ]] && SCORES+=("$SCORE")
        done

        # Compute average of top-K numeric scores
        AGG_SCORE=$(printf "%s\n" "${SCORES[@]}" | grep -E '^[0-9]+([.][0-9]+)?$' | sort -gr | head -"$TOP_K_AGG" | awk '{s+=$1; n+=1} END{ if (n==0) print "NA"; else printf "%.6f\n", s/n }')

        PREV_SCORE_VAR="BEST_$KEY"
        DECLINE_VAR="DECLINE_$KEY"
        STEP_VAR="STEPS_$KEY"

        prev="${!PREV_SCORE_VAR}"
        decline="${!DECLINE_VAR}"
        steps="${!STEP_VAR}"

        steps=$((steps+1))
        export "STEPS_$KEY=$steps"

        # During warm-up, just record best and continue
        if [[ "$steps" -le "$WARMUP_STEPS" ]]; then
          # Update best if current aggregate is valid and better
          is_less=$(lt_float "$AGG_SCORE" "$prev")
          if [[ "$is_less" -ne 0 ]]; then
            export "BEST_$KEY"="$AGG_SCORE"
          fi
          export "DECLINE_$KEY"="0"
          echo "[AGG-ES] $KEY | Warm-up $steps/${WARMUP_STEPS} | Agg: ${AGG_SCORE:-NA} | Best: ${!PREV_SCORE_VAR}"
          continue
        fi

        # Compare floats (NA treated as very small)
        is_less=$(lt_float "$AGG_SCORE" "$prev")
        if [[ "$is_less" -eq 0 ]]; then
          # score < prev
          decline=$((decline+1))
        else
          decline=0
          prev="$AGG_SCORE"
        fi

        export "BEST_$KEY"="$prev"
        export "DECLINE_$KEY"="$decline"
        echo "[AGG-ES] $KEY | Agg: ${AGG_SCORE:-NA} | Best: $prev | Declines: $decline"

        if [[ $decline -ge $EARLY_STOPPING_PATIENCE ]]; then
          echo "[EARLY STOP] $KEY hit $decline declines. Skipping remaining TOKEN_LIMITS for this config/context."
          # Skip to next CONTEXT/TOKENS combo
          break
        fi

      done # TOKEN_LIMITS
    done   # CONTEXT_WINDOWS
  done     # configs
done       # models
