#!/bin/bash
set -euo pipefail

# =========================
# GLOBAL SETTINGS
# =========================
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-2}   # Stop after 2 consecutive declines
export METRIC_COLUMN="${METRIC_COLUMN:-Mean}"

# New script parameters (tweak as you like)
export NUM_MIDDLE_LAYERS=${NUM_MIDDLE_LAYERS:-3}               # random middle layers (final layer is always added)
export BATCH_SIZE=${BATCH_SIZE:-64}
export SEED=${SEED:-42}

# Ranges for tuning
declare -a TOKEN_LIMITS=(200 300 400 500)
declare -a CONTEXT_WINDOWS=(0 1 2 3)

# All 8 flag combinations (USE_CLS USE_CONTEXT USE_HYBRID)
declare -a configs=(
  "false false false"
  "true  false false"
  "false true  false"
  "false false true"
  "true  true  false"
  "true  false true"
  "false true  true"
  "true  true  true"
)

# Paths
PY_SCRIPT="rq2_train_scripts/Embeddings/EXTENDED/combined_extended.py"
BASE_OUTDIR="rq2_train_scripts/Embeddings/EXTENDED/outputs/combined_extended"

# Helper: compute aggregate score from a CSV (mean of METRIC_COLUMN)
compute_score_from_csv () {
  local csv="$1"
  local col="${2:-Mean}"
  # Get column index from header
  local col_idx
  col_idx=$(head -n 1 "$csv" | awk -F',' -v C="$col" '{
    for (i=1;i<=NF;i++) if ($i==C) {print i; exit}
  }')

  if [[ -z "${col_idx:-}" ]]; then
    echo "NaN"
    return 0
  fi

  # Average over all data rows for that column, ignoring non-numerics and empty cells.
  awk -F',' -v i="$col_idx" '
    NR>1 {
      gsub(/\r/,"",$i);                    # strip CR if present
      if ($i ~ /^-?[0-9.]+([eE][-+]?[0-9]+)?$/) { sum+=$i; n++ }
    }
    END {
      if (n>0) printf("%.6f\n", sum/n);
      else print "NaN";
    }
  ' "$csv"
}

# Helper: pick the final-layer CSV (largest layerN) for a given OUTDIR
pick_final_layer_csv () {
  local outdir="$1"
  # Example files: pearson_to_run_layer12.csv
  # We create a sortable list "N path" then pick the highest N.
  ls -1 "$outdir"/pearson_to_run_layer*.csv 2>/dev/null \
    | sed -E 's/.*layer([0-9]+)\.csv/\1 &/' \
    | sort -n -k1,1 \
    | tail -n 1 \
    | cut -d' ' -f2
}

# =========================
# GRID SEARCH
# =========================
for MODEL in "xlmr" "mbert"; do
  for i in "${!configs[@]}"; do
    IFS=' ' read -r USE_CLS USE_CONTEXT USE_HYBRID <<< "${configs[$i]}"

    for CONTEXT_WINDOW in "${CONTEXT_WINDOWS[@]}"; do
      for MAX_TOKENS_PER_TYPE in "${TOKEN_LIMITS[@]}"; do

        OUTDIR="${BASE_OUTDIR}/${MODEL}/config_${i}_cls${USE_CLS}_ctx${USE_CONTEXT}_hyb${USE_HYBRID}/tokens_${MAX_TOKENS_PER_TYPE}_ctx${CONTEXT_WINDOW}"
        mkdir -p "$OUTDIR"

        echo ""
        echo "==============================================="
        echo "[INFO] MODEL=$MODEL | CONFIG=$i | TOKENS=$MAX_TOKENS_PER_TYPE | CTX=$CONTEXT_WINDOW"
        echo "USE_CLS=$USE_CLS, USE_CONTEXT=$USE_CONTEXT, USE_HYBRID=$USE_HYBRID"
        echo "NUM_MIDDLE_LAYERS=$NUM_MIDDLE_LAYERS, BATCH_SIZE=$BATCH_SIZE, SEED=$SEED"
        echo "Saving to $OUTDIR"
        echo "==============================================="

        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 "$PY_SCRIPT" \
          --model_type "$MODEL" \
          $( [ "$USE_CLS" = true ] && echo "--use_cls" ) \
          $( [ "$USE_CONTEXT" = true ] && echo "--use_context" ) \
          $( [ "$USE_HYBRID" = true ] && echo "--use_hybrid" ) \
          --context_window "$CONTEXT_WINDOW" \
          --max_tokens_per_type "$MAX_TOKENS_PER_TYPE" \
          --batch_size "$BATCH_SIZE" \
          --seed "$SEED" \
          --num_middle_layers "$NUM_MIDDLE_LAYERS" \
          --output_dir "$OUTDIR"

        # =========================
        # EARLY STOPPING CHECK
        # =========================
        # Look for the final-layer CSV (highest layer index)
        FINAL_CSV="$(pick_final_layer_csv "$OUTDIR" || true)"
        if [[ -n "${FINAL_CSV:-}" && -f "$FINAL_CSV" ]]; then
          SCORE="$(compute_score_from_csv "$FINAL_CSV" "$METRIC_COLUMN")"
        else
          echo "[WARN] No pearson_to_run_layer*.csv found in $OUTDIR — skipping early-stop check."
          continue
        fi

        # Track best + consecutive declines per (MODEL, config, CONTEXT_WINDOW)
        KEY="combo_${MODEL}_${i}_${CONTEXT_WINDOW}"
        PREV_SCORE_VAR="BEST_${KEY//[-.]/_}"     # sanitize for env var name
        DECLINE_VAR="DECLINE_${KEY//[-.]/_}"

        prev=${!PREV_SCORE_VAR:-"-inf"}
        decline=${!DECLINE_VAR:-0}

        # Compare numerically; handle NaN/-inf by treating them as very poor
        better=0
        if [[ "$SCORE" != "NaN" ]]; then
          if [[ "$prev" == "-inf" || "$(awk -v a="$SCORE" -v b="$prev" 'BEGIN{print (a>b) ? 1 : 0}')" -eq 1 ]]; then
            better=1
          fi
        fi

        if [[ $better -eq 1 ]]; then
          decline=0
          export ${PREV_SCORE_VAR}="$SCORE"
        else
          decline=$((decline+1))
        fi
        export ${DECLINE_VAR}=$decline

        echo "[INFO] $KEY | Score: $SCORE | Best: $prev | Declines: $decline"

        if [[ $decline -ge $EARLY_STOPPING_PATIENCE ]]; then
          echo "[EARLY STOP] $KEY reached $decline consecutive declines. Skipping rest of this combo."
          break
        fi

      done
    done
  done
done
