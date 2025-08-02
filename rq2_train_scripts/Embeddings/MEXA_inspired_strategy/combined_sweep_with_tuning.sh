#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export EARLY_STOPPING_PATIENCE=2  # Stop after 2 declines
export METRIC_COLUMN="Mean"

# Ranges for tuning
declare -a TOKEN_LIMITS=(200 300 400 500)
declare -a CONTEXT_WINDOWS=(0 1 2 3)

# All 8 flag combinations
declare -a configs=(
  "false false false"
  "true false false"
  "false true false"
  "false false true"
  "true true false"
  "true false true"
  "false true true"
  "true true true"
)

for MODEL in "xlmr" "mbert"; do
  for i in "${!configs[@]}"; do
    IFS=' ' read -r USE_CLS USE_CONTEXT USE_HYBRID <<< "${configs[$i]}"

    for CONTEXT_WINDOW in "${CONTEXT_WINDOWS[@]}"; do
      for MAX_TOKENS_PER_TYPE in "${TOKEN_LIMITS[@]}"; do

        OUTDIR="outputs/combined/${MODEL}/config_${i}_cls${USE_CLS}_ctx${USE_CONTEXT}_hyb${USE_HYBRID}/tokens_${MAX_TOKENS_PER_TYPE}_ctx${CONTEXT_WINDOW}"
        mkdir -p "$OUTDIR"

        echo ""
        echo "==============================================="
        echo "[INFO] MODEL=$MODEL | CONFIG=$i | TOKENS=$MAX_TOKENS_PER_TYPE | CTX=$CONTEXT_WINDOW"
        echo "USE_CLS=$USE_CLS, USE_CONTEXT=$USE_CONTEXT, USE_HYBRID=$USE_HYBRID"
        echo "Saving to $OUTDIR"
        echo "==============================================="

        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 combined.py \
          --model_type "$MODEL" \
          $( [ "$USE_CLS" = true ] && echo "--use_cls" ) \
          $( [ "$USE_CONTEXT" = true ] && echo "--use_context" ) \
          $( [ "$USE_HYBRID" = true ] && echo "--use_hybrid" ) \
          --context_window "$CONTEXT_WINDOW" \
          --max_tokens_per_type "$MAX_TOKENS_PER_TYPE" \
          --output_dir "$OUTDIR"

        # EARLY STOPPING CHECK
        CSV_FILE="$OUTDIR/similarity_to_run.csv"
        if [[ -f "$CSV_FILE" ]]; then
          SCORE=$(tail -n +2 "$CSV_FILE" | awk -F',' -v col="$METRIC_COLUMN" 'NR==1{for (i=1;i<=NF;i++) if ($i==col) c=i} END{print $c}')
          KEY="combo_${MODEL}_${i}_$CONTEXT_WINDOW"
          PREV_SCORE_VAR="BEST_$KEY"
          DECLINE_VAR="DECLINE_$KEY"

          prev=${!PREV_SCORE_VAR:-0}
          decline=${!DECLINE_VAR:-0}

          if (( $(echo "$SCORE < $prev" | bc -l) )); then
            ((decline++))
          else
            decline=0
          fi

          export BEST_$KEY=$SCORE
          export DECLINE_$KEY=$decline

          echo "[INFO] $KEY | Score: $SCORE | Best: $prev | Declines: $decline"

          if [[ $decline -ge $EARLY_STOPPING_PATIENCE ]]; then
            echo "[EARLY STOP] $KEY reached $decline consecutive declines. Skipping rest of this combo."
            break
          fi
        fi

      done
    done
  done
done
