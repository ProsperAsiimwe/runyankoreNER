#!/bin/bash
set -euo pipefail

# =========================
# Config
# =========================
export CUDA_VISIBLE_DEVICES=1

# Early stop (for core mptc.py only — uses avg top-K layer Means)
export EARLY_STOPPING_PATIENCE=2
export METRIC_COLUMN="Mean"
export WARMUP_STEPS=2
export TOP_K_AGG=3

# Runtime knobs
export USE_FP16=true           # toggle AMP
export MAX_LENGTH=96           # token budget for spans + context

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

# ---- Paths (adjust if needed) ----
PY_CORE="rq2_train_scripts/Embeddings/CLEAN/mptc.py"
PY_PER_ENTITY="rq2_train_scripts/Embeddings/CLEAN/mptc_per_entity_similarity.py"
PY_ALT_MEAS="rq2_train_scripts/Embeddings/CLEAN/mptc_alt_measures_similarity.py"
PY_SWD="rq2_train_scripts/Embeddings/CLEAN/mptc_distribution_distance.py"

# Optional: override layers explicitly (comma list) e.g. "1,6,12"
# If empty, layers are auto-detected from the model config (all layers).
EXPLICIT_LAYERS=""

# =========================
# Helpers
# =========================

# Return comma-separated hidden-layer indices (1..L) for a model key
get_layers_csv() {
  local model_key="$1"  # "xlmr" | "mbert"
  if [[ -n "$EXPLICIT_LAYERS" ]]; then
    echo "$EXPLICIT_LAYERS"; return
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

# Read the 'Mean' value from a layer CSV (core mptc.py output)
read_layer_mean() {
  local csv_file="$1"; local col="$2"
  if [[ ! -f "$csv_file" ]]; then echo "NA"; return; fi
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

# Convenience wrapper for scripts that accept common flags (and optionally title suffix)
run_py_common() {
  local script="$1"; shift
  local model="$1"; shift
  local outdir="$1"; shift
  local use_cls="$1"; shift
  local use_ctx="$1"; shift
  local ctx_win="$1"; shift
  local max_tokens="$1"; shift
  local with_title="$1"; shift       # "yes" or "no"
  local title_suffix="$1"; shift     # ignored if with_title=no

  local CLS_FLAG=""; [[ "$use_cls" == "true" ]] && CLS_FLAG="--use_cls"
  local CTX_FLAG=""; [[ "$use_ctx" == "true" ]] && CTX_FLAG="--use_context"
  local FP16_FLAG=""; [[ "$USE_FP16" == "true" ]] && FP16_FLAG="--fp16"

  if [[ "$with_title" == "yes" ]]; then
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 "$script" \
      --model_type "$model" \
      $CLS_FLAG \
      $CTX_FLAG \
      $FP16_FLAG \
      --span_mode bio \
      --pool mean \
      --context_window "$ctx_win" \
      --max_tokens_per_type "$max_tokens" \
      --max_length "$MAX_LENGTH" \
      --title_suffix "$title_suffix" \
      --output_dir "$outdir"
  else
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 "$script" \
      --model_type "$model" \
      $CLS_FLAG \
      $CTX_FLAG \
      $FP16_FLAG \
      --span_mode bio \
      --pool mean \
      --context_window "$ctx_win" \
      --max_tokens_per_type "$max_tokens" \
      --max_length "$MAX_LENGTH" \
      --output_dir "$outdir"
  fi
}

# =========================
# Main grid
# =========================

for MODEL in "xlmr" "mbert"; do
  LAYERS_CSV="$(get_layers_csv "$MODEL")"
  IFS=',' read -r -a LAYERS <<< "$LAYERS_CSV"

  for i in "${!configs[@]}"; do
    IFS=' ' read -r USE_CLS USE_CONTEXT <<< "${configs[$i]}"

    for CONTEXT_WINDOW in "${CONTEXT_WINDOWS[@]}"; do
      # Early-stop state for this (model, config, context) triplet
      KEY="agg_${MODEL}_CFG${i}_CTX${CONTEXT_WINDOW}"
      export "BEST_$KEY"="0"
      export "DECLINE_$KEY"="0"
      export "STEPS_$KEY"="0"

      for MAX_TOKENS_PER_TYPE in "${TOKEN_LIMITS[@]}"; do

        # Consistent folder tree
        ROOT="rq2_train_scripts/Embeddings/CLEAN/outputs/mptc/${MODEL}/config_${i}_cls${USE_CLS}_ctx${USE_CONTEXT}/tokens_${MAX_TOKENS_PER_TYPE}_ctx${CONTEXT_WINDOW}"
        OUTDIR_CORE="${ROOT}/core"
        OUTDIR_PER="${ROOT}/per_entity"
        OUTDIR_ALT="${ROOT}/alt_measures"
        OUTDIR_SWD="${ROOT}/swd"
        mkdir -p "$OUTDIR_CORE" "$OUTDIR_PER" "$OUTDIR_ALT" "$OUTDIR_SWD"

        echo ""
        echo "=========================================================="
        echo "[INFO] MODEL=$MODEL | CONFIG=$i | TOKENS=$MAX_TOKENS_PER_TYPE | CTX=$CONTEXT_WINDOW"
        echo "USE_CLS=$USE_CLS, USE_CONTEXT=$USE_CONTEXT"
        echo "Layers: ${LAYERS[*]}"
        echo "Saving under: $ROOT/{core,per_entity,alt_measures,swd}"
        echo "=========================================================="

        TITLE_SUFFIX="L=all, tokens=${MAX_TOKENS_PER_TYPE}, ctx=${CONTEXT_WINDOW}, cls=${USE_CLS}, fp16=${USE_FP16}"

        # ---------------- Core (mptc.py) ----------------
        run_py_common "$PY_CORE" "$MODEL" "$OUTDIR_CORE" "$USE_CLS" "$USE_CONTEXT" "$CONTEXT_WINDOW" "$MAX_TOKENS_PER_TYPE" "yes" "$TITLE_SUFFIX"

        # -------- Aggregate Early Stopping over layers (avg top-K) --------
        SCORES=()
        for L in "${LAYERS[@]}"; do
          CSV_FILE="$OUTDIR_CORE/${MODEL}/cosine_to_run_layer${L}.csv"
          SCORE="$(read_layer_mean "$CSV_FILE" "$METRIC_COLUMN")"
          [[ -n "$SCORE" ]] && SCORES+=("$SCORE")
        done
        AGG_SCORE=$(printf "%s\n" "${SCORES[@]}" | grep -E '^[0-9]+([.][0-9]+)?$' | sort -gr | head -"$TOP_K_AGG" | awk '{s+=$1; n+=1} END{ if (n==0) print "NA"; else printf "%.6f\n", s/n }')

        PREV_SCORE_VAR="BEST_$KEY"
        DECLINE_VAR="DECLINE_$KEY"
        STEP_VAR="STEPS_$KEY"

        prev="${!PREV_SCORE_VAR}"
        decline="${!DECLINE_VAR}"
        steps="${!STEP_VAR}"

        steps=$((steps+1))
        export "STEPS_$KEY=$steps"

        if [[ "$steps" -le "$WARMUP_STEPS" ]]; then
          is_less=$(lt_float "$AGG_SCORE" "$prev")
          if [[ "$is_less" -ne 0 ]]; then export "BEST_$KEY"="$AGG_SCORE"; fi
          export "DECLINE_$KEY"="0"
          echo "[AGG-ES] $KEY | Warm-up $steps/${WARMUP_STEPS} | Agg: ${AGG_SCORE:-NA} | Best: ${!PREV_SCORE_VAR}"
        else
          is_less=$(lt_float "$AGG_SCORE" "$prev")
          if [[ "$is_less" -eq 0 ]]; then
            decline=$((decline+1))
          else
            decline=0
            prev="$AGG_SCORE"
          fi
          export "BEST_$KEY"="$prev"
          export "DECLINE_$KEY"="$decline"
          echo "[AGG-ES] $KEY | Agg: ${AGG_SCORE:-NA} | Best: $prev | Declines: $decline"
        fi

        # ---------------- Side-by-side analyses ----------------

        # Per-entity cosine (no --title_suffix)
        run_py_common "$PY_PER_ENTITY" "$MODEL" "$OUTDIR_PER" "$USE_CLS" "$USE_CONTEXT" "$CONTEXT_WINDOW" "$MAX_TOKENS_PER_TYPE" "no" ""

        # Alternative measures (Euclid, centered-cos, CKA) (no --title_suffix)
        run_py_common "$PY_ALT_MEAS" "$MODEL" "$OUTDIR_ALT" "$USE_CLS" "$USE_CONTEXT" "$CONTEXT_WINDOW" "$MAX_TOKENS_PER_TYPE" "no" ""

        # Distribution distance (SWD) — this script does NOT accept --use_cls/--use_context/--context_window/title_suffix
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 "$PY_SWD" \
          --model_type "$MODEL" \
          $([ "$USE_FP16" = true ] && echo "--fp16") \
          --span_mode bio \
          --pool mean \
          --max_tokens_per_type "$MAX_TOKENS_PER_TYPE" \
          --max_length "$MAX_LENGTH" \
          --samples_per_tag 200 \
          --num_projections 256 \
          --output_dir "$OUTDIR_SWD"

        # ---------------- Early-stop decision ----------------
        declines="${!DECLINE_VAR}"   # safe because you initialized DECLINE_$KEY="0" earlier
        
        if [[ "$steps" -gt "$WARMUP_STEPS" && "$declines" -ge "$EARLY_STOPPING_PATIENCE" ]]; then
          echo "[EARLY STOP] $KEY hit $declines declines. Skipping remaining TOKEN_LIMITS for this config/context."
          break
        fi

      done # TOKEN_LIMITS
    done   # CONTEXT_WINDOWS
  done     # configs
done       # models
