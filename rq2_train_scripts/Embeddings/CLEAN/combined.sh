#!/bin/bash
set -euo pipefail

# =========================
# Config
# =========================
export CUDA_VISIBLE_DEVICES=0

# Early stop (for core combined.py only — uses avg top-K layer Means)
export EARLY_STOPPING_PATIENCE=2
export METRIC_COLUMN="Mean"
export WARMUP_STEPS=2
export TOP_K_AGG=3

# Runtime knobs
export USE_FP16=true
export MAX_LENGTH=96

# Tuning ranges
declare -a TOKEN_LIMITS=(200 300 400 500)
declare -a CONTEXT_WINDOWS=(0 1 2 3)

# CLS / Context combos
declare -a configs=(
  "false false"
  "true false"
  "false true"
  "true true"
)

# ---- Paths ----
PY_CORE="rq2_train_scripts/Embeddings/CLEAN/combined.py"
PY_PER_ENTITY="rq2_train_scripts/Embeddings/CLEAN/combined_per_entity_similarity.py"
PY_ALT_MEAS="rq2_train_scripts/Embeddings/CLEAN/combined_alt_measures_similarity.py"
PY_SWD="rq2_train_scripts/Embeddings/CLEAN/combined_distribution_distance.py"

EXPLICIT_LAYERS=""

# =========================
# Helpers
# =========================

# Return comma-separated hidden-layer indices (1..L)
get_layers_csv() {
  local model_key="$1"
  if [[ -n "$EXPLICIT_LAYERS" ]]; then
    echo "$EXPLICIT_LAYERS"; return
  fi
  python3 - "$model_key" <<'PY'
import sys
from transformers import AutoModel

MAP = {
    "xlmr": "xlm-roberta-base",
    "mbert": "bert-base-multilingual-cased",
    "afro-xlmr": "Davlan/afro-xlmr-base",
}

m = AutoModel.from_pretrained(MAP[sys.argv[1]])
L = m.config.num_hidden_layers
print(",".join(str(i) for i in range(1, L + 1)))
PY
}

read_layer_mean() {
  local csv_file="$1"; local col="$2"
  [[ ! -f "$csv_file" ]] && echo "NA" && return
  awk -F',' -v COL="$col" '
    NR==1 { for (i=1;i<=NF;i++) if ($i==COL) c=i; next }
    NR==2 { if (c>0) print $c; else print "NA"; exit }
  ' "$csv_file"
}

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

run_py_common() {
  local script="$1"; shift
  local model="$1"; shift
  local outdir="$1"; shift
  local use_cls="$1"; shift
  local use_ctx="$1"; shift
  local ctx_win="$1"; shift
  local max_tokens="$1"; shift
  local with_title="$1"; shift
  local title_suffix="$1"; shift

  local CLS_FLAG=""; [[ "$use_cls" == "true" ]] && CLS_FLAG="--use_cls"
  local CTX_FLAG=""; [[ "$use_ctx" == "true" ]] && CTX_FLAG="--use_context"
  local FP16_FLAG=""; [[ "$USE_FP16" == "true" ]] && FP16_FLAG="--fp16"

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
    ${with_title:+--title_suffix "$title_suffix"} \
    --output_dir "$outdir"
}

# =========================
# Main grid
# =========================

for MODEL in "xlmr" "mbert" "afro-xlmr"; do
  LAYERS_CSV="$(get_layers_csv "$MODEL")"
  IFS=',' read -r -a LAYERS <<< "$LAYERS_CSV"

  for i in "${!configs[@]}"; do
    IFS=' ' read -r USE_CLS USE_CONTEXT <<< "${configs[$i]}"

    for CONTEXT_WINDOW in "${CONTEXT_WINDOWS[@]}"; do
      KEY="agg_${MODEL}_CFG${i}_CTX${CONTEXT_WINDOW}"
      export "BEST_$KEY=0"
      export "DECLINE_$KEY=0"
      export "STEPS_$KEY=0"

      for MAX_TOKENS_PER_TYPE in "${TOKEN_LIMITS[@]}"; do

        ROOT="rq2_train_scripts/Embeddings/CLEAN/outputs/combined/${MODEL}/config_${i}_cls${USE_CLS}_ctx${USE_CONTEXT}/tokens_${MAX_TOKENS_PER_TYPE}_ctx${CONTEXT_WINDOW}"
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
        echo "=========================================================="

        TITLE_SUFFIX="L=all, tokens=${MAX_TOKENS_PER_TYPE}, ctx=${CONTEXT_WINDOW}, cls=${USE_CLS}, fp16=${USE_FP16}"

        # ---------------- Core ----------------
        run_py_common "$PY_CORE" "$MODEL" "$OUTDIR_CORE" "$USE_CLS" "$USE_CONTEXT" "$CONTEXT_WINDOW" "$MAX_TOKENS_PER_TYPE" "yes" "$TITLE_SUFFIX"

        # -------- Aggregate Early Stopping --------
        SCORES=()
        for L in "${LAYERS[@]}"; do
          CSV_FILE="$OUTDIR_CORE/${MODEL}/cosine_to_run_layer${L}.csv"
          SCORE="$(read_layer_mean "$CSV_FILE" "$METRIC_COLUMN")"
          [[ -n "$SCORE" ]] && SCORES+=("$SCORE")
        done

        AGG_SCORE=$(printf "%s\n" "${SCORES[@]}" | grep -E '^[0-9]' | sort -gr | head -"$TOP_K_AGG" | awk '{s+=$1; n+=1} END{ if (n==0) print "NA"; else printf "%.6f\n", s/n }')

        PREV="BEST_$KEY"
        DECL="DECLINE_$KEY"
        STEP="STEPS_$KEY"

        steps=$(( ${!STEP} + 1 ))
        export "$STEP=$steps"

        is_less=$(lt_float "$AGG_SCORE" "${!PREV}")
        if [[ "$steps" -le "$WARMUP_STEPS" ]]; then
          [[ "$is_less" -ne 0 ]] && export "$PREV=$AGG_SCORE"
          export "$DECL=0"
        else
          if [[ "$is_less" -eq 0 ]]; then
            export "$DECL=$(( ${!DECL} + 1 ))"
          else
            export "$DECL=0"
            export "$PREV=$AGG_SCORE"
          fi
        fi

        echo "[AGG-ES] $KEY | Agg=$AGG_SCORE | Best=${!PREV} | Declines=${!DECL}"

        # ---------------- Side analyses ----------------
        run_py_common "$PY_PER_ENTITY" "$MODEL" "$OUTDIR_PER" "$USE_CLS" "$USE_CONTEXT" "$CONTEXT_WINDOW" "$MAX_TOKENS_PER_TYPE" "" ""
        run_py_common "$PY_ALT_MEAS" "$MODEL" "$OUTDIR_ALT" "$USE_CLS" "$USE_CONTEXT" "$CONTEXT_WINDOW" "$MAX_TOKENS_PER_TYPE" "" ""

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

        # ---------------- Early stop ----------------
        if [[ "$steps" -gt "$WARMUP_STEPS" && "${!DECL}" -ge "$EARLY_STOPPING_PATIENCE" ]]; then
          echo "[EARLY STOP] $KEY — stopping token sweep"
          break
        fi

      done
    done
  done
done
