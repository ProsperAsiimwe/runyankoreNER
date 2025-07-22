#!/bin/bash

# ========== EXPERIMENT CONFIGURATION ==========
export DATA_DIR="data/ZeroShotTransfer/pcm_COMBINED/"
export OUTPUT_DIR="models/ZeroShotTransfer/combined_pcm_runyankore_xlmr"
export HUGGINGFACE_MODEL_PATH="xlm-roberta-base"

export MAX_LENGTH=164
export BATCH_SIZE=32
export NUM_EPOCHS=10
export SAVE_STEPS=5000
export SEED=1
export CUDA_VISIBLE_DEVICES=0  # Change if running on a different GPU

# ========== RUN TRAINING ==========
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 code/train_ner.py \
  --data_dir "$DATA_DIR" \
  --model_type xlmroberta \
  --model_name_or_path "$HUGGINGFACE_MODEL_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --max_seq_length "$MAX_LENGTH" \
  --num_train_epochs "$NUM_EPOCHS" \
  --per_gpu_train_batch_size "$BATCH_SIZE" \
  --save_steps "$SAVE_STEPS" \
  --seed "$SEED" \
  --do_train \
  --do_eval \
  --do_predict \
  --overwrite_output_dir
