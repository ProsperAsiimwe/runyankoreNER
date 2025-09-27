#!/bin/bash

# ===============================
# TRAINING WITH AfroXLMR
# ===============================
echo "Starting training with AfroXLMR..."

export DATA_DIR="data/TypologicalGroups/ZERO-SHOT/LinguaMeta_Intermediate/COMBINED/"
export OUTPUT_DIR="models/TypologicalGroups/ZERO-SHOT/LinguaMeta_Intermediate/combined_afroxlmr"
export HUGGINGFACE_MODEL_PATH="Davlan/afro-xlmr-base"

export MAX_LENGTH=164
export BATCH_SIZE=32
export NUM_EPOCHS=10
export SAVE_STEPS=5000
export SEED=1
export CUDA_VISIBLE_DEVICES=1

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


# ===============================
# TRAINING WITH mBERT
# ===============================
echo "Starting training with mBERT..."

export OUTPUT_DIR="models/TypologicalGroups/ZERO-SHOT/LinguaMeta_Intermediate/combined_mbert"
export HUGGINGFACE_MODEL_PATH="bert-base-multilingual-cased"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 code/train_ner.py \
  --data_dir "$DATA_DIR" \
  --model_type bert \
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


# ===============================
# TRAINING WITH XLM-R
# ===============================
echo "Starting training with XLM-R..."

export OUTPUT_DIR="models/TypologicalGroups/ZERO-SHOT/LinguaMeta_Intermediate/combined_xlmr"
export HUGGINGFACE_MODEL_PATH="xlm-roberta-base"

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

echo "All training jobs completed."
