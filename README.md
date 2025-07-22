 1️⃣ Validate the Environment Setup

Since your .env file already defines your hyperparameters, ensure that your environment variables are correctly loaded before running the scripts.

Run this before training:

source .env

## git reset --hard

<!-- export $(grep -v '^#' .env | xargs)
echo "Loaded environment variables." -->


✅ Running Training with .env Variables. In your terminal, paste this entire command:


CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 code/train_ner.py \
    --data_dir $DATA_DIR \
    --model_type xlmroberta \
    --model_name_or_path $BERT_MODEL \
    --output_dir $OUTPUT_DIR \
    --max_seq_length $MAX_LENGTH \
    --num_train_epochs $NUM_EPOCHS \
    --per_gpu_train_batch_size $BATCH_SIZE \
    --save_steps $SAVE_STEPS \
    --seed $SEED \
    --do_train \
    --do_eval \
    --do_predict