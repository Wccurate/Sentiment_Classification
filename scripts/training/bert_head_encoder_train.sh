#!/bin/bash

set -e


# GPU settings (optional)
CUDA_VISIBLE_DEVICES="0"  # Specify GPU ID(s), e.g., "0" or "0,1,2,3"

# Learning rates (separate for head and encoder)
HEAD_LR=3e-4           # Classification head learning rate
ENCODER_LR=3e-5        # Encoder (BERT body) learning rate (only used if TRAIN_ENCODER=true)
TRAIN_ENCODER=true    # Set to "true" to train encoder, "false" to freeze encoder
if [ "${TRAIN_ENCODER}" = "true" ]; then
    METHOD="head_encoder"
else
    METHOD="head_only"
fi


# ========== Configuration ==========
WANDB_API_KEY="xxx"
DATASET="project"  # Options: atis, hwu64, snips, clinc_oos, project
CLINC_VERSION="plus"  # For clinc_oos: small, plus, imbalanced
DATASET_DIR="data/raw/${DATASET}"


if [ "${DATASET}" = "project" ]; then
    OUTPUT_PROJECT_BASE="ee6483"
else
    OUTPUT_PROJECT_BASE="intent_exp"
fi

BERT_VERSION="bert"
MODEL_PATH="/usr1/home/s125mdg41_03/models/hf_models/bert_base_cased_google_bert"
TOKENIZER_PATH="/usr1/home/s125mdg41_03/models/hf_models/bert_base_cased_google_bert"
OUTPUT_DIR="/usr1/home/s125mdg41_03/models/${OUTPUT_PROJECT_BASE}/${BERT_VERSION}_${DATASET}_hf_${METHOD}"

# Training hyperparameters
EPOCHS=100
BATCH_SIZE=64
MAX_LENGTH=128

WEIGHT_DECAY=0.01
WARMUP_RATIO=0.06
VAL_RATIO=0.1

# Accelerate settings
MIXED_PRECISION="no"  # Options: no, fp16, bf16
GRAD_ACCUM_STEPS=1

SEED=42
SAVE_BEST_ONLY=""  # Set to "--save-best-only" to save disk space

# W&B settings
# WANDB_PROJECT=""
if [ "${DATASET}" = "project" ]; then
    WANDB_PROJECT="Sentiment_Project"
else
    WANDB_PROJECT="Intent_Recognition"
fi
WANDB_ENTITY=""  # Leave empty to use default
WANDB_RUN_NAME="${BERT_VERSION}_${METHOD}_${DATASET}_experiment"
WANDB_TAGS="albert,${DATASET}"



# ========== Run Training ==========
echo "========================================"
echo " ALBERT Classifier Head Fine-tuning"
echo "========================================"
echo "Dataset: ${DATASET}"
echo "Output: ${OUTPUT_DIR}"
echo "Epochs: ${EPOCHS}, Batch Size: ${BATCH_SIZE}"
if [ "${TRAIN_ENCODER}" = "true" ]; then
    echo "Training Mode: Full (Encoder + Head)"
    echo "Head Learning Rate: ${HEAD_LR}"
    echo "Encoder Learning Rate: ${ENCODER_LR}"
else
    echo "Training Mode: Head Only"
    echo "Head Learning Rate: ${HEAD_LR}"
fi
echo "GPU(s): ${CUDA_VISIBLE_DEVICES}"
echo "========================================"

# Build training command
TRAIN_CMD="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python train_bert_hf_head.py \
    --dataset \"${DATASET}\" \
    --clinc-version \"${CLINC_VERSION}\" \
    --dataset-dir \"${DATASET_DIR}\" \
    --model-path \"${MODEL_PATH}\" \
    --tokenizer-path \"${TOKENIZER_PATH}\" \
    --output-dir \"${OUTPUT_DIR}\" \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --max-length ${MAX_LENGTH} \
    --head-lr ${HEAD_LR} \
    --encoder-lr ${ENCODER_LR}"

# Add --train-encoder flag if enabled
if [ "${TRAIN_ENCODER}" = "true" ]; then
    TRAIN_CMD="${TRAIN_CMD} --train-encoder"
fi

# Add remaining arguments
TRAIN_CMD="${TRAIN_CMD} \
    --weight-decay ${WEIGHT_DECAY} \
    --warmup-ratio ${WARMUP_RATIO} \
    --val-ratio ${VAL_RATIO} \
    --mixed-precision \"${MIXED_PRECISION}\" \
    --grad-accum-steps ${GRAD_ACCUM_STEPS} \
    --seed ${SEED} \
    --use-wandb \
    --wandb-api-key \"${WANDB_API_KEY}\" \
    --wandb-project \"${WANDB_PROJECT}\" \
    --wandb-entity \"${WANDB_ENTITY}\" \
    --wandb-run-name \"${WANDB_RUN_NAME}\" \
    --wandb-tags \"${WANDB_TAGS}\" \
    ${SAVE_BEST_ONLY}"

# Execute training command
eval ${TRAIN_CMD}

echo ""
echo "========================================"
echo "Training complete!"
echo "Checkpoints saved to: ${OUTPUT_DIR}"
echo "========================================"
