#!/bin/bash
# Training script for BERT Classifier Head fine-tuning

set -e

# GPU settings (optional)
CUDA_VISIBLE_DEVICES="2"  # Specify GPU ID(s), e.g., "0" or "0,1,2,3"


# ========== Configuration ==========
WANDB_API_KEY="xxx"
DATASET="hwu64"  # Options: atis, hwu64, snips, clinc_oos, project
CLINC_VERSION="plus"  # For clinc_oos: small, plus, imbalanced
DATASET_DIR="data/raw/${DATASET}"
PROMPT_TUNING_INIT="text"  # Options: random, text
PROMPT_TUNING_INIT_TEXT="classify the intent of this text:"
EARLY_STOPPING_PATIENCE=20



MODEL_PATH="/usr1/home/s125mdg41_03/models/hf_models/bert_base_cased_google_bert"
TOKENIZER_PATH="/usr1/home/s125mdg41_03/models/hf_models/bert_base_cased_google_bert"
OUTPUT_DIR="/usr1/home/s125mdg41_03/models/intent_exp/bert_${DATASET}_hf_p_tuning"

# Training hyperparameters
EPOCHS=100
BATCH_SIZE=128
MAX_LENGTH=128
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.06
VAL_RATIO=0.1
SCHEDULER_TYPE="cosine"
PROMPT_LR=3e-3
CLASSIFIER_LR=5e-4


# Accelerate settings
MIXED_PRECISION="no"  # Options: no, fp16, bf16
GRAD_ACCUM_STEPS=1

SEED=42
SAVE_BEST_ONLY=""  # Set to "--save-best-only" to save disk space

# W&B settings
WANDB_PROJECT="Intent_Recognition"
WANDB_ENTITY=""  # Leave empty to use default
WANDB_RUN_NAME="bert_${DATASET}_hf_p_tuning_experiment"
WANDB_TAGS="bert,classifier-head,${DATASET}"



# ========== Run Training ==========
echo "========================================"
echo " BERT Classifier Head Fine-tuning"
echo "========================================"
echo "Dataset: ${DATASET}"
echo "Output: ${OUTPUT_DIR}"
echo "Epochs: ${EPOCHS}, Batch Size: ${BATCH_SIZE}"
echo "Learning Rate: ${LR} (Head: ${LR}*10)"
echo "GPU(s): ${CUDA_VISIBLE_DEVICES}"
echo "========================================"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python train_bert_prompt_tuning.py \
    --dataset "${DATASET}" \
    --clinc-version "${CLINC_VERSION}" \
    --dataset-dir "${DATASET_DIR}" \
    --model-path "${MODEL_PATH}" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --max-length ${MAX_LENGTH} \
    --num-virtual-tokens 50 \
    --prompt-tuning-init ${PROMPT_TUNING_INIT} \
    --prompt-tuning-init-text "${PROMPT_TUNING_INIT_TEXT}" \
    --prompt-lr ${PROMPT_LR} \
    --classifier-lr ${CLASSIFIER_LR} \
    --scheduler-type ${SCHEDULER_TYPE} \
    --weight-decay ${WEIGHT_DECAY} \
    --warmup-ratio ${WARMUP_RATIO} \
    --val-ratio ${VAL_RATIO} \
    --mixed-precision "${MIXED_PRECISION}" \
    --grad-accum-steps ${GRAD_ACCUM_STEPS} \
    --early-stopping-patience ${EARLY_STOPPING_PATIENCE} \
    --seed ${SEED} \
    --use-wandb \
    --wandb-api-key "${WANDB_API_KEY}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-entity "${WANDB_ENTITY}" \
    --wandb-run-name "${WANDB_RUN_NAME}" \
    --wandb-tags "${WANDB_TAGS}" \
    ${SAVE_BEST_ONLY}

echo ""
echo "========================================"
echo "Training complete!"
echo "Checkpoints saved to: ${OUTPUT_DIR}"
echo "========================================"