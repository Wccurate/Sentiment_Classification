#!/bin/bash
# Training script for BERT LoRA fine-tuning

set -e

# ========== Configuration ==========
WANDB_API_KEY="xxx"
DATASET="project"  # Options: atis, hwu64, snips, clinc_oos, project
CLINC_VERSION="plus"  # For clinc_oos: small, plus, imbalanced
DATASET_DIR="data/raw/${DATASET}"

# GPU settings (optional)
CUDA_VISIBLE_DEVICES="0"  # Specify GPU ID(s), e.g., "0" or "0,1,2,3"

MODEL_PATH="/usr1/home/s125mdg41_03/models/hf_models/bert_base_cased_google_bert"
TOKENIZER_PATH="/usr1/home/s125mdg41_03/models/hf_models/bert_base_cased_google_bert"
OUTPUT_DIR="/usr1/home/s125mdg41_03/models/ee6483/bert_lora_${DATASET}"

# Training hyperparameters
EPOCHS=20
BATCH_SIZE=128
MAX_LENGTH=128

# Learning rates (separate for LoRA and classifier head)
LORA_LR=3e-4           # LoRA adapter learning rate
HEAD_LR=1e-3           # Classifier head learning rate (typically higher than LoRA)

WEIGHT_DECAY=0.01
WARMUP_RATIO=0.06
VAL_RATIO=0.1

# LoRA configuration
LORA_R=8               # LoRA rank (typically 4, 8, 16, 32)
LORA_ALPHA=16          # LoRA alpha (typically 2*r)
LORA_DROPOUT=0.1       # LoRA dropout
LORA_TARGET_MODULES="query,key,value"  # Target modules for LoRA

# Accelerate settings
MIXED_PRECISION="no"   # Options: no, fp16, bf16
GRAD_ACCUM_STEPS=1
MAX_GRAD_NORM=1.0
SCHEDULER_TYPE="cosine"  # Options: linear, cosine

SEED=42
SAVE_BEST_ONLY=""      # Set to "--save-best-only" to save disk space
EARLY_STOPPING=0       # Early stopping patience (0 to disable)

# W&B settings
WANDB_PROJECT=""
WANDB_ENTITY=""        # Leave empty to use default
WANDB_RUN_NAME="bert_${DATASET}_lora_r${LORA_R}_experiment"
WANDB_TAGS="bert,lora,${DATASET}"


# ========== Run Training ==========
echo "========================================"
echo " BERT LoRA Fine-tuning"
echo "========================================"
echo "Dataset: ${DATASET}"
echo "Output: ${OUTPUT_DIR}"
echo "Epochs: ${EPOCHS}, Batch Size: ${BATCH_SIZE}"
echo "LoRA Config: r=${LORA_R}, alpha=${LORA_ALPHA}, dropout=${LORA_DROPOUT}"
echo "LoRA Target Modules: ${LORA_TARGET_MODULES}"
echo "LoRA Learning Rate: ${LORA_LR}"
echo "Head Learning Rate: ${HEAD_LR}"
echo "GPU(s): ${CUDA_VISIBLE_DEVICES}"
echo "========================================"

# Build training command
TRAIN_CMD="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python train_bert_lora.py \
    --dataset \"${DATASET}\" \
    --clinc-version \"${CLINC_VERSION}\" \
    --dataset-dir \"${DATASET_DIR}\" \
    --model-path \"${MODEL_PATH}\" \
    --tokenizer-path \"${TOKENIZER_PATH}\" \
    --output-dir \"${OUTPUT_DIR}\" \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --max-length ${MAX_LENGTH} \
    --lora-lr ${LORA_LR} \
    --head-lr ${HEAD_LR} \
    --weight-decay ${WEIGHT_DECAY} \
    --warmup-ratio ${WARMUP_RATIO} \
    --val-ratio ${VAL_RATIO} \
    --lora-r ${LORA_R} \
    --lora-alpha ${LORA_ALPHA} \
    --lora-dropout ${LORA_DROPOUT} \
    --lora-target-modules \"${LORA_TARGET_MODULES}\" \
    --mixed-precision \"${MIXED_PRECISION}\" \
    --grad-accum-steps ${GRAD_ACCUM_STEPS} \
    --max-grad-norm ${MAX_GRAD_NORM} \
    --scheduler-type \"${SCHEDULER_TYPE}\" \
    --seed ${SEED} \
    --early-stopping-patience ${EARLY_STOPPING} \
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
echo "Best model: ${OUTPUT_DIR}/best_model"
echo "========================================"
