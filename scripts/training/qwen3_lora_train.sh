#!/bin/bash
# Training script for Qwen3 1.7B LoRA fine-tuning
# Supports both standard CE loss and Focal Loss with class weighting

set -e

# ========== Mode Selection ==========
USE_FOCAL_LOSS=true  # Set to true to use Focal Loss with class weighting

# ========== Configuration ==========
WANDB_API_KEY="xxx"
WANDB_PROJECT="EE6483"
DATASET="project"  # Options: atis, hwu64, snips, clinc_oos, project
CLINC_VERSION="plus"  # For clinc_oos: small, plus, imbalanced
DATASET_DIR="data/raw/project"

MODEL_PATH="/usr1/home/s125mdg41_03/models/hf_models/qwen317b"

# Output directory depends on loss type
if [ "$USE_FOCAL_LOSS" = true ]; then
    OUTPUT_DIR="/usr1/home/s125mdg41_03/models/ee6483/qwen3_${DATASET}_lora_focal_3"
else
    OUTPUT_DIR="/usr1/home/s125mdg41_03/models/ee6483/qwen3_${DATASET}_lora"
fi

# Training hyperparameters
EPOCHS=20
BATCH_SIZE=32
MAX_LENGTH=128
LR=1e-4
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.06
VAL_RATIO=0.1

# LoRA configuration
LORA_R=4
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# Accelerate settings
MIXED_PRECISION="bf16"  # Options: no, fp16, bf16
GRAD_ACCUM_STEPS=1

SEED=42
SAVE_BEST_ONLY=""  # Set to "--save-best-only" to save disk space

# Focal Loss specific parameters (only used if USE_FOCAL_LOSS=true)
FOCAL_GAMMA=5.0  # Focusing parameter (0=CE, 2=standard focal, higher=more focus on hard)
CLASS_WEIGHT_METHOD="inverse_sqrt"  # Options: inverse, inverse_sqrt, effective_samples
NO_CLASS_WEIGHTS=""  # Set to "--no-class-weights" to disable class weighting

# ========== Select Training Script ==========
if [ "$USE_FOCAL_LOSS" = true ]; then
    TRAIN_SCRIPT="train_qwen3_lora_focal.py"
    LOSS_TYPE="Focal Loss (gamma=${FOCAL_GAMMA}, method=${CLASS_WEIGHT_METHOD})"
else
    TRAIN_SCRIPT="train_qwen3_lora.py"
    LOSS_TYPE="Cross-Entropy"
fi

# ========== Run Training ==========
echo "========================================"
echo " Qwen3 1.7B LoRA Fine-tuning"
echo "========================================"
echo "Loss Type: ${LOSS_TYPE}"
echo "Dataset: ${DATASET}"
echo "Output: ${OUTPUT_DIR}"
echo "Epochs: ${EPOCHS}, Batch Size: ${BATCH_SIZE}"
echo "LoRA r=${LORA_R}, alpha=${LORA_ALPHA}"
echo "========================================"

CUDA_VISIBLE_DEVICES=1 python ${TRAIN_SCRIPT} \
    --dataset "${DATASET}" \
    --clinc-version "${CLINC_VERSION}" \
    --dataset-dir "${DATASET_DIR}" \
    --model-path "${MODEL_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --max-length ${MAX_LENGTH} \
    --lr ${LR} \
    --weight-decay ${WEIGHT_DECAY} \
    --warmup-ratio ${WARMUP_RATIO} \
    --val-ratio ${VAL_RATIO} \
    --lora-r ${LORA_R} \
    --lora-alpha ${LORA_ALPHA} \
    --lora-dropout ${LORA_DROPOUT} \
    --lora-target-modules "${LORA_TARGET_MODULES}" \
    --mixed-precision "${MIXED_PRECISION}" \
    --grad-accum-steps ${GRAD_ACCUM_STEPS} \
    --seed ${SEED} \
    --use-wandb \
    --wandb-api-key ${WANDB_API_KEY} \
    --wandb-project ${WANDB_PROJECT} \
    --wandb-run-name "qwen3_${DATASET}_lora_experiment" \
    --wandb-tags "qwen3,lora,${DATASET}" \
    ${SAVE_BEST_ONLY} \
    $(if [ "$USE_FOCAL_LOSS" = true ]; then
        echo "--focal-gamma ${FOCAL_GAMMA}"
        echo "--class-weight-method ${CLASS_WEIGHT_METHOD}"
        echo "${NO_CLASS_WEIGHTS}"
    fi)

echo ""
echo "========================================"
echo "Training complete!"
echo "Loss type: ${LOSS_TYPE}"
echo "Checkpoints saved to: ${OUTPUT_DIR}"
echo "========================================"
