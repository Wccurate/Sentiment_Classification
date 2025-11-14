#!/bin/bash
# Evaluation script for Qwen3 1.7B LoRA fine-tuned model

set -e

# ========== Configuration ==========
DATASET="clinc_oos"  # Options: atis, hwu64, snips, clinc_oos
CLINC_VERSION="plus"
DATASET_DIR="data/raw/clinc_oos_deeppavlov"

BASE_MODEL_PATH="/usr1/home/s125mdg41_03/models/hf_models/qwen317b"
LORA_PATH="/usr1/home/s125mdg41_03/models/intent_exp/qwen3_clinc_oos_lora_first"
OUTPUT_DIR="results/qwen3_${DATASET}_lora_eval_first_3"

BATCH_SIZE=32
MAX_LENGTH=256
WARMUP_STEPS=10
METHOD="Qwen3-1.7B-LoRA"

SEED=42

# ========== Run Evaluation ==========
echo "========================================"
echo "📊 Qwen3 1.7B LoRA Evaluation"
echo "========================================"
echo "Dataset: ${DATASET}"
echo "LoRA Adapter: ${LORA_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "========================================"

CUDA_VISIBLE_DEVICES=3 python eval_qwen3_lora.py \
    --dataset "${DATASET}" \
    --clinc-version "${CLINC_VERSION}" \
    --dataset-dir "${DATASET_DIR}" \
    --base-model-path "${BASE_MODEL_PATH}" \
    --lora-path "${LORA_PATH}/best_model" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size ${BATCH_SIZE} \
    --max-length ${MAX_LENGTH} \
    --warmup-steps ${WARMUP_STEPS} \
    --method "${METHOD}" \
    --seed ${SEED}

echo ""
echo "✅ Evaluation complete!"
echo "📁 Results saved to: ${OUTPUT_DIR}"
