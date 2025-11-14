#!/bin/bash
# ============================================================================
# Unified Evaluation Script for All Models
# ============================================================================
# This script evaluates different model types on various datasets.
# Supports: BERT Head Only, BERT Full Fine-tuning, BERT LoRA, BERT Prompt Tuning, Qwen3 LoRA
#
# Usage:
#   1. Set configuration variables below
#   2. Run: bash scripts/evaluation/eval_unified.sh
# ============================================================================

set -e  # Exit on error

# ========== CONFIGURATION ==========

CUDA_VISIBLE_DEVICES="3"

# Model type: bert-head | bert-lora | bert-prompt | qwen3-lora
MODEL_TYPE="qwen3-lora"
BERT_VERSION="bert"
FT_METHOD="lora"
# Dataset: atis | hwu64 | snips | clinc_oos | project
DATASET="project"

# For CLINC OOS only: small | plus | imbalanced
CLINC_VERSION="plus"

# Dataset directory
DATASET_DIR="data/raw/${DATASET}"

# ========== MODEL PATHS ==========
# Configure based on MODEL_TYPE

# For bert-head (Head Only or Full Fine-tuning)
MODEL_PATH="/usr1/home/s125mdg41_03/models/xxx/${BERT_VERSION}_${DATASET}_hf_head_encoder/best_model"

# For bert-lora (BERT LoRA)
BERT_BASE_MODEL_LORA="/usr1/home/s125mdg41_03/models/hf_models/bert_base_cased_google_bert"
BERT_LORA_ADAPTER_PATH="/usr1/home/s125mdg41_03/models/xxx/bert_lora_${DATASET}/best_model"

# For bert-prompt (Prompt Tuning)
BERT_BASE_MODEL_PROMPT="/usr1/home/s125mdg41_03/models/hf_models/bert_base_cased_google_bert"
BERT_PROMPT_ADAPTER_PATH="/usr1/home/s125mdg41_03/models/xxx/bert_${DATASET}_hf_p_tuning/best_model"

# For qwen3-lora (Qwen3 LoRA)
QWEN3_BASE_MODEL="/usr1/home/s125mdg41_03/models/hf_models/qwen317b"
QWEN3_LORA_ADAPTER_PATH="/usr1/home/s125mdg41_03/models/xxx/qwen3_${DATASET}_lora_focal_3/best_model"

# ========== EVALUATION SETTINGS ==========

# Output directory (will be created if not exists)
OUTPUT_DIR="results/xxx/eval_qwen3_${DATASET}_lora_eval_final"

# Batch size for batched inference
BATCH_SIZE=32

# Maximum sequence length
MAX_LENGTH=128

# Device: cuda | cpu
DEVICE="cuda"

# Random seed
SEED=42

# Warmup steps (for stable timing)
WARMUP_STEPS=10

# Method name (optional, auto-generated if not set)
METHOD=""

# Merge PEFT adapter for faster inference (for PEFT models)
MERGE_ADAPTER=true

# Use bfloat16 for Qwen3 (recommended if supported)
USE_BF16=false

# ========== DISPLAY CONFIGURATION ==========

echo "============================================================"
echo "Unified Model Evaluation"
echo "============================================================"
echo "Model Type:       ${MODEL_TYPE}"
echo "Dataset:          ${DATASET}"
if [[ "${DATASET}" == "clinc_oos" ]]; then
    echo "CLINC Version:    ${CLINC_VERSION}"
fi
echo "Device:           ${DEVICE}"
echo "Batch Size:       ${BATCH_SIZE}"
echo "Max Length:       ${MAX_LENGTH}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "------------------------------------------------------------"

# ========== BUILD COMMAND ==========

CMD="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python eval_unified.py"
CMD="${CMD} --model-type ${MODEL_TYPE}"
CMD="${CMD} --dataset ${DATASET}"
CMD="${CMD} --dataset-dir ${DATASET_DIR}"
CMD="${CMD} --output-dir ${OUTPUT_DIR}"
CMD="${CMD} --batch-size ${BATCH_SIZE}"
CMD="${CMD} --max-length ${MAX_LENGTH}"
CMD="${CMD} --device ${DEVICE}"
CMD="${CMD} --seed ${SEED}"
CMD="${CMD} --warmup-steps ${WARMUP_STEPS}"

if [[ "${DATASET}" == "clinc_oos" ]]; then
    CMD="${CMD} --clinc-version ${CLINC_VERSION}"
fi

if [[ -n "${METHOD}" ]]; then
    CMD="${CMD} --method ${METHOD}"
fi

# Add model-specific arguments
if [[ "${MODEL_TYPE}" == "bert-head" ]]; then
    echo "Model Path:       ${MODEL_PATH}"
    CMD="${CMD} --model-path ${MODEL_PATH}"
    
elif [[ "${MODEL_TYPE}" == "bert-lora" ]]; then
    echo "Base Model:       ${BERT_BASE_MODEL_LORA}"
    echo "LoRA Adapter:     ${BERT_LORA_ADAPTER_PATH}"
    CMD="${CMD} --base-model-path ${BERT_BASE_MODEL_LORA}"
    CMD="${CMD} --adapter-path ${BERT_LORA_ADAPTER_PATH}"
    if [[ "${MERGE_ADAPTER}" == true ]]; then
        CMD="${CMD} --merge-adapter"
    fi
    
elif [[ "${MODEL_TYPE}" == "bert-prompt" ]]; then
    echo "Base Model:       ${BERT_BASE_MODEL_PROMPT}"
    echo "Prompt Adapter:   ${BERT_PROMPT_ADAPTER_PATH}"
    CMD="${CMD} --base-model-path ${BERT_BASE_MODEL_PROMPT}"
    CMD="${CMD} --adapter-path ${BERT_PROMPT_ADAPTER_PATH}"
    if [[ "${MERGE_ADAPTER}" == true ]]; then
        CMD="${CMD} --merge-adapter"
    fi
    
elif [[ "${MODEL_TYPE}" == "qwen3-lora" ]]; then
    echo "Base Model:       ${QWEN3_BASE_MODEL}"
    echo "LoRA Adapter:     ${QWEN3_LORA_ADAPTER_PATH}"
    CMD="${CMD} --base-model-path ${QWEN3_BASE_MODEL}"
    CMD="${CMD} --adapter-path ${QWEN3_LORA_ADAPTER_PATH}"
    if [[ "${MERGE_ADAPTER}" == true ]]; then
        CMD="${CMD} --merge-adapter"
    fi
    if [[ "${USE_BF16}" == true ]]; then
        CMD="${CMD} --use-bf16"
    fi
fi

echo "============================================================"
echo ""

# ========== RUN EVALUATION ==========

echo "🚀 Starting evaluation..."
echo ""
echo "Command:"
echo "${CMD}"
echo ""

eval ${CMD}

# ========== COMPLETION MESSAGE ==========

echo ""
echo "============================================================"
echo "✅ Evaluation completed successfully!"
echo "📁 Results saved to: ${OUTPUT_DIR}"
echo "============================================================"
