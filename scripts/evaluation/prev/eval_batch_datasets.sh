#!/bin/bash
# ============================================================================
# Batch Evaluation Script - Evaluate All Datasets
# ============================================================================
# This script evaluates a specific model type on all datasets.
# ============================================================================

set -e

# ========== CONFIGURATION ==========

# Model type to evaluate: bert-head | bert-prompt | qwen3-lora
MODEL_TYPE="bert-head"

# Datasets to evaluate
DATASETS=("atis" "hwu64" "snips" "clinc_oos")

# Common settings
DATASET_DIR="data/raw"
BATCH_SIZE=32
DEVICE="cuda"

# ========== MODEL PATHS ==========
# Edit these based on your trained model locations

# For bert-head
BERT_HEAD_BASE_PATH="outputs/checkpoints"

# For bert-prompt
BERT_BASE_MODEL="/projects/wangshibohdd/models/model_from_hf/bert-base-cased"
PROMPT_BASE_PATH="outputs/checkpoints"

# For qwen3-lora
QWEN3_BASE_MODEL="/projects/wangshibohdd/models/model_from_hf/qwen317b"
LORA_BASE_PATH="outputs/checkpoints"

# ========== EVALUATION LOOP ==========

echo "============================================================"
echo "Batch Evaluation - Model Type: ${MODEL_TYPE}"
echo "============================================================"
echo "Datasets: ${DATASETS[@]}"
echo "Device: ${DEVICE}"
echo "Batch Size: ${BATCH_SIZE}"
echo "============================================================"
echo ""

for DATASET in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "Evaluating on dataset: ${DATASET}"
    echo "=========================================="
    
    # Set max_length based on dataset
    if [[ "${DATASET}" == "atis" || "${DATASET}" == "hwu64" ]]; then
        MAX_LENGTH=128
    else
        MAX_LENGTH=256
    fi
    
    # Build command based on model type
    if [[ "${MODEL_TYPE}" == "bert-head" ]]; then
        MODEL_PATH="${BERT_HEAD_BASE_PATH}/bert_hf_head_${DATASET}/best_model"
        OUTPUT_DIR="results/bert_head_${DATASET}_eval"
        
        # Check if model exists
        if [[ ! -d "${MODEL_PATH}" ]]; then
            echo "⚠️  Model not found: ${MODEL_PATH}"
            echo "Skipping ${DATASET}..."
            echo ""
            continue
        fi
        
        python eval_unified.py \
            --model-type bert-head \
            --dataset ${DATASET} \
            --dataset-dir ${DATASET_DIR} \
            --model-path ${MODEL_PATH} \
            --output-dir ${OUTPUT_DIR} \
            --batch-size ${BATCH_SIZE} \
            --max-length ${MAX_LENGTH} \
            --device ${DEVICE}
            
    elif [[ "${MODEL_TYPE}" == "bert-prompt" ]]; then
        ADAPTER_PATH="${PROMPT_BASE_PATH}/bert_prompt_tuning_${DATASET}/best_model"
        OUTPUT_DIR="results/bert_prompt_${DATASET}_eval"
        
        if [[ ! -d "${ADAPTER_PATH}" ]]; then
            echo "⚠️  Adapter not found: ${ADAPTER_PATH}"
            echo "Skipping ${DATASET}..."
            echo ""
            continue
        fi
        
        python eval_unified.py \
            --model-type bert-prompt \
            --dataset ${DATASET} \
            --dataset-dir ${DATASET_DIR} \
            --base-model-path ${BERT_BASE_MODEL} \
            --adapter-path ${ADAPTER_PATH} \
            --output-dir ${OUTPUT_DIR} \
            --batch-size ${BATCH_SIZE} \
            --max-length ${MAX_LENGTH} \
            --device ${DEVICE} \
            --merge-adapter
            
    elif [[ "${MODEL_TYPE}" == "qwen3-lora" ]]; then
        ADAPTER_PATH="${LORA_BASE_PATH}/qwen3_${DATASET}_lora/best_model"
        OUTPUT_DIR="results/qwen3_lora_${DATASET}_eval"
        
        if [[ ! -d "${ADAPTER_PATH}" ]]; then
            echo "⚠️  Adapter not found: ${ADAPTER_PATH}"
            echo "Skipping ${DATASET}..."
            echo ""
            continue
        fi
        
        python eval_unified.py \
            --model-type qwen3-lora \
            --dataset ${DATASET} \
            --dataset-dir ${DATASET_DIR} \
            --base-model-path ${QWEN3_BASE_MODEL} \
            --adapter-path ${ADAPTER_PATH} \
            --output-dir ${OUTPUT_DIR} \
            --batch-size ${BATCH_SIZE} \
            --max-length ${MAX_LENGTH} \
            --device ${DEVICE} \
            --merge-adapter \
            --use-bf16
    fi
    
    echo ""
    echo "✅ ${DATASET} evaluation completed"
    echo ""
    echo "------------------------------------------------------------"
    echo ""
done

# ========== SUMMARY ==========
echo "============================================================"
echo "✅ All dataset evaluations completed!"
echo "Model Type: ${MODEL_TYPE}"
echo "Datasets: ${DATASETS[@]}"
echo "============================================================"
echo ""
echo "Results saved to: results/${MODEL_TYPE}_*/evaluation_results.json"
echo ""
