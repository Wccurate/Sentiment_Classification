#!/bin/bash
# Batch evaluation script for Qwen3 LoRA on all datasets

set -e

BASE_MODEL_PATH="/projects/wangshibohdd/models/model_from_hf/qwen317b"
DATASET_DIR="data/raw"
BATCH_SIZE=32

DATASETS=("atis" "hwu64" "snips" "clinc_oos")

echo "========================================"
echo "📊 Batch Evaluation: Qwen3 LoRA"
echo "========================================"
echo "Datasets: ${DATASETS[*]}"
echo "========================================"

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "========================================"
    echo "▶️  Evaluating: ${DATASET}"
    echo "========================================"
    
    LORA_PATH="outputs/checkpoints/qwen3_${DATASET}_lora/best_model"
    OUTPUT_DIR="results/qwen3_${DATASET}_lora_eval"
    
    if [ ! -d "${LORA_PATH}" ]; then
        echo "⚠️  LoRA checkpoint not found: ${LORA_PATH}"
        echo "   Skipping ${DATASET}..."
        continue
    fi
    
    if [ "${DATASET}" == "clinc_oos" ]; then
        CLINC_VERSION="plus"
    else
        CLINC_VERSION="plus"
    fi
    
    python eval_qwen3_lora.py \
        --dataset "${DATASET}" \
        --clinc-version "${CLINC_VERSION}" \
        --dataset-dir "${DATASET_DIR}" \
        --base-model-path "${BASE_MODEL_PATH}" \
        --lora-path "${LORA_PATH}" \
        --output-dir "${OUTPUT_DIR}" \
        --batch-size ${BATCH_SIZE} \
        --seed 42
    
    echo "✅ ${DATASET} evaluation complete!"
done

echo ""
echo "========================================"
echo "✅ All evaluation jobs complete!"
echo "========================================"

# Summary
echo ""
echo "📊 Evaluation Results Summary:"
echo "========================================"
for DATASET in "${DATASETS[@]}"; do
    SUMMARY_FILE="results/qwen3_${DATASET}_lora_eval/summary.txt"
    if [ -f "${SUMMARY_FILE}" ]; then
        echo ""
        echo "▶️  ${DATASET}:"
        grep "Accuracy:" "${SUMMARY_FILE}" | head -n 1
    fi
done
echo "========================================"
