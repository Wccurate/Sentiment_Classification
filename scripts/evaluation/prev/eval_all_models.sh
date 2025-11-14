#!/bin/bash
# ============================================================================
# Quick Evaluation Scripts for Different Model Types
# ============================================================================
# This script provides quick evaluation commands for all supported models.
# Edit the paths below and uncomment the model you want to evaluate.
# ============================================================================

set -e

# ========== COMMON SETTINGS ==========
DATASET_DIR="data/raw"
BATCH_SIZE=32
MAX_LENGTH=256
DEVICE="cuda"

# ========== DATASET SELECTION ==========
# Uncomment one of the following:
DATASET="atis"
# DATASET="hwu64"
# DATASET="snips"
# DATASET="clinc_oos"

# ========== 1. BERT HEAD ONLY (or Full Fine-tuning) ==========
echo "========================================"
echo "1. Evaluating BERT Head Only"
echo "========================================"

python eval_unified.py \
    --model-type bert-head \
    --dataset ${DATASET} \
    --dataset-dir ${DATASET_DIR} \
    --model-path outputs/checkpoints/bert_hf_head_${DATASET}/best_model \
    --output-dir results/bert_head_${DATASET}_eval \
    --batch-size ${BATCH_SIZE} \
    --max-length 128 \
    --device ${DEVICE} \
    --method "BERT-Head-Only"

echo ""
echo "✅ BERT Head Only evaluation completed"
echo ""

# ========== 2. BERT PROMPT TUNING ==========
# echo "========================================"
# echo "2. Evaluating BERT Prompt Tuning"
# echo "========================================"
# 
# python eval_unified.py \
#     --model-type bert-prompt \
#     --dataset ${DATASET} \
#     --dataset-dir ${DATASET_DIR} \
#     --base-model-path /projects/wangshibohdd/models/model_from_hf/bert-base-cased \
#     --adapter-path outputs/checkpoints/bert_prompt_tuning_${DATASET}/best_model \
#     --output-dir results/bert_prompt_${DATASET}_eval \
#     --batch-size ${BATCH_SIZE} \
#     --max-length 128 \
#     --device ${DEVICE} \
#     --merge-adapter \
#     --method "BERT-PromptTuning"
# 
# echo ""
# echo "✅ BERT Prompt Tuning evaluation completed"
# echo ""

# ========== 3. QWEN3 LORA ==========
# echo "========================================"
# echo "3. Evaluating Qwen3 LoRA"
# echo "========================================"
# 
# python eval_unified.py \
#     --model-type qwen3-lora \
#     --dataset ${DATASET} \
#     --dataset-dir ${DATASET_DIR} \
#     --base-model-path /projects/wangshibohdd/models/model_from_hf/qwen317b \
#     --adapter-path outputs/checkpoints/qwen3_${DATASET}_lora/best_model \
#     --output-dir results/qwen3_lora_${DATASET}_eval \
#     --batch-size ${BATCH_SIZE} \
#     --max-length ${MAX_LENGTH} \
#     --device ${DEVICE} \
#     --merge-adapter \
#     --use-bf16 \
#     --method "Qwen3-1.7B-LoRA"
# 
# echo ""
# echo "✅ Qwen3 LoRA evaluation completed"
# echo ""

# ========== COMPLETION ==========
echo "========================================"
echo "✅ All evaluations completed!"
echo "========================================"
