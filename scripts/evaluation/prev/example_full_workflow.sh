#!/bin/bash
# ============================================================================
# Example: Complete Training and Evaluation Workflow
# ============================================================================
# This script demonstrates a complete workflow from training to evaluation
# for all supported model types.
# ============================================================================

set -e

echo "============================================================"
echo "Complete Training and Evaluation Workflow"
echo "============================================================"
echo ""

# Configuration
DATASET="atis"
DATASET_DIR="data/raw"
DEVICE="cuda"

# ========== BERT Head Only ==========
echo "=========================================="
echo "1. BERT Head Only - Training & Evaluation"
echo "=========================================="
echo ""

# Train
echo "Step 1.1: Training BERT Head Only..."
python train_bert_hf_head.py \
    --dataset ${DATASET} \
    --dataset-dir ${DATASET_DIR} \
    --model-path /projects/wangshibohdd/models/model_from_hf/bert-base-cased \
    --output-dir outputs/checkpoints/bert_head_${DATASET}_demo \
    --epochs 3 \
    --batch-size 32 \
    --head-lr 3e-4 \
    --encoder-lr 3e-5 \
    --max-length 128 \
    --mixed-precision fp16

echo ""
echo "Step 1.2: Evaluating BERT Head Only..."
python eval_unified.py \
    --model-type bert-head \
    --dataset ${DATASET} \
    --dataset-dir ${DATASET_DIR} \
    --model-path outputs/checkpoints/bert_head_${DATASET}_demo/best_model \
    --output-dir results/bert_head_${DATASET}_demo_eval \
    --batch-size 32 \
    --max-length 128 \
    --device ${DEVICE}

echo ""
echo "Results saved to: results/bert_head_${DATASET}_demo_eval/"
cat results/bert_head_${DATASET}_demo_eval/summary.txt
echo ""

# ========== BERT Prompt Tuning ==========
echo "=========================================="
echo "2. BERT Prompt Tuning - Training & Evaluation"
echo "=========================================="
echo ""

# Train
echo "Step 2.1: Training BERT Prompt Tuning..."
python train_bert_prompt_tuning.py \
    --dataset ${DATASET} \
    --dataset-dir ${DATASET_DIR} \
    --model-path /projects/wangshibohdd/models/model_from_hf/bert-base-cased \
    --output-dir outputs/checkpoints/bert_prompt_${DATASET}_demo \
    --epochs 20 \
    --batch-size 32 \
    --num-virtual-tokens 20 \
    --prompt-lr 3e-2 \
    --classifier-lr 1e-3 \
    --max-length 128 \
    --scheduler-type cosine

echo ""
echo "Step 2.2: Evaluating BERT Prompt Tuning..."
python eval_unified.py \
    --model-type bert-prompt \
    --dataset ${DATASET} \
    --dataset-dir ${DATASET_DIR} \
    --base-model-path /projects/wangshibohdd/models/model_from_hf/bert-base-cased \
    --adapter-path outputs/checkpoints/bert_prompt_${DATASET}_demo/best_model \
    --output-dir results/bert_prompt_${DATASET}_demo_eval \
    --batch-size 32 \
    --max-length 128 \
    --merge-adapter \
    --device ${DEVICE}

echo ""
echo "Results saved to: results/bert_prompt_${DATASET}_demo_eval/"
cat results/bert_prompt_${DATASET}_demo_eval/summary.txt
echo ""

# ========== Qwen3 LoRA ==========
echo "=========================================="
echo "3. Qwen3 LoRA - Training & Evaluation"
echo "=========================================="
echo ""

# Train
echo "Step 3.1: Training Qwen3 LoRA..."
python train_qwen3_lora.py \
    --dataset ${DATASET} \
    --dataset-dir ${DATASET_DIR} \
    --model-path /projects/wangshibohdd/models/model_from_hf/qwen317b \
    --output-dir outputs/checkpoints/qwen3_${DATASET}_demo \
    --epochs 5 \
    --batch-size 16 \
    --lr 5e-4 \
    --lora-r 16 \
    --lora-alpha 32 \
    --max-length 256 \
    --mixed-precision bf16

echo ""
echo "Step 3.2: Evaluating Qwen3 LoRA..."
python eval_unified.py \
    --model-type qwen3-lora \
    --dataset ${DATASET} \
    --dataset-dir ${DATASET_DIR} \
    --base-model-path /projects/wangshibohdd/models/model_from_hf/qwen317b \
    --adapter-path outputs/checkpoints/qwen3_${DATASET}_demo/best_model \
    --output-dir results/qwen3_${DATASET}_demo_eval \
    --batch-size 32 \
    --max-length 256 \
    --merge-adapter \
    --use-bf16 \
    --device ${DEVICE}

echo ""
echo "Results saved to: results/qwen3_${DATASET}_demo_eval/"
cat results/qwen3_${DATASET}_demo_eval/summary.txt
echo ""

# ========== Summary ==========
echo "============================================================"
echo "✅ Complete workflow finished!"
echo "============================================================"
echo ""
echo "Summary of results:"
echo ""
echo "1. BERT Head Only:"
cat results/bert_head_${DATASET}_demo_eval/summary.txt | grep "Accuracy"
echo ""
echo "2. BERT Prompt Tuning:"
cat results/bert_prompt_${DATASET}_demo_eval/summary.txt | grep "Accuracy"
echo ""
echo "3. Qwen3 LoRA:"
cat results/qwen3_${DATASET}_demo_eval/summary.txt | grep "Accuracy"
echo ""
echo "============================================================"
echo "All results saved to: results/*_demo_eval/"
echo "============================================================"
