#!/bin/bash
# ============================================================================
# Batch measure GPU memory usage for all models
# ============================================================================

set -e

# ========== Config ==========

CUDA_VISIBLE_DEVICES="0"
DATASET="atis"
NUM_LABELS=21  # atis: 21, hwu64: 64, snips: 7, clinc_oos: 151
SEQ_LENGTH=128

# Model checkpoints
BERT_HEAD="/usr1/home/s125mdg41_03/models/intent_exp/bert_base_atis_hf_head_encoder/best_model"
BERT_LORA_BASE="/usr1/home/s125mdg41_03/models/hf_models/bert_base_cased_google_bert"
BERT_LORA_ADAPTER="/usr1/home/s125mdg41_03/models/intent_exp/bert_lora_clinc_oos/best_model"
BERT_PROMPT_BASE="/usr1/home/s125mdg41_03/models/hf_models/bert_base_cased_google_bert"
BERT_PROMPT_ADAPTER="/usr1/home/s125mdg41_03/models/intent_exp/bert_atis_hf_p_tuning/best_model"
QWEN3_BASE="/usr1/home/s125mdg41_03/models/hf_models/qwen317b"
QWEN3_ADAPTER="/usr1/home/s125mdg41_03/models/intent_exp/qwen3_atis_lora/best_model"

# ========== Helpers ==========

measure() {
    local name=$1
    shift
    echo ""
    echo "========================================"
    echo "Measuring: $name"
    echo "========================================"
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python calculate_memory.py \
        --num-labels ${NUM_LABELS} \
        --seq-length ${SEQ_LENGTH} \
        "$@" | tee -a memory_results.txt
}

# ========== Run ==========

echo "GPU memory measurement - all models"
echo "Dataset: ${DATASET}"
echo "Num labels: ${NUM_LABELS}"
echo "Seq length: ${SEQ_LENGTH}"
echo ""

# Reset results file
> memory_results.txt

# Measure BERT Head
if [ -d "${BERT_HEAD}" ]; then
    measure "BERT Head" \
        --model-type bert-head \
        --model-path "${BERT_HEAD}"
else
    echo "⚠️  Missing BERT Head model: ${BERT_HEAD}"
fi

# Measure BERT LoRA
if [ -d "${BERT_LORA_BASE}" ] && [ -d "${BERT_LORA_ADAPTER}" ]; then
    measure "BERT LoRA" \
        --model-type bert-lora \
        --base-model-path "${BERT_LORA_BASE}" \
        --adapter-path "${BERT_LORA_ADAPTER}"
else
    echo "⚠️  Missing BERT LoRA model"
fi

# Measure BERT Prompt Tuning
if [ -d "${BERT_PROMPT_BASE}" ] && [ -d "${BERT_PROMPT_ADAPTER}" ]; then
    measure "BERT Prompt Tuning" \
        --model-type bert-prompt \
        --base-model-path "${BERT_PROMPT_BASE}" \
        --adapter-path "${BERT_PROMPT_ADAPTER}"
else
    echo "⚠️  Missing BERT Prompt Tuning model"
fi

# Measure Qwen3 LoRA
if [ -d "${QWEN3_BASE}" ] && [ -d "${QWEN3_ADAPTER}" ]; then
    measure "Qwen3 LoRA" \
        --model-type qwen3-lora \
        --base-model-path "${QWEN3_BASE}" \
        --adapter-path "${QWEN3_ADAPTER}"
else
    echo "⚠️  Missing Qwen3 LoRA model"
fi

# ========== Summary ==========

echo ""
echo "========================================"
echo "Measurement complete. Results saved to memory_results.txt"
echo "========================================"
