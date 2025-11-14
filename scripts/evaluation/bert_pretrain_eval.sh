#!/bin/bash
# BERT pretrained encoder + cosine similarity zero-shot evaluation

# Dataset configuration
DATASET="clinc_oos"  # atis, hwu64, snips, clinc_oos
CLINC_VERSION="plus"  # small, plus, imbalanced (for clinc_oos only)

# GPU settings (optional)
CUDA_VISIBLE_DEVICES="3"  # Example: "0" or "0,1,2,3"

# Model configuration
ENCODER_PATH="/usr1/home/s125mdg41_03/models/hf_models/bert_base_cased_google_bert"
TOKENIZER_PATH=""  # Fallback to ENCODER_PATH when empty

# Output directory
OUTPUT_DIR="results/intent_new/bert_pretrain_${DATASET}_eval"

# Other args
BATCH_SIZE=32
MAX_LENGTH=128
DEVICE="cuda"  # cuda or cpu
SEED=42

# Dataset directory
DATASET_DIR="data/raw/${DATASET}"

# Build command
CMD="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python eval_bert_pretrain.py \
    --dataset ${DATASET} \
    --encoder-path ${ENCODER_PATH} \
    --output-dir ${OUTPUT_DIR} \
    --batch-size ${BATCH_SIZE} \
    --max-length ${MAX_LENGTH} \
    --device ${DEVICE} \
    --seed ${SEED} \
    --dataset-dir ${DATASET_DIR}"

# Append tokenizer path when provided
if [ -n "${TOKENIZER_PATH}" ]; then
    CMD="${CMD} --tokenizer-path ${TOKENIZER_PATH}"
fi

# Append clinc version for clinc_oos dataset
if [ "${DATASET}" = "clinc_oos" ]; then
    CMD="${CMD} --clinc-version ${CLINC_VERSION}"
fi

# Logs
echo "========================================"
echo "BERT Pretrained + Cosine Similarity Eval"
echo "========================================"
echo "Dataset: ${DATASET}"
echo "Encoder: ${ENCODER_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Device: ${DEVICE}"
echo "========================================"
echo ""
echo "Running command:"
echo "${CMD}"
echo ""

# Execute
eval ${CMD}

# Result check
if [ $? -eq 0 ]; then
    echo ""
    echo "Evaluation completed successfully!"
    echo "Results saved to: ${OUTPUT_DIR}"
else
    echo ""
    echo "Evaluation failed!"
    exit 1
fi
