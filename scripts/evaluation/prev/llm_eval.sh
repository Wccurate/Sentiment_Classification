#!/bin/bash
# LLM intent classification evaluation script

# Variables
DATASET="atis"  # atis, hwu64, snips, clinc_oos
MODEL="gpt-4o-mini"  # e.g., gpt-4 or deepseek-chat
API_BASE="https://api.openai.com/v1"
API_KEY="your-api-key-here"  # Replace with real API key
MODE="zero-shot"  # zero-shot or few-shot
NUM_SHOTS=5
SHOT_SELECTION="balanced"  # random or balanced
TEMPERATURE=0.0
MAX_TOKENS=50
MAX_SAMPLES=""  # Leave empty for all samples or set e.g. 100

# Output dir
OUTPUT_DIR="results/llm_${DATASET}_${MODE}_${MODEL//\//_}_eval"

# Dataset dir
DATASET_DIR="data/raw"

# Build command
CMD="python eval_llm.py \
    --dataset ${DATASET} \
    --model ${MODEL} \
    --api-base ${API_BASE} \
    --api-key ${API_KEY} \
    --output-dir ${OUTPUT_DIR} \
    --mode ${MODE} \
    --temperature ${TEMPERATURE} \
    --max-tokens ${MAX_TOKENS} \
    --dataset-dir ${DATASET_DIR}"

# Few-shot extras
if [ "${MODE}" = "few-shot" ]; then
    CMD="${CMD} --num-shots ${NUM_SHOTS} --shot-selection ${SHOT_SELECTION}"
fi

# Optional max-samples argument
if [ -n "${MAX_SAMPLES}" ]; then
    CMD="${CMD} --max-samples ${MAX_SAMPLES}"
fi

# Display command
echo "================================"
echo "LLM Intent Recognition Evaluation"
echo "================================"
echo "Dataset: ${DATASET}"
echo "Model: ${MODEL}"
echo "Mode: ${MODE}"
echo "Output: ${OUTPUT_DIR}"
echo "================================"
echo ""
echo "Running command:"
echo "${CMD}"
echo ""

# Execute
eval ${CMD}

# Check status
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Evaluation completed successfully!"
    echo "📁 Results saved to: ${OUTPUT_DIR}"
else
    echo ""
    echo "❌ Evaluation failed!"
    exit 1
fi
