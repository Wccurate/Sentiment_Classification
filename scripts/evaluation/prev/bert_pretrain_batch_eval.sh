#!/bin/bash
# Batch evaluate BERT pretrained models on all datasets (zero-shot)

# Model config
ENCODER_PATH="/projects/wangshibohdd/models/model_from_hf/albert_base_v2_albert"

# Dataset list
DATASETS=("atis" "hwu64" "snips" "clinc_oos")

# Extra params
BATCH_SIZE=32
MAX_LENGTH=128
DEVICE="cuda"
SEED=42
DATASET_DIR="data/raw"
BASE_OUTPUT_DIR="results"

# Create results directory
mkdir -p ${BASE_OUTPUT_DIR}

# Log file
LOG_FILE="${BASE_OUTPUT_DIR}/bert_pretrain_batch_eval_$(date +%Y%m%d_%H%M%S).log"
echo "Batch BERT Pretrained Evaluation - $(date)" > ${LOG_FILE}
echo "======================================" >> ${LOG_FILE}

# Helper to run a single experiment
run_experiment() {
    local dataset=$1
    local output_dir="${BASE_OUTPUT_DIR}/bert_pretrain_${dataset}_eval"
    
    echo ""
    echo "======================================" | tee -a ${LOG_FILE}
    echo "Dataset: ${dataset}" | tee -a ${LOG_FILE}
    echo "Output: ${output_dir}" | tee -a ${LOG_FILE}
    echo "======================================" | tee -a ${LOG_FILE}
    
    # Build command
    local cmd="python eval_bert_pretrain.py \
        --dataset ${dataset} \
        --encoder-path ${ENCODER_PATH} \
        --output-dir ${output_dir} \
        --batch-size ${BATCH_SIZE} \
        --max-length ${MAX_LENGTH} \
        --device ${DEVICE} \
        --seed ${SEED} \
        --dataset-dir ${DATASET_DIR}"
    
    # Execute
    echo "Running: ${cmd}" >> ${LOG_FILE}
    eval ${cmd} 2>&1 | tee -a ${LOG_FILE}
    
    # Check result
    if [ $? -eq 0 ]; then
        echo "✅ Success: ${dataset}" | tee -a ${LOG_FILE}
    else
        echo "❌ Failed: ${dataset}" | tee -a ${LOG_FILE}
    fi
}

# Loop over datasets
echo "Starting batch evaluation..." | tee -a ${LOG_FILE}
echo "Datasets: ${DATASETS[@]}" | tee -a ${LOG_FILE}

total_experiments=${#DATASETS[@]}
current=0

for dataset in "${DATASETS[@]}"; do
    current=$((current + 1))
    echo ""
    echo "Progress: ${current}/${total_experiments}"
    run_experiment ${dataset}
done

# Build summary
echo ""
echo "======================================" | tee -a ${LOG_FILE}
echo "Batch Evaluation Complete!" | tee -a ${LOG_FILE}
echo "======================================" | tee -a ${LOG_FILE}
echo "Total experiments: ${total_experiments}" | tee -a ${LOG_FILE}
echo "Results directory: ${BASE_OUTPUT_DIR}" | tee -a ${LOG_FILE}
echo "Log file: ${LOG_FILE}" | tee -a ${LOG_FILE}

# Extract accuracy numbers into a table
echo ""
echo "Accuracy Summary:" | tee -a ${LOG_FILE}
echo "----------------" | tee -a ${LOG_FILE}

for dataset in "${DATASETS[@]}"; do
    result_dir="${BASE_OUTPUT_DIR}/bert_pretrain_${dataset}_eval"
    result_file="${result_dir}/evaluation_results.json"
    
    if [ -f "${result_file}" ]; then
        accuracy=$(python -c "import json; print(json.load(open('${result_file}'))['experiment_metadata']['accuracy'])" 2>/dev/null)
        if [ $? -eq 0 ]; then
            echo "${dataset}: ${accuracy}" | tee -a ${LOG_FILE}
        fi
    fi
done

echo ""
echo "✅ All evaluations completed!"
