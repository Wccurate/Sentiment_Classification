#!/bin/bash
# Batch LLM evaluation script covering multiple datasets and modes

# API config
API_BASE="https://api.openai.com/v1"
API_KEY="your-api-key-here"  # Replace with your key

# Model config
MODELS=("gpt-4o-mini" "gpt-4o")
DATASETS=("atis" "hwu64" "snips" "clinc_oos")
MODES=("zero-shot" "few-shot")

# Few-shot config
NUM_SHOTS=10
SHOT_SELECTION="balanced"

# Misc parameters
TEMPERATURE=0.0
MAX_TOKENS=50
DATASET_DIR="data/raw"
BASE_OUTPUT_DIR="results"

# Create results directory
mkdir -p ${BASE_OUTPUT_DIR}

# Log file
LOG_FILE="${BASE_OUTPUT_DIR}/batch_eval_$(date +%Y%m%d_%H%M%S).log"
echo "Batch LLM Evaluation - $(date)" > ${LOG_FILE}
echo "======================================" >> ${LOG_FILE}

# Run a single experiment
run_experiment() {
    local dataset=$1
    local model=$2
    local mode=$3
    
    # Sanitize model name for paths
    local model_clean=$(echo ${model} | sed 's/\//_/g')
    
    # Output directory
    local output_dir="${BASE_OUTPUT_DIR}/llm_${dataset}_${mode}_${model_clean}"
    
    echo ""
    echo "======================================" | tee -a ${LOG_FILE}
    echo "Dataset: ${dataset}" | tee -a ${LOG_FILE}
    echo "Model: ${model}" | tee -a ${LOG_FILE}
    echo "Mode: ${mode}" | tee -a ${LOG_FILE}
    echo "Output: ${output_dir}" | tee -a ${LOG_FILE}
    echo "======================================" | tee -a ${LOG_FILE}
    
    # Build command
    local cmd="python eval_llm.py \
        --dataset ${dataset} \
        --model ${model} \
        --api-base ${API_BASE} \
        --api-key ${API_KEY} \
        --output-dir ${output_dir} \
        --mode ${mode} \
        --temperature ${TEMPERATURE} \
        --max-tokens ${MAX_TOKENS} \
        --dataset-dir ${DATASET_DIR}"
    
    # Attach few-shot args when needed
    if [ "${mode}" = "few-shot" ]; then
        cmd="${cmd} --num-shots ${NUM_SHOTS} --shot-selection ${SHOT_SELECTION}"
    fi
    
    # Execute
    echo "Running: ${cmd}" >> ${LOG_FILE}
    eval ${cmd} 2>&1 | tee -a ${LOG_FILE}
    
    # Check status
    if [ $? -eq 0 ]; then
        echo "✅ Success: ${dataset} / ${model} / ${mode}" | tee -a ${LOG_FILE}
    else
        echo "❌ Failed: ${dataset} / ${model} / ${mode}" | tee -a ${LOG_FILE}
    fi
}

# Enumerate experiment grid
echo "Starting batch evaluation..." | tee -a ${LOG_FILE}
echo "Models: ${MODELS[@]}" | tee -a ${LOG_FILE}
echo "Datasets: ${DATASETS[@]}" | tee -a ${LOG_FILE}
echo "Modes: ${MODES[@]}" | tee -a ${LOG_FILE}

total_experiments=$((${#MODELS[@]} * ${#DATASETS[@]} * ${#MODES[@]}))
current=0

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for mode in "${MODES[@]}"; do
            current=$((current + 1))
            echo ""
            echo "Progress: ${current}/${total_experiments}"
            run_experiment ${dataset} ${model} ${mode}
            
            # Delay to avoid rate limits
            if [ ${current} -lt ${total_experiments} ]; then
                echo "Waiting 5 seconds before next experiment..."
                sleep 5
            fi
        done
    done
done

# Create summary report
echo ""
echo "======================================" | tee -a ${LOG_FILE}
echo "Batch Evaluation Complete!" | tee -a ${LOG_FILE}
echo "======================================" | tee -a ${LOG_FILE}
echo "Total experiments: ${total_experiments}" | tee -a ${LOG_FILE}
echo "Results directory: ${BASE_OUTPUT_DIR}" | tee -a ${LOG_FILE}
echo "Log file: ${LOG_FILE}" | tee -a ${LOG_FILE}

# Aggregate accuracy into a table
echo ""
echo "Accuracy Summary:" | tee -a ${LOG_FILE}
echo "----------------" | tee -a ${LOG_FILE}

for model in "${MODELS[@]}"; do
    model_clean=$(echo ${model} | sed 's/\//_/g')
    for dataset in "${DATASETS[@]}"; do
        for mode in "${MODES[@]}"; do
            result_dir="${BASE_OUTPUT_DIR}/llm_${dataset}_${mode}_${model_clean}"
            result_file="${result_dir}/evaluation_results.json"
            
            if [ -f "${result_file}" ]; then
                accuracy=$(python -c "import json; print(json.load(open('${result_file}'))['experiment_metadata']['accuracy'])" 2>/dev/null)
                if [ $? -eq 0 ]; then
                    echo "${dataset} | ${model} | ${mode}: ${accuracy}" | tee -a ${LOG_FILE}
                fi
            fi
        done
    done
done

echo ""
echo "✅ All evaluations completed!"
