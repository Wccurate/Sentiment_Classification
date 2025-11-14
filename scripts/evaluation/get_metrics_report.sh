#!/usr/bin/env bash


METHOD="bert"
DATASET="clinc_plus"
RESULT_DIR=/usr1/home/s125mdg41_03/code/Intent_Recognition_Exp/results/textcnn/${DATASET}/results.json
OUTPUT_BASE_DIR=/usr1/home/s125mdg41_03/code/Intent_Recognition_Exp/results/textcnn/${DATASET}/reports
TOP_K=5


python -m evaluation.analyze_results \
    ${RESULT_DIR} \
    --output ${OUTPUT_BASE_DIR}/report.json \
    --output-txt ${OUTPUT_BASE_DIR}/metrics_report.txt \
    --top-k ${TOP_K}

