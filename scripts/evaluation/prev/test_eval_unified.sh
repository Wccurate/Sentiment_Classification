#!/bin/bash
# ============================================================================
# Quick Test Script for eval_unified.py
# ============================================================================
# This script performs a quick test to verify eval_unified.py works correctly.
# ============================================================================

set -e

echo "============================================================"
echo "Testing eval_unified.py"
echo "============================================================"
echo ""

# Test 1: Check if script exists and has correct syntax
echo "Test 1: Checking script syntax..."
python -m py_compile eval_unified.py
echo "✅ Syntax check passed"
echo ""

# Test 2: Display help message
echo "Test 2: Displaying help message..."
python eval_unified.py --help | head -20
echo "✅ Help message displayed"
echo ""

# Test 3: Check required imports
echo "Test 3: Checking required imports..."
python -c "
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
print('✅ All required imports available')
"
echo ""

echo "============================================================"
echo "✅ All tests passed!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. Train a model using train_bert_hf_head.py or other training scripts"
echo "2. Run evaluation:"
echo "   python eval_unified.py --model-type bert-head --dataset atis \\"
echo "       --model-path outputs/checkpoints/bert_hf_head_atis/best_model \\"
echo "       --output-dir results/test_eval"
echo ""
