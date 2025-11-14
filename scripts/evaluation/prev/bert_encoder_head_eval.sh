#!/bin/bash

# Evaluation script for BERT Encoder+Head models
# Corresponds to bert_encoder_head_train.sh training runs

# ATIS Evaluation
python eval_bert_detailed.py \
  --dataset atis \
  --model-path /projects/wangshibohdd/nlp_proj/bert_cased_atis_encoder_head/best_head \
  --encoder-path /projects/wangshibohdd/nlp_proj/bert_cased_atis_encoder_head/best_head \
  --tokenizer-path /projects/wangshibo/code/Intent_Recognition_Exp/models/huggingface_models/bert_base_cased_google_bert \
  --output-dir /projects/wangshibohdd/nlp_proj/results/bert_cased_atis_encoder_head_eval \
  --method "BERT-Base-Cased" \
  --method-detail "Encoder + Head trained with layered LR (enc=2e-5, head=1e-4)" \
  --dataset-dir /projects/wangshibo/code/Intent_Recognition_Exp/data/raw/atis_intents_fathyshalab \
  --batch-size 64 \
  --max-length 128 \
  --device cuda \
  --warmup-steps 10

echo "✅ ATIS evaluation completed"
echo ""


# SNIPS Evaluation
python eval_bert_detailed.py \
  --dataset snips \
  --model-path /projects/wangshibohdd/nlp_proj/bert_cased_snips_encoder_head/best_head \
  --encoder-path /projects/wangshibohdd/nlp_proj/bert_cased_snips_encoder_head/best_head \
  --tokenizer-path /projects/wangshibo/code/Intent_Recognition_Exp/models/huggingface_models/bert_base_cased_google_bert \
  --output-dir /projects/wangshibohdd/nlp_proj/results/bert_cased_snips_encoder_head_eval \
  --method "BERT-Base-Cased" \
  --method-detail "Encoder + Head trained with layered LR (enc=2e-5, head=1e-4)" \
  --dataset-dir /projects/wangshibo/code/Intent_Recognition_Exp/data/raw/snips_benayas \
  --batch-size 64 \
  --max-length 128 \
  --device cuda \
  --warmup-steps 10

echo "✅ SNIPS evaluation completed"
echo ""


# CLINC OOS Evaluation
python eval_bert_detailed.py \
  --dataset clinc_oos \
  --clinc-version plus \
  --model-path /projects/wangshibohdd/nlp_proj/bert_cased_clinc_encoder_head/best_head \
  --encoder-path /projects/wangshibohdd/nlp_proj/bert_cased_clinc_encoder_head/best_head \
  --tokenizer-path /projects/wangshibo/code/Intent_Recognition_Exp/models/huggingface_models/bert_base_cased_google_bert \
  --output-dir /projects/wangshibohdd/nlp_proj/results/bert_cased_clinc_encoder_head_eval \
  --method "BERT-Base-Cased" \
  --method-detail "Encoder + Head trained with layered LR (enc=2e-5, head=1e-4)" \
  --dataset-dir /projects/wangshibo/code/Intent_Recognition_Exp/data/raw/clinc_oos_deeppavlov \
  --batch-size 64 \
  --max-length 128 \
  --device cuda \
  --warmup-steps 10

echo "✅ CLINC OOS evaluation completed"
echo ""


# HWU64 Evaluation
python eval_bert_detailed.py \
  --dataset hwu64 \
  --model-path /projects/wangshibohdd/nlp_proj/bert_cased_hwu64_encoder_head/best_head \
  --encoder-path /projects/wangshibohdd/nlp_proj/bert_cased_hwu64_encoder_head/best_head \
  --tokenizer-path /projects/wangshibo/code/Intent_Recognition_Exp/models/huggingface_models/bert_base_cased_google_bert \
  --output-dir /projects/wangshibohdd/nlp_proj/results/bert_cased_hwu64_encoder_head_eval \
  --method "BERT-Base-Cased" \
  --method-detail "Encoder + Head trained with layered LR (enc=2e-5, head=1e-4)" \
  --dataset-dir /projects/wangshibo/code/Intent_Recognition_Exp/data/raw/hwu64_deeppavlov \
  --batch-size 64 \
  --max-length 128 \
  --device cuda \
  --warmup-steps 10

echo "✅ HWU64 evaluation completed"
echo ""

echo "🎉 All evaluations completed!"
echo "📁 Results saved in: /projects/wangshibohdd/nlp_proj/results/"

## Improved

python eval_bert_detailed.py \
  --dataset clinc_oos \
  --clinc-version plus \
  --model-path /projects/wangshibohdd/nlp_proj/bert_cased_clinc_improved/best_head \
  --encoder-path /projects/wangshibohdd/nlp_proj/bert_cased_clinc_improved/best_head \
  --tokenizer-path /projects/wangshibohdd/models/model_from_hf/bert_base_cased_google_bert \
  --output-dir /projects/wangshibo/code/Intent_Recognition_Exp/results/bert_cased_clinc_improved_eval \
  --method "BERT-Base-Cased" \
  --method-detail "Encoder + Head trained with layered LR (enc=2e-5, head=1e-4)" \
  --dataset-dir /projects/wangshibo/code/Intent_Recognition_Exp/data/raw/clinc_oos_deeppavlov \
  --batch-size 64 \
  --max-length 128 \
  --device cuda \
  --warmup-steps 10