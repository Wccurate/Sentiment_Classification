python train_bert_encoder_head.py \
  --dataset atis \
  --model-path /projects/wangshibo/code/Intent_Recognition_Exp/models/huggingface_models/bert_base_cased_google_bert \
  --tokenizer-path /projects/wangshibo/code/Intent_Recognition_Exp/models/huggingface_models/bert_base_cased_google_bert \
  --output-dir /projects/wangshibohdd/nlp_proj/bert_cased_atis_encoder_head \
  --epochs 100 --batch-size 64 --max-length 128 \
  --dataset-dir /projects/wangshibo/code/Intent_Recognition_Exp/data/raw/atis_intents_fathyshalab \
  --mixed-precision fp16 \
  --encoder-lr 2e-5 \
  --head-lr 1e-4


python train_bert_encoder_head.py \
  --dataset snips \
  --model-path /projects/wangshibo/code/Intent_Recognition_Exp/models/huggingface_models/bert_base_cased_google_bert \
  --tokenizer-path /projects/wangshibo/code/Intent_Recognition_Exp/models/huggingface_models/bert_base_cased_google_bert \
  --output-dir /projects/wangshibohdd/nlp_proj/bert_cased_snips_encoder_head \
  --epochs 100 --batch-size 64 --max-length 128 \
  --dataset-dir /projects/wangshibo/code/Intent_Recognition_Exp/data/raw/snips_benayas \
  --mixed-precision fp16 \
  --encoder-lr 2e-5 \
  --head-lr 1e-4




python train_bert_encoder_head.py \
  --dataset clinc_oos \
  --model-path /projects/wangshibo/code/Intent_Recognition_Exp/models/huggingface_models/bert_base_cased_google_bert \
  --tokenizer-path /projects/wangshibo/code/Intent_Recognition_Exp/models/huggingface_models/bert_base_cased_google_bert \
  --output-dir /projects/wangshibohdd/nlp_proj/bert_cased_clinc_encoder_head \
  --epochs 100 --batch-size 64 --max-length 128 \
  --dataset-dir /projects/wangshibo/code/Intent_Recognition_Exp/data/raw/clinc_oos_deeppavlov \
  --mixed-precision fp16 \
  --encoder-lr 2e-5 \
  --head-lr 1e-4



python train_bert_encoder_head.py \
  --dataset hwu64 \
  --model-path /projects/wangshibo/code/Intent_Recognition_Exp/models/huggingface_models/bert_base_cased_google_bert \
  --tokenizer-path /projects/wangshibo/code/Intent_Recognition_Exp/models/huggingface_models/bert_base_cased_google_bert \
  --output-dir /projects/wangshibohdd/nlp_proj/bert_cased_hwu64_encoder_head \
  --epochs 100 --batch-size 64 --max-length 128 \
  --dataset-dir /projects/wangshibo/code/Intent_Recognition_Exp/data/raw/hwu64_deeppavlov \
  --mixed-precision fp16 \
  --encoder-lr 2e-5 \
  --head-lr 1e-4



python train_bert_improved.py   --dataset clinc_oos   --model-path /projects/wangshibohdd/models/model_from_hf/bert_base_cased_google_bert   --tokenizer-path /projects/wangshibohdd/models/model_from_hf/bert_base_cased_google_bert   --output-dir /projects/wangshibohdd/nlp_proj/bert_cased_clinc_improved   --epochs 100 --batch-size 128 --m
ax-length 32   --dataset-dir /projects/wangshibo/code/Intent_Recognition_Exp/data/raw/clinc_oos_deeppavlov   --mixed-precision no   --encoder-lr 2e-5   --head-lr 1e-4









python train_bert_improved.py \
  --dataset hwu64 \
  --model-path /projects/wangshibohdd/models/model_from_hf/bert_base_cased_google_bert \
  --tokenizer-path /projects/wangshibohdd/models/model_from_hf/bert_base_cased_google_bert \
  --output-dir /projects/wangshibohdd/nlp_proj/bert_cased_hwu64_improved \
  --epochs 100 --batch-size 32 --max-length 256 \
  --dataset-dir /projects/wangshibo/code/Intent_Recognition_Exp/data/raw/hwu64_deeppavlov \
  --mixed-precision no \
  --encoder-lr 2e-5 \
  --head-lr 1e-4