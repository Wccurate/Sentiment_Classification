# Sentiment Classification Project Overview

This document distills the current README, highlights representative code, and gives a top-level map of the repository so you can quickly understand how to train/evaluate each sentiment analysis method.

## Methods Implemented

1. TF-IDF + PCA + SVM
2. Word2Vec (average pooling) + SVM
3. Sentence-BERT (Bilingual-Embedding-Small) + SVM
4. BERT + MLP (frozen encoder)
5. BERT + MLP (full fine-tuning)
6. Prompt-Tuning + BERT + MLP
7. LoRA + BERT + MLP
8. LoRA + Qwen3 + MLP
9. LoRA + Qwen3 + MLP + Focal Loss

## Repository Structure (top levels)

```
Sentiment_Classification/
├── configs/                # YAML configs for unified training
├── data/                   # Dataset loaders + raw data placeholder
├── docs/                   # Project overview (this file)
├── evaluation/             # Metrics utilities + analyzers
├── scripts/
│   ├── train.py           # Unified training entry
│   ├── evaluate.py        # Unified evaluation entry (classical methods)
│   ├── download/          # Dataset/model download helpers
│   ├── evaluation/        # Legacy/advanced eval scripts
│   ├── legacy/            # Original train/eval scripts
│   └── training/          # Shell wrappers kept for reference
├── src/
│   ├── data/              # Dataset abstractions & tokenized dataloaders
│   ├── features/          # TF-IDF, Word2Vec, Sentence-BERT encoders
│   ├── models/            # SVM wrapper + utilities
│   ├── training/          # Classical + transformer training loops
│   ├── evaluation/        # Evaluation helpers
│   └── utils/             # Logging/config helpers
└── outputs/, results/, wandb/   # Default artifact locations
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate  # optional
pip install -r requirements.txt
```

If you need gated checkpoints (e.g., Qwen), configure the appropriate Hugging Face tokens.

### Base model downloads

Use the helper scripts to download datasets and encoder checkpoints into the expected folders:

```bash
python scripts/download/dataset_download.py --base-dir data/raw
python scripts/download/model_download.py --base-dir models/huggingface_models
python scripts/download/download_embedder.py --base-dir models/huggingface_models
```

## Training Workflows

### Config-driven (recommended)

Each YAML file under `configs/training/` captures the settings from the original shell scripts. Launch any experiment with a single command:

```bash
python scripts/train.py --config configs/training/bert_lora.yaml
```

Representative config (`configs/training/bert_lora.yaml`):

```yaml
method: bert_lora
train_csv: data/project/train.csv
text_col: text
label_col: label
output_dir: outputs/bert_lora
model_name: bert-base-cased
epochs: 20
batch_size: 128
max_len: 128
lr: 3e-4
lora_r: 8
lora_alpha: 16
lora_dropout: 0.1
lora_target_modules:
  - query
  - key
  - value
seed: 42
```

### Ad-hoc CLI example

```bash
python scripts/train.py --method tfidf_svm --train_csv data/train.csv --output_dir outputs
python scripts/train.py --method bert_finetune --train_csv data/train.csv \
    --model_name bert-base-cased --epochs 3 --batch_size 32 --lr 2e-5
```

### Legacy scripts

Original shell scripts remain under `scripts/training/`. Run them directly if you need the historical workflow:

```bash
bash scripts/training/bert_head_encoder_train.sh
bash scripts/training/qwen3_lora_train.sh
```

## Representative Code

The classical pipelines share the reusable trainer below (`src/training/trainer.py`). It illustrates how TF-IDF + SVM training is orchestrated:

```python
def run_tfidf_svm(cfg: ClassicalRunConfig, output_dir: str = "outputs/tfidf_svm"):
    logger = get_logger()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    data = load_csv_dataset(cfg.train_csv, cfg.text_col, cfg.label_col)
    train, val = split_dataset(data, cfg.val_ratio, cfg.seed)

    tfidf = TfidfFeaturizer(TfidfConfig(
        max_features=cfg.max_features,
        ngram_range=(cfg.ngram_min, cfg.ngram_max),
        pca_components=cfg.pca_components,
    ))
    X_train = tfidf.fit_transform(train)
    y_train = _labels_or_fail(train)
    X_val = tfidf.transform(val)
    y_val = _labels_or_fail(val)

    svm = SVMClassifier(SVMConfig(C=cfg.C))
    svm.fit(X_train, y_train)
    preds = svm.predict(X_val)
    acc = float((preds == y_val).mean())
    logger.info(f"Validation accuracy: {acc:.4f}")

    joblib.dump(tfidf, Path(output_dir) / "tfidf_featurizer.joblib")
    svm.save(Path(output_dir) / "svm_model.joblib")
```

Transformer-based methods share `run_transformer_training` in `src/training/bert_pipeline.py`, which configures frozen/LoRA/prompt modes, handles gradient accumulation, FP16, and optional focal loss.

## Evaluation & Analysis

1. Run classical evaluations:

   ```bash
   python scripts/evaluate.py \
       --method tfidf_svm \
       --checkpoint outputs/tfidf_svm \
       --data_csv data/val.csv \
       --output_dir outputs/eval
   ```

2. Convert `evaluation_results.json` into detailed metrics/reports:

   ```bash
   python -m evaluation.analyze_results outputs/eval/evaluation_results.json
   ```

## Pretrained Checkpoints

Download-and-use checkpoints:

| Method | Link |
| --- | --- |
| BERT full fine-tuning | [Drive](https://drive.google.com/file/d/1_NzccVd1DF5vPv3NJCkN1jWYcj5uF3az/view?usp=drive_link) |
| BERT LoRA adapter | [Drive](https://drive.google.com/file/d/18G7fm_-uRdlAy9QZLc-udd16vzzQC3dh/view?usp=drive_link) |
| BERT Prompt Tuning adapter | [Drive](https://drive.google.com/file/d/1uXRLKbD4iDH0XHX_DXy7WN4iQvfT9N6d/view?usp=drive_link) |
| Qwen3 LoRA + Focal adapter | [Drive](https://drive.google.com/file/d/13V_L3p_OJno5CccBJgc7hBoUkzrrwwO_/view?usp=drive_link) |
| Qwen3 LoRA adapter | [Drive](https://drive.google.com/file/d/1YZESoaXCW6dwFUHcI5FI-mPaBBhLX27Z/view?usp=drive_link) |

Place the extracted checkpoints under `outputs/<method>/checkpoint-best` (or update `--checkpoint` paths) and reuse the evaluation commands above.

## Legacy vs Refactored Version

- The refactored `src/` + `scripts/train.py`/`scripts/evaluate.py` stack is the primary interface and powers the README instructions.
- `scripts/legacy/` and `scripts/training/` contain the original project scripts; keep them for audits or regression testing, but all new experiments should go through the unified CLI/config approach.

This document should serve as a quick guide for both the code layout and the practical commands described in the README.
