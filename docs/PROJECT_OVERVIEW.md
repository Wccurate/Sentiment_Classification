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

### Classical (TF-IDF + SVM)

The reusable trainer in `src/training/trainer.py`:

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

### Deep Learning (BERT/Qwen Loops)

`src/training/bert_pipeline.py` powers frozen, prompt, LoRA, and focal variants:

```python
def run_transformer_training(cfg: TransformerRunConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    if cfg.method == "bert_frozen":
        for param in model.base_model.parameters():
            param.requires_grad = False
    elif cfg.method in {"bert_lora", "qwen3_lora", "qwen3_lora_focal"}:
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules or ["query", "value"],
            task_type=TaskType.SEQ_CLS,
        )
        model = get_peft_model(model, lora_cfg)
    elif cfg.method == "bert_prompt":
        prompt_cfg = PromptTuningConfig(
            task_type=TaskType.SEQ_CLS,
            prompt_tuning_init=PromptTuningInit.RANDOM,
            num_virtual_tokens=cfg.prompt_tokens,
        )
        model = get_peft_model(model, prompt_cfg)

    train_loader = build_dataloader(train, tokenizer, cfg.max_length, cfg.batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler(enabled=cfg.fp16)
    use_focal = cfg.focal_gamma > 0 or cfg.method.endswith("focal")
    focal_loss = FocalLoss(gamma=cfg.focal_gamma) if use_focal else None

    for epoch in range(cfg.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(cfg.device) for k, v in batch.items()}
            labels = batch["labels"]
            with autocast(enabled=cfg.fp16):
                outputs = model(**batch)
                logits = outputs.logits
                loss = focal_loss(logits, labels) if use_focal else outputs.loss
            scaler.scale(loss).backward()
            if (step + 1) % cfg.gradient_accumulation == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        # evaluate & save best checkpoint
```

This loop handles gradient accumulation, mixed precision, LoRA adapter injection, prompt tuning, and focal loss by toggling options in `TransformerRunConfig`.

### Metrics Computation

`evaluation/metrics/compute_metrics.py` exposes helpers used by analyzers:

```python
def compute_classification_metrics(y_true, y_pred, num_classes=None):
    accuracy = accuracy_score(y_true, y_pred)
    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": float(accuracy),
        "precision_micro": float(precision_micro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "confusion_matrix": cm.tolist(),
    }

def compute_latency_stats(latencies):
    latencies_ms = np.array(latencies) * 1000
    return {
        "mean_ms": float(np.mean(latencies_ms)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "max_ms": float(np.max(latencies_ms)),
    }
```

Downstream tools (`evaluation/analyze_results.py`) combine these metrics with prediction metadata to produce reports.

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
