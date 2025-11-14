# Sentiment Classification — EE6483 Project

This repository consolidates multiple sentiment-classification strategies into a single, configurable codebase. Classical ML pipelines (TF-IDF / Word2Vec / Sentence-BERT + SVM) coexist with deep models ranging from frozen BERT heads to LoRA-tuned Qwen models, all exposed through one training entry point.

> This version is a refactor of the legacy intent/sentiment project. Legacy scripts are still present for reference, but every experiment can now be launched via the unified interfaces described below.

## Highlights

- Unified training/evaluation scripts with reusable modules under `src/`
- Prebuilt YAML configs mirroring the original course experiments
- Support for classical features, full fine-tuning, prompt tuning, LoRA, and focal-loss variants

## Methods Covered

1. TF-IDF + PCA + SVM
2. Word2Vec (average pooling) + SVM
3. Sentence-BERT (Bilingual-Embedding-Small) + SVM
4. BERT + MLP (frozen encoder)
5. BERT + MLP (full fine-tuning)
6. Prompt-Tuning + BERT + MLP
7. LoRA + BERT + MLP
8. LoRA + Qwen3 + MLP
9. LoRA + Qwen3 + MLP + Focal Loss

## Setup

```bash
python -m venv .venv && source .venv/bin/activate  # optional
pip install -r requirements.txt
```

> If you need gated checkpoints (e.g., Qwen), configure Hugging Face access tokens before running the scripts.

## Training

### Config-driven runs (recommended)

Each YAML config under `configs/training/` encodes the parameters that used to live in separate shell scripts. Launch any experiment with a single command:

```bash
python scripts/train.py --config configs/training/bert_lora.yaml
```

Available configs:

| Config | Method | Mirrors original script |
| --- | --- | --- |
| `configs/training/bert_head_encoder.yaml` | `bert_finetune` | `scripts/training/bert_head_encoder_train.sh` |
| `configs/training/bert_lora.yaml` | `bert_lora` | `scripts/training/bert_lora_train.sh` |
| `configs/training/bert_prompt_tuning.yaml` | `bert_prompt` | `scripts/training/bert_p_tuning.sh` |
| `configs/training/qwen3_lora_focal.yaml` | `qwen3_lora_focal` | `scripts/training/qwen3_lora_train.sh` |

Feel free to duplicate a file, change `train_csv`, `model_name`, `epochs`, etc., and point the command to your custom config. All keys map directly onto the CLI options of `scripts/train.py`.

### Direct CLI usage (optional)

If you prefer ad-hoc runs, the same entry point still accepts explicit arguments:

```bash
# Classical TF-IDF baseline
python scripts/train.py --method tfidf_svm --train_csv data/train.csv --output_dir outputs

# Full BERT fine-tuning
python scripts/train.py --method bert_finetune --train_csv data/train.csv \
    --model_name bert-base-cased --epochs 3 --batch_size 32 --lr 2e-5
```

## Evaluation

Classical methods share a unified evaluator (transformer evaluation is coming next):

```bash
python scripts/evaluate.py \
    --method tfidf_svm \
    --checkpoint outputs/tfidf_svm \
    --data_csv data/val.csv \
    --output_dir outputs/eval
```

For deeper analysis, convert `evaluation_results.json` into metrics/reports:

```bash
python -m evaluation.analyze_results outputs/eval/evaluation_results.json
```

## Pretrained Checkpoints

Use these ready-made checkpoints to reproduce the full runs quickly (download and place them under `outputs/<method>/checkpoint-best` or point `scripts/evaluate.py` to the extracted folder):

| Method | Checkpoint |
| --- | --- |
| BERT full fine-tuning | [Google Drive](https://drive.google.com/file/d/1_NzccVd1DF5vPv3NJCkN1jWYcj5uF3az/view?usp=drive_link) |
| BERT LoRA adapter | [Google Drive](https://drive.google.com/file/d/18G7fm_-uRdlAy9QZLc-udd16vzzQC3dh/view?usp=drive_link) |
| BERT Prompt Tuning adapter | [Google Drive](https://drive.google.com/file/d/1uXRLKbD4iDH0XHX_DXy7WN4iQvfT9N6d/view?usp=drive_link) |
| Qwen3 LoRA + Focal adapter | [Google Drive](https://drive.google.com/file/d/13V_L3p_OJno5CccBJgc7hBoUkzrrwwO_/view?usp=drive_link) |
| Qwen3 LoRA adapter | [Google Drive](https://drive.google.com/file/d/1YZESoaXCW6dwFUHcI5FI-mPaBBhLX27Z/view?usp=drive_link) |

Download the corresponding base model (e.g., `bert-base-cased`, `Qwen/Qwen1.5-1.8B-Chat`) and place it under `models/huggingface_models/` before loading the adapters.

### Base Model Download Scripts

Helper scripts for mirrored checkpoints live in `scripts/download/`:

```bash
# Download datasets into data/raw/
python scripts/download/dataset_download.py --base-dir data/raw

# Download BERT / other encoder models into models/huggingface_models/
python scripts/download/model_download.py --base-dir models/huggingface_models

# Download sentence embedder backbones
python scripts/download/download_embedder.py --base-dir models/huggingface_models
```

Adjust the script arguments to match the model you need; once downloaded, point `scripts/train.py` or `scripts/evaluate.py` to the local model paths.

## Repository Layout

- `src/` – reusable packages (data loaders, feature builders, models, trainers, evaluators, utils)
- `scripts/train.py` – unified training entry (config-aware)
- `scripts/evaluate.py` – unified evaluation entry (classical methods today)
- `configs/training/` – YAML configs for the main experiments
- `scripts/legacy/` – previous ad-hoc scripts preserved for reference
- `outputs/`, `results/`, `wandb/` – default artifact locations

## Legacy Scripts

All original `train_*.py`, `eval_*.py`, and helper bash files now live under `scripts/legacy/`. They can still be executed directly when you need the exact historical behavior:

```bash
# Training examples
bash scripts/training/bert_head_encoder_train.sh
bash scripts/training/bert_lora_train.sh
bash scripts/training/bert_p_tuning.sh
bash scripts/training/qwen3_lora_train.sh

# Evaluation examples
python scripts/legacy/eval_unified.py --help
python scripts/legacy/eval_bert_improved.py --dataset project --model-path outputs/.../best_model
```

Use these when you want 1:1 compatibility with the prior workflow. Otherwise, prefer the unified `scripts/train.py`/`scripts/evaluate.py` entry points.

## Notes

- The provided configs assume CSV datasets (columns: `text`, `label`). Update `train_csv`, `text_col`, or `label_col` in the YAML files to match your data.
- LoRA and Qwen configs may require larger GPUs. Adjust batch size, sequence length, or `gradient_accumulation` as needed.
- Remove or replace sensitive strings (e.g., W&B API keys) inside the configs before sharing them externally.
