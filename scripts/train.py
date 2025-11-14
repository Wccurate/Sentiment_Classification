#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import Any, Mapping

try:
    import yaml
except ImportError:  # pragma: no cover - dependency declared in requirements
    yaml = None

from src.utils.config import TrainConfig
from src.utils.logging import get_logger
from src.training.trainer import ClassicalRunConfig, run_tfidf_svm, run_w2v_svm, run_sbert_svm
from src.training.bert_pipeline import TransformerRunConfig, run_transformer_training


def load_experiment_config(path: str) -> dict[str, Any]:
    """Load a YAML or JSON config file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.suffix.lower() in {".yml", ".yaml"}:
            if yaml is None:
                raise RuntimeError("PyYAML is required for YAML config files")
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    return data or {}


def apply_config(namespace: argparse.Namespace, config: Mapping[str, Any]) -> None:
    """Inject config values into the argparse namespace (supports nested dicts)."""

    def _assign(key: str, value: Any):
        if hasattr(namespace, key):
            setattr(namespace, key, value)

    def _walk(prefix: str, value: Any):
        if isinstance(value, Mapping):
            for sub_key, sub_val in value.items():
                _walk(sub_key, sub_val)
        else:
            _assign(prefix, value)

    for top_key, top_val in config.items():
        _walk(top_key, top_val)


def main():
    parser = argparse.ArgumentParser(description="Unified training entry for sentiment classification")
    parser.add_argument("--method", required=False,
                        choices=[
                            "tfidf_svm","w2v_svm","sbert_svm",
                            "bert_frozen","bert_finetune","bert_prompt",
                            "bert_lora","qwen3_lora","qwen3_lora_focal"
                        ])
    parser.add_argument("--config", type=str, help="Optional config file (YAML/JSON)")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--max_len", type=int)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--train_csv", type=str, default="data/train.csv")
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--w2v_path", type=str)
    parser.add_argument("--sbert_name", type=str)
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_modules", type=str, help="Comma-separated LoRA target modules")
    parser.add_argument("--prompt_tokens", type=int, default=20)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.config:
        config_payload = load_experiment_config(args.config)
        apply_config(args, config_payload)

    if not args.method:
        parser.error("--method must be provided via CLI or config file")

    logger = get_logger()
    cfg = TrainConfig(method=args.method)
    if args.epochs: cfg.epochs = args.epochs
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.lr: cfg.lr = args.lr
    if args.max_len: cfg.max_len = args.max_len

    logger.info(f"Starting training with method={cfg.method}")

    lora_targets = None
    if args.lora_target_modules:
        lora_targets = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]

    if cfg.method == "tfidf_svm":
        # Minimal CSV expectation: columns `text`,`label`
        # You can override via config file later.
        run_tfidf_svm(ClassicalRunConfig(
            train_csv=args.train_csv,
            text_col=args.text_col,
            label_col=args.label_col,
            val_ratio=args.val_ratio,
            seed=args.seed,
        ), output_dir=str(Path(args.output_dir) / cfg.method))
        logger.info("TF-IDF+SVM run completed.")
        return

    if cfg.method == "w2v_svm":
        run_w2v_svm(ClassicalRunConfig(
            train_csv=args.train_csv,
            text_col=args.text_col,
            label_col=args.label_col,
            w2v_path=args.w2v_path,
            val_ratio=args.val_ratio,
            seed=args.seed,
        ), output_dir=str(Path(args.output_dir) / cfg.method))
        logger.info("Word2Vec+SVM run completed.")
        return

    if cfg.method == "sbert_svm":
        run_sbert_svm(ClassicalRunConfig(
            train_csv=args.train_csv,
            text_col=args.text_col,
            label_col=args.label_col,
            sbert_name=args.sbert_name or "BAAI/bge-small-en-v1.5",
            val_ratio=args.val_ratio,
            seed=args.seed,
        ), output_dir=str(Path(args.output_dir) / cfg.method))
        logger.info("Sentence-BERT+SVM run completed.")
        return

    if cfg.method in {"bert_frozen", "bert_finetune", "bert_prompt", "bert_lora", "qwen3_lora", "qwen3_lora_focal"}:
        run_transformer_training(TransformerRunConfig(
            method=cfg.method,
            train_csv=args.train_csv,
            text_col=args.text_col,
            label_col=args.label_col,
            val_ratio=args.val_ratio,
            model_name=args.model_name,
            output_dir=str(Path(args.output_dir) / cfg.method),
            max_length=args.max_len or cfg.max_len,
            batch_size=args.batch_size or cfg.batch_size,
            epochs=args.epochs or cfg.epochs,
            lr=args.lr or cfg.lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            gradient_accumulation=args.grad_accum,
            fp16=not args.no_fp16,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
             lora_target_modules=lora_targets,
            prompt_tokens=args.prompt_tokens,
            focal_gamma=args.focal_gamma if cfg.method.endswith("focal") else 0.0,
            trust_remote_code=args.trust_remote_code,
            seed=args.seed,
        ))
        logger.info("Transformer run completed.")
        return

    logger.info("Selected method not yet wired. To be implemented next.")


if __name__ == "__main__":
    main()
