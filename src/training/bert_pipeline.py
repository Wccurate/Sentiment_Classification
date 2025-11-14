from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import json
import math

import numpy as np
import torch
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)
from peft import LoraConfig, PromptTuningConfig, PromptTuningInit, TaskType, get_peft_model

from src.data.datasets import load_csv_dataset, split_dataset
from src.data.dataloaders import build_dataloader
from src.training.losses import FocalLoss
from src.utils.logging import get_logger


@dataclass
class TransformerRunConfig:
    method: str
    train_csv: str
    text_col: str = "text"
    label_col: str = "label"
    val_ratio: float = 0.1
    model_name: str = "bert-base-uncased"
    output_dir: str = "outputs/transformer"
    max_length: int = 128
    batch_size: int = 16
    epochs: int = 3
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation: int = 1
    fp16: bool = True
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # LoRA
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] | None = None
    # Prompt tuning
    prompt_tokens: int = 20
    # Loss tweaks
    focal_gamma: float = 0.0
    # Model loading
    trust_remote_code: bool = False


def run_transformer_training(cfg: TransformerRunConfig):
    logger = get_logger()
    set_seed(cfg.seed)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading dataset from {cfg.train_csv}")
    data = load_csv_dataset(cfg.train_csv, cfg.text_col, cfg.label_col)
    train, val = split_dataset(data, cfg.val_ratio, cfg.seed)
    labels = sorted({ex.label for ex in data if ex.label is not None})
    if not labels:
        raise ValueError("Training data requires labels")
    num_labels = len(labels)
    id2label = {i: str(lbl) for i, lbl in enumerate(labels)}
    label2id = {str(lbl): i for i, lbl in enumerate(labels)}

    label_map = {orig: idx for idx, orig in enumerate(labels)}
    for ex in data:
        if ex.label is not None:
            ex.label = label_map[ex.label]
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=cfg.trust_remote_code)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        trust_remote_code=cfg.trust_remote_code,
    )

    if cfg.method == "bert_frozen":
        for param in model.base_model.parameters():
            param.requires_grad = False
    elif cfg.method in {"bert_lora", "qwen3_lora", "qwen3_lora_focal"}:
        target_modules = cfg.lora_target_modules or ["query", "value"]
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_CLS,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_cfg)
    elif cfg.method == "bert_prompt":
        prompt_cfg = PromptTuningConfig(
            task_type=TaskType.SEQ_CLS,
            prompt_tuning_init=PromptTuningInit.RANDOM,
            num_virtual_tokens=cfg.prompt_tokens,
        )
        model = get_peft_model(model, prompt_cfg)

    model.to(cfg.device)
    train_loader = build_dataloader(train, tokenizer, cfg.max_length, cfg.batch_size, shuffle=True)
    val_loader = build_dataloader(val, tokenizer, cfg.max_length, cfg.batch_size, shuffle=False)

    total_steps = math.ceil(len(train_loader) / cfg.gradient_accumulation) * cfg.epochs
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * cfg.warmup_ratio),
        num_training_steps=total_steps,
    )

    scaler = GradScaler(enabled=cfg.fp16)
    use_focal = cfg.focal_gamma > 0 or cfg.method.endswith("focal")
    focal_loss = FocalLoss(gamma=cfg.focal_gamma) if use_focal else None

    def eval_model():
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(cfg.device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits
                preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
                trues.extend(batch["labels"].cpu().tolist())
        trues = np.array(trues)
        preds_arr = np.array(preds)
        acc = float((preds_arr == trues).mean())
        return acc

    best_acc = 0.0
    best_state_path = Path(cfg.output_dir) / "checkpoint-best"

    global_step = 0
    for epoch in range(cfg.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(cfg.device) for k, v in batch.items()}
            labels = batch["labels"]
            with autocast(enabled=cfg.fp16):
                outputs = model(**batch)
                logits = outputs.logits
                if use_focal and focal_loss is not None:
                    loss = focal_loss(logits, labels)
                else:
                    loss = outputs.loss if hasattr(outputs, "loss") else torch.nn.functional.cross_entropy(logits, labels)
            loss = loss / cfg.gradient_accumulation
            scaler.scale(loss).backward()

            if (step + 1) % cfg.gradient_accumulation == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        val_acc = eval_model()
        logger.info(f"Epoch {epoch+1}/{cfg.epochs} - val_acc={val_acc:.4f}")
        if val_acc >= best_acc:
            best_acc = val_acc
            model.save_pretrained(best_state_path)
            tokenizer.save_pretrained(best_state_path)

    metrics = {"best_val_accuracy": best_acc}
    with open(Path(cfg.output_dir) / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Training complete. Best accuracy={best_acc:.4f}")

    return metrics
