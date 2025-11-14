from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from accelerate import Accelerator
from tqdm.auto import tqdm

# Dataset loaders
from data.loaders.load_raw_data import (
    load_raw_atis,
    load_raw_hwu64,
    load_raw_snips,
    load_raw_clinc_oos,
    load_raw_project,
)

# Import wandb with error handling
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed. Install with: pip install wandb")

warnings.filterwarnings("ignore", category=FutureWarning)


# ========== Focal Loss Implementation ==========

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Class weights tensor of shape (num_classes,). 
               If None, no weighting is applied.
        gamma: Focusing parameter. Higher gamma increases focus on hard examples.
               gamma=0 reduces to standard cross-entropy.
        reduction: 'mean', 'sum', or 'none'
    """
    def __init__(
        self, 
        alpha: Optional[torch.Tensor] = None, 
        gamma: float = 2.0, 
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, num_classes) - raw model outputs
            targets: (batch_size,) - ground truth class indices
            
        Returns:
            loss: scalar tensor
        """
        # Compute softmax probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Get probability of the true class
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1))
        p_t = (probs * targets_one_hot).sum(dim=-1)  # (batch_size,)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Cross-entropy loss: -log(p_t)
        ce_loss = -torch.log(p_t + 1e-8)
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets]  # (batch_size,)
            loss = alpha_t * focal_weight * ce_loss
        else:
            loss = focal_weight * ce_loss
        
        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def compute_class_weights(
    labels: list, 
    num_classes: int, 
    method: str = 'inverse_sqrt'
) -> torch.Tensor:
    """
    Compute class weights based on label distribution.
    
    Args:
        labels: List of integer labels
        num_classes: Total number of classes
        method: 'inverse' - inverse frequency
                'inverse_sqrt' - inverse square root of frequency (smoother)
                'effective_samples' - effective number of samples (beta=0.9999)
    
    Returns:
        weights: Tensor of shape (num_classes,) with weights normalized to sum to num_classes
    """
    label_array = np.array(labels)
    class_counts = np.bincount(label_array, minlength=num_classes)
    
    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1)
    
    if method == 'inverse':
        weights = 1.0 / class_counts
    elif method == 'inverse_sqrt':
        weights = 1.0 / np.sqrt(class_counts)
    elif method == 'effective_samples':
        # Effective number of samples: (1 - beta^n) / (1 - beta)
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / effective_num
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    # Normalize weights to sum to num_classes (keeps loss magnitude similar)
    weights = weights * (num_classes / weights.sum())
    
    return torch.tensor(weights, dtype=torch.float32)


# ========== Dataset Loading ==========

def get_loader_by_name(name: str):
    """Return dataset loader function by name."""
    name = name.lower()
    if name == "atis":
        return load_raw_atis
    if name == "hwu64":
        return load_raw_hwu64
    if name == "snips":
        return load_raw_snips
    if name == "clinc_oos":
        return load_raw_clinc_oos
    if name == "project":
        return load_raw_project
    raise ValueError(f"Unknown dataset name: {name}")


def stratified_split(train_ds, val_ratio: float = 0.1, seed: int = 42):
    """
    Split HF Dataset into train/val stratified by label.
    
    Returns:
        (new_train_ds, val_ds)
    """
    if val_ratio <= 0 or val_ratio >= 1:
        return train_ds, None

    labels = train_ds["label"]
    n = len(labels)
    rng = np.random.default_rng(seed)

    idx_by_label: Dict[int, list] = {}
    for idx, y in enumerate(labels):
        idx_by_label.setdefault(int(y), []).append(idx)

    val_indices = []
    for y, idxs in idx_by_label.items():
        m = len(idxs)
        if m <= 1:
            continue
        k = int(round(m * val_ratio))
        if k <= 0 and val_ratio > 0:
            k = 1
        if k >= m:
            k = m - 1
        if k > 0:
            chosen = rng.choice(idxs, size=k, replace=False).tolist()
            val_indices.extend(chosen)

    val_set = set(val_indices)
    all_idx = set(range(n))
    train_indices = sorted(all_idx - val_set)
    val_indices = sorted(val_set)

    if len(val_indices) == 0:
        return train_ds, None

    new_train = train_ds.select(train_indices)
    new_val = train_ds.select(val_indices)
    return new_train, new_val


# ========== Dataset & Collator ==========

class Qwen3TextDataset(torch.utils.data.Dataset):
    """Simple text classification dataset for Qwen3."""

    def __init__(self, raw_data, tokenizer, max_length: int = 512):
        self.data = raw_data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = sample["text"]
        label = int(sample["label"])

        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,  # dynamic padding in collator
            return_tensors=None,
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": label,
        }


class DynamicPaddingCollator:
    """Dynamic padding collator for batching."""

    def __init__(self, tokenizer, padding_side: str = "right"):
        self.tokenizer = tokenizer
        self.padding_side = padding_side

    def __call__(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]

        # Pad sequences
        max_len = max(len(ids) for ids in input_ids)

        padded_input_ids = []
        padded_attention_mask = []

        for ids, mask in zip(input_ids, attention_mask):
            pad_len = max_len - len(ids)
            if self.padding_side == "right":
                padded_input_ids.append(ids + [self.tokenizer.pad_token_id] * pad_len)
                padded_attention_mask.append(mask + [0] * pad_len)
            else:
                padded_input_ids.append([self.tokenizer.pad_token_id] * pad_len + ids)
                padded_attention_mask.append([0] * pad_len + mask)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ========== Metrics ==========

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute classification accuracy."""
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == labels).sum().item()
    total = labels.numel()
    return correct / max(1, total)


def print_trainable_parameters(model):
    """Print trainable vs total parameters."""
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f"Trainable params: {trainable_params:,} || "
          f"All params: {all_params:,} || "
          f"Trainable%: {100 * trainable_params / all_params:.2f}%")


# ========== Argument Parser ==========

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3 1.7B LoRA fine-tuning with Focal Loss for intent classification"
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="atis",
        choices=["atis", "hwu64", "snips", "clinc_oos", "project"],
        help="Dataset to train on",
    )
    parser.add_argument(
        "--clinc-version",
        type=str,
        default="plus",
        choices=["small", "plus", "imbalanced"],
        help="CLINC OOS dataset version",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/raw",
        help="Base directory for raw datasets",
    )

    # Model
    parser.add_argument(
        "--model-path",
        type=str,
        default="/projects/wangshibohdd/models/model_from_hf/qwen317b",
        help="Path to pretrained Qwen3 model",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="",
        help="Tokenizer path (defaults to model-path)",
    )

    # Training
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/checkpoints/qwen3_lora_focal",
        help="Output directory for checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per device")
    parser.add_argument("--max-length", type=int, default=256, help="Max sequence length")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for LoRA + head")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.06, help="Warmup ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")

    # LoRA config
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (scaling)")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target modules",
    )

    # Focal Loss config
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma parameter (0 = CE loss, higher = more focus on hard examples)",
    )
    parser.add_argument(
        "--class-weight-method",
        type=str,
        default="inverse_sqrt",
        choices=["inverse", "inverse_sqrt", "effective_samples"],
        help="Method for computing class weights",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class weighting (only use focal term)",
    )

    # Accelerate
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training",
    )
    parser.add_argument(
        "--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps"
    )

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save-best-only",
        action="store_true",
        help="Only save best checkpoint (saves disk space)",
    )

    # Wandb
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-api-key",
        type=str,
        default=None,
        help="W&B API key (if not set, will use WANDB_API_KEY env var or saved login)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="intent-recognition-qwen3-lora",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity (username or team)",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name (auto-generated if not specified)",
    )
    parser.add_argument(
        "--wandb-tags",
        type=str,
        default="",
        help="Comma-separated W&B tags",
    )

    return parser.parse_args()


# ========== Main Training Loop ==========

def main():
    args = parse_args()

    # Initialize wandb
    use_wandb = args.use_wandb and WANDB_AVAILABLE
    if args.use_wandb and not WANDB_AVAILABLE:
        print("--use-wandb specified but wandb not available. Continuing without wandb.")
    
    # Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum_steps,
        mixed_precision=args.mixed_precision,
        log_with=None,
        project_dir=args.output_dir,
    )

    # Set seed
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb on main process only
    if use_wandb and accelerator.is_main_process:
        # Login with API key if provided
        if args.wandb_api_key:
            try:
                wandb.login(key=args.wandb_api_key)
            except Exception as e:
                accelerator.print(f"Warning: wandb login failed: {e}")
        
        # Auto-generate run name if not specified
        run_name = args.wandb_run_name
        if run_name is None:
            run_name = f"{args.dataset}_focal_gamma{args.focal_gamma}_r{args.lora_r}_lr{args.lr}"
        
        # Parse tags
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        tags.extend([args.dataset, "qwen3", "lora", "focal_loss"])
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            tags=tags,
            config={
                "dataset": args.dataset,
                "model_path": args.model_path,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "val_ratio": args.val_ratio,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "lora_target_modules": args.lora_target_modules,
                "focal_gamma": args.focal_gamma,
                "class_weight_method": args.class_weight_method,
                "use_class_weights": not args.no_class_weights,
                "mixed_precision": args.mixed_precision,
                "grad_accum_steps": args.grad_accum_steps,
                "seed": args.seed,
            },
        )
        accelerator.print("Weights & Biases initialized")

    # ========== Load Dataset ==========
    accelerator.print(f"📂 Loading dataset: {args.dataset}")
    loader_fn = get_loader_by_name(args.dataset)

    if args.dataset == "clinc_oos":
        raw_dataset, stats, text_label_to_label, label_to_text_label = loader_fn(
            data_dir=args.dataset_dir, version=args.clinc_version, return_dicts=True
        )
    else:
        raw_dataset, stats, text_label_to_label, label_to_text_label = loader_fn(
            data_dir=args.dataset_dir, return_dicts=True
        )

    assert "train" in raw_dataset, "Expected 'train' split in dataset"
    base_train = raw_dataset["train"]
    train_split, val_split = stratified_split(
        base_train, val_ratio=args.val_ratio, seed=args.seed
    )

    num_labels = len(set(train_split["label"]))
    accelerator.print(f"Train samples: {len(train_split)}")
    if val_split:
        accelerator.print(f"Validation samples: {len(val_split)}")
    accelerator.print(f"Number of labels: {num_labels}")

    # ========== Compute Class Weights ==========
    train_labels = train_split["label"]
    class_weights = None
    
    if not args.no_class_weights:
        class_weights = compute_class_weights(
            train_labels, 
            num_labels, 
            method=args.class_weight_method
        )
        
        accelerator.print(f"\n📊 Class distribution and weights:")
        label_counts = np.bincount(train_labels, minlength=num_labels)
        for i in range(num_labels):
            label_name = label_to_text_label.get(i, f"class_{i}")
            accelerator.print(
                f"  Class {i:2d} ({label_name:30s}): "
                f"{label_counts[i]:5d} samples, weight: {class_weights[i]:.4f}"
            )
        
        # Log class distribution to wandb
        if use_wandb and accelerator.is_main_process:
            class_dist_table = wandb.Table(
                columns=["class_id", "class_name", "count", "weight"],
                data=[
                    [i, label_to_text_label.get(i, f"class_{i}"), 
                     int(label_counts[i]), float(class_weights[i])]
                    for i in range(num_labels)
                ]
            )
            wandb.log({"class_distribution": class_dist_table})
    else:
        accelerator.print("\n⚠️  Class weighting disabled (using only focal term)")

    # ========== Load Tokenizer ==========
    tokenizer_path = args.tokenizer_path if args.tokenizer_path else args.model_path
    accelerator.print(f"\nLoading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=True,
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        accelerator.print(f"Set pad_token to eos_token: {tokenizer.pad_token}")
    
    # Store pad_token_id for later use
    pad_token_id = tokenizer.pad_token_id
    accelerator.print(f"pad_token_id: {pad_token_id}")

    # ========== Build Datasets & Dataloaders ==========
    train_ds = Qwen3TextDataset(train_split, tokenizer, max_length=args.max_length)
    val_ds = (
        Qwen3TextDataset(val_split, tokenizer, max_length=args.max_length)
        if val_split
        else None
    )

    collator = DynamicPaddingCollator(tokenizer, padding_side="right")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator
    )
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
        if val_ds
        else None
    )

    # ========== Load Base Model ==========
    accelerator.print(f"\nLoading Qwen3 model from: {args.model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=num_labels,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32,
    )

    # Resize token embeddings if tokenizer was modified
    model.resize_token_embeddings(len(tokenizer))
    
    # Set pad_token_id in model config (critical for batch processing)
    model.config.pad_token_id = pad_token_id
    accelerator.print(f"Set model.config.pad_token_id to: {model.config.pad_token_id}")

    # ========== Freeze Base Model ==========
    accelerator.print("\nFreezing all base model parameters...")
    for name, param in model.named_parameters():
        param.requires_grad = False

    accelerator.print(f"All parameters frozen")

    # ========== Apply LoRA ==========
    accelerator.print(f"\nApplying LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")

    target_modules = [m.strip() for m in args.lora_target_modules.split(",")]

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        inference_mode=False,
    )

    model = get_peft_model(model, lora_config)

    # Unfreeze classification head (score layer)
    accelerator.print("\nUnfreezing classification head...")
    for name, param in model.named_parameters():
        if "score" in name or "classifier" in name:
            param.requires_grad = True

    print_trainable_parameters(model)
    
    head_params = []
    lora_adapters = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "lora_" in name:
                lora_adapters.append(param)
            else:
                head_params.append(param)

    # ========== Initialize Focal Loss ==========
    accelerator.print(f"\n🎯 Initializing Focal Loss:")
    accelerator.print(f"   - Gamma: {args.focal_gamma}")
    accelerator.print(f"   - Class weights: {'enabled' if not args.no_class_weights else 'disabled'}")
    if not args.no_class_weights:
        accelerator.print(f"   - Weight method: {args.class_weight_method}")
    
    # Move class weights to device if they exist
    if class_weights is not None:
        class_weights = class_weights.to(accelerator.device)
    
    criterion = FocalLoss(
        alpha=class_weights,
        gamma=args.focal_gamma,
        reduction='mean'
    )

    # ========== Optimizer & Scheduler ==========
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_adapters, "lr": args.lr},
            {"params": head_params, "lr": args.lr * 30}
        ],
        weight_decay=args.weight_decay,
    )

    # Prepare with Accelerator
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    max_train_steps = args.epochs * num_update_steps_per_epoch
    num_warmup_steps = int(args.warmup_ratio * max_train_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=max_train_steps
    )
    scheduler = accelerator.prepare(scheduler)

    accelerator.print(f"\nTraining configuration:")
    accelerator.print(f"   - Total epochs: {args.epochs}")
    accelerator.print(f"   - Steps per epoch: {num_update_steps_per_epoch}")
    accelerator.print(f"   - Total training steps: {max_train_steps}")
    accelerator.print(f"   - Warmup steps: {num_warmup_steps} ({args.warmup_ratio:.1%})")
    accelerator.print(f"   - Learning rate (LoRA): {args.lr:.2e}")
    accelerator.print(f"   - Learning rate (head): {args.lr * 30:.2e}")
    accelerator.print(f"   - Effective batch size: {args.batch_size * args.grad_accum_steps}")

    # ========== Training Loop ==========
    best_val_acc = -1.0
    best_dir = output_dir / "best_model"
    last_dir = output_dir / "last_state"
    recent_epoch_dirs = []
    max_grad_norm = 1.0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        seen = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs}",
            disable=not accelerator.is_local_main_process,
        )

        for step, batch in enumerate(progress_bar, start=1):
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                logits = outputs.logits
                
                # Use Focal Loss instead of standard CE
                loss = criterion(logits, batch["labels"])

                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Metrics
            batch_size = batch["labels"].size(0)
            running_loss += loss.detach().float().item() * batch_size
            running_acc += compute_accuracy(logits.detach(), batch["labels"]) * batch_size
            seen += batch_size

            # Update progress bar
            avg_loss = running_loss / max(1, seen)
            avg_acc = running_acc / max(1, seen)
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix(
                {"loss": f"{avg_loss:.4f}", "acc": f"{avg_acc:.4f}", "lr": f"{current_lr:.2e}"}
            )
            
            # Log to wandb (every step)
            if use_wandb and accelerator.is_main_process:
                wandb.log({
                    "train/step_loss": loss.detach().float().item(),
                    "train/step_accuracy": compute_accuracy(logits.detach(), batch["labels"]),
                    "train/learning_rate": current_lr,
                    "global_step": (epoch - 1) * len(train_loader) + step,
                })

        train_loss = running_loss / max(1, seen)
        train_acc = running_acc / max(1, seen)

        # ========== Validation ==========
        val_acc = None
        val_loss = None
        if val_loader is not None:
            model.eval()
            v_loss = 0.0
            v_acc = 0.0
            v_seen = 0

            val_progress_bar = tqdm(
                val_loader,
                desc="Validation",
                disable=not accelerator.is_local_main_process,
            )

            with torch.no_grad():
                for batch in val_progress_bar:
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                    logits = outputs.logits
                    
                    # Use Focal Loss for validation as well
                    loss = criterion(logits, batch["labels"])

                    batch_size = batch["labels"].size(0)
                    v_loss += loss.detach().float().item() * batch_size
                    v_acc += compute_accuracy(logits, batch["labels"]) * batch_size
                    v_seen += batch_size

            val_loss = v_loss / max(1, v_seen)
            val_acc = v_acc / max(1, v_seen)

        accelerator.print(
            f"Epoch {epoch} done | train_loss={train_loss:.4f} train_acc={train_acc:.4f}"
            + (
                ""
                if val_loader is None
                else f" | val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )
        )
        
        # Log epoch-level metrics to wandb
        if use_wandb and accelerator.is_main_process:
            log_dict = {
                "epoch": epoch,
                "train/epoch_loss": train_loss,
                "train/epoch_accuracy": train_acc,
            }
            if val_loader is not None:
                log_dict["val/epoch_loss"] = val_loss
                log_dict["val/epoch_accuracy"] = val_acc
            wandb.log(log_dict)

        # ========== Save Checkpoints ==========
        accelerator.wait_for_everyone()

        # Save last state
        last_dir.mkdir(parents=True, exist_ok=True)
        accelerator.save_state(last_dir)

        # Determine score for best model
        score = float(val_acc) if val_loader else float(train_acc)

        if score > best_val_acc:
            best_val_acc = score
            accelerator.print(f"🌟 New best score: {best_val_acc:.4f}")

            # Save best model
            best_dir.mkdir(parents=True, exist_ok=True)
            
            unwrapped_model = accelerator.unwrap_model(model)
            
            # Save LoRA adapters
            if isinstance(unwrapped_model, PeftModel):
                unwrapped_model.save_pretrained(best_dir)
                accelerator.print(f"   - LoRA adapters saved to {best_dir}")
            
            # Save tokenizer
            if accelerator.is_main_process:
                tokenizer.save_pretrained(best_dir)
                accelerator.print(f"   - Tokenizer saved to {best_dir}")
            
            # Save training args
            if accelerator.is_main_process:
                args_file = best_dir / "training_args.json"
                import json
                with open(args_file, "w") as f:
                    json.dump(vars(args), f, indent=2)
                accelerator.print(f"   - Training args saved to {args_file}")

        # Save per-epoch checkpoints (optional)
        if not args.save_best_only:
            epoch_dir = output_dir / f"checkpoint_epoch_{epoch}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            accelerator.save_state(epoch_dir)
            
            recent_epoch_dirs.append(epoch_dir)
            
            # Keep only last 3 epoch checkpoints
            if len(recent_epoch_dirs) > 3:
                old_dir = recent_epoch_dirs.pop(0)
                import shutil
                if old_dir.exists():
                    shutil.rmtree(old_dir)
                    accelerator.print(f"   - Removed old checkpoint: {old_dir.name}")

    # ========== Training Complete ==========
    accelerator.print("\n" + "=" * 60)
    accelerator.print("Training complete!")
    accelerator.print(f"Best validation score: {best_val_acc:.4f}")
    accelerator.print(f"Best model saved to: {best_dir}")
    if not args.save_best_only:
        accelerator.print(f"Recent epoch checkpoints: {[d.name for d in recent_epoch_dirs]}")
    accelerator.print("=" * 60)
    
    # Finalize wandb
    if use_wandb and accelerator.is_main_process:
        # Log final summary
        wandb.run.summary["final_best_val_accuracy"] = best_val_acc
        wandb.run.summary["total_epochs"] = args.epochs
        wandb.run.summary["best_model_path"] = str(best_dir)
        wandb.run.summary["focal_gamma"] = args.focal_gamma
        wandb.run.summary["class_weight_method"] = args.class_weight_method
        
        # Save best model as artifact (optional but recommended)
        if best_dir.exists():
            try:
                artifact = wandb.Artifact(
                    name=f"{args.dataset}_qwen3_lora_focal_best",
                    type="model",
                    description=f"Best Qwen3 LoRA model with Focal Loss on {args.dataset} (val_acc={best_val_acc:.4f})",
                    metadata={
                        "dataset": args.dataset,
                        "val_accuracy": best_val_acc,
                        "focal_gamma": args.focal_gamma,
                        "class_weight_method": args.class_weight_method,
                        "lora_r": args.lora_r,
                        "lora_alpha": args.lora_alpha,
                    }
                )
                artifact.add_dir(str(best_dir))
                wandb.log_artifact(artifact)
                accelerator.print("   - Best model uploaded to W&B as artifact")
            except Exception as e:
                accelerator.print(f"   - Warning: Could not upload model artifact: {e}")
        
        wandb.finish()
        accelerator.print("W&B run finished")


if __name__ == "__main__":
    main()
