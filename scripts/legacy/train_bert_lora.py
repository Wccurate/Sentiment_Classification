from __future__ import annotations

import argparse
import math
import shutil
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from accelerate import Accelerator
from tqdm.auto import tqdm

# Dataset loaders
from data.loaders.bert_dataloaders import BertTextDataset, DynamicPaddingCollator
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


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute classification accuracy."""
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == labels).sum().item()
    total = labels.numel()
    return correct / max(1, total)


def print_trainable_parameters(model, accelerator):
    """Print trainable vs total parameters."""
    trainable_params = 0
    all_params = 0
    lora_params = 0
    classifier_params = 0
    
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if "lora_" in name:
                lora_params += param.numel()
            elif "classifier" in name or "score" in name:
                classifier_params += param.numel()
    
    accelerator.print(f"\n{'='*70}")
    accelerator.print(f"Model Configuration:")
    accelerator.print(f"  Model type: BERT with LoRA (PEFT)")
    accelerator.print(f"  Total parameters: {all_params:,}")
    accelerator.print(f"  Trainable parameters: {trainable_params:,}")
    accelerator.print(f"  - LoRA adapters: {lora_params:,}")
    accelerator.print(f"  - Classifier head: {classifier_params:,}")
    accelerator.print(f"  Frozen parameters: {all_params - trainable_params:,}")
    accelerator.print(f"  Trainable ratio: {100.0 * trainable_params / all_params:.4f}%")
    accelerator.print(f"{'='*70}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BERT LoRA fine-tuning for intent classification"
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
        default="/usr1/home/s125mdg41_03/models/hf_models/bert_base_cased_google_bert",
        help="Path to pretrained BERT model",
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
        default="outputs/checkpoints/bert_lora",
        help="Output directory for checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per device")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--lora-lr", type=float, default=3e-4, help="Learning rate for LoRA adapters")
    parser.add_argument("--head-lr", type=float, default=1e-3, help="Learning rate for classifier head")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.06, help="Warmup ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")

    # LoRA config
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha (scaling)")
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="query,key,value",
        help="Comma-separated LoRA target modules (e.g., query,key,value,output)",
    )

    # Accelerate
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training",
    )
    parser.add_argument(
        "--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=1.0, help="Max gradient norm for clipping"
    )
    parser.add_argument(
        "--scheduler-type",
        type=str,
        default="linear",
        choices=["linear", "cosine"],
        help="Learning rate scheduler type",
    )

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save-best-only",
        action="store_true",
        help="Only save best checkpoint (saves disk space)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Early stopping patience (0 to disable)",
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
        default="intent-recognition-bert-lora",
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
                accelerator.print("Logged in to W&B with provided API key")
            except Exception as e:
                accelerator.print(f"Failed to login with API key: {e}")
                accelerator.print("Attempting to continue with existing login...")
        
        # Auto-generate run name if not specified
        run_name = args.wandb_run_name
        if run_name is None:
            run_name = f"{args.dataset}_lora_r{args.lora_r}_lr{args.lora_lr}"
        
        # Parse tags
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        tags.extend([args.dataset, "bert", "lora"])
        
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
                "lora_learning_rate": args.lora_lr,
                "head_learning_rate": args.head_lr,
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "val_ratio": args.val_ratio,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "lora_target_modules": args.lora_target_modules,
                "mixed_precision": args.mixed_precision,
                "grad_accum_steps": args.grad_accum_steps,
                "max_grad_norm": args.max_grad_norm,
                "scheduler_type": args.scheduler_type,
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

    # ========== Load Tokenizer ==========
    tokenizer_path = args.tokenizer_path if args.tokenizer_path else args.model_path
    accelerator.print(f"\nLoading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True,
    )

    # ========== Build Datasets & Dataloaders ==========
    train_ds = BertTextDataset(train_split, tokenizer, max_length=args.max_length)
    val_ds = (
        BertTextDataset(val_split, tokenizer, max_length=args.max_length)
        if val_split
        else None
    )

    collator = DynamicPaddingCollator(tokenizer)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator
    )
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
        if val_ds
        else None
    )

    # ========== Load Base Model ==========
    accelerator.print(f"\nLoading BERT model from: {args.model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=num_labels,
    )

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

    # ========== Unfreeze Classification Head ==========
    accelerator.print("\nUnfreezing classification head...")
    for name, param in model.named_parameters():
        if "classifier" in name or "score" in name:
            param.requires_grad = True
            accelerator.print(f"  Unfrozen: {name}")

    print_trainable_parameters(model, accelerator)

    # ========== Separate Parameters for Differential Learning Rates ==========
    lora_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "lora_" in name:
                lora_params.append(param)
            else:
                head_params.append(param)

    accelerator.print(f"\nParameter groups:")
    accelerator.print(f"  LoRA adapters: {sum(p.numel() for p in lora_params):,} params (lr={args.lora_lr})")
    accelerator.print(f"  Classifier head: {sum(p.numel() for p in head_params):,} params (lr={args.head_lr})")

    # ========== Optimizer & Scheduler ==========
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.lora_lr, "weight_decay": args.weight_decay},
            {"params": head_params, "lr": args.head_lr, "weight_decay": args.weight_decay}
        ],
    )

    # Prepare with Accelerator
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    max_train_steps = args.epochs * num_update_steps_per_epoch
    num_warmup_steps = int(args.warmup_ratio * max_train_steps)

    if args.scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=max_train_steps
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=max_train_steps
        )
    scheduler = accelerator.prepare(scheduler)

    accelerator.print(f"\nTraining configuration:")
    accelerator.print(f"   - Total epochs: {args.epochs}")
    accelerator.print(f"   - Steps per epoch: {num_update_steps_per_epoch}")
    accelerator.print(f"   - Total training steps: {max_train_steps}")
    accelerator.print(f"   - Warmup steps: {num_warmup_steps} ({args.warmup_ratio:.1%})")
    accelerator.print(f"   - LoRA learning rate: {args.lora_lr:.2e}")
    accelerator.print(f"   - Head learning rate: {args.head_lr:.2e}")
    accelerator.print(f"   - Effective batch size: {args.batch_size * args.grad_accum_steps * accelerator.num_processes}")
    accelerator.print(f"   - Scheduler: {args.scheduler_type}")
    accelerator.print(f"   - Max grad norm: {args.max_grad_norm}")

    # ========== Training Loop ==========
    best_val_acc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_dir = output_dir / "best_model"
    last_dir = output_dir / "last_state"

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
                    labels=batch["labels"],
                )
                loss = outputs.loss
                logits = outputs.logits

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
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
            current_lora_lr = scheduler.get_last_lr()[0]
            current_head_lr = scheduler.get_last_lr()[1]
            progress_bar.set_postfix(
                {
                    "loss": f"{avg_loss:.4f}",
                    "acc": f"{avg_acc:.4f}",
                    "lora_lr": f"{current_lora_lr:.2e}",
                    "head_lr": f"{current_head_lr:.2e}",
                }
            )
            
            # Log to wandb (every step)
            if use_wandb and accelerator.is_main_process:
                global_step = (epoch - 1) * num_update_steps_per_epoch + step
                wandb.log({
                    "train/loss": avg_loss,
                    "train/accuracy": avg_acc,
                    "train/lora_learning_rate": current_lora_lr,
                    "train/head_learning_rate": current_head_lr,
                    "train/epoch": epoch,
                    "train/step": global_step,
                }, step=global_step)

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
                        labels=batch["labels"],
                    )
                    loss = outputs.loss
                    logits = outputs.logits
                    bs = batch["labels"].size(0)
                    v_loss += loss.detach().float().item() * bs
                    v_acc += compute_accuracy(logits.detach(), batch["labels"]) * bs
                    v_seen += bs

                    val_progress_bar.set_postfix(
                        {
                            "val_loss": f"{v_loss / max(1, v_seen):.4f}",
                            "val_acc": f"{v_acc / max(1, v_seen):.4f}",
                        }
                    )

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
                log_dict.update({
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                })
            wandb.log(log_dict, step=(epoch * num_update_steps_per_epoch))

        # ========== Save Checkpoints ==========
        accelerator.wait_for_everyone()

        # Save last state
        last_dir.mkdir(parents=True, exist_ok=True)
        accelerator.save_state(last_dir)

        # Determine score for best model
        score = float(val_acc) if val_loader else float(train_acc)

        if score > best_val_acc:
            best_val_acc = score
            best_epoch = epoch
            epochs_without_improvement = 0
            
            if accelerator.is_main_process:
                best_dir.mkdir(parents=True, exist_ok=True)
                unwrapped = accelerator.unwrap_model(model)

                # Save LoRA adapter + config (PEFT format)
                unwrapped.save_pretrained(best_dir)

                # Save tokenizer
                tokenizer.save_pretrained(best_dir)
                
                # Save label mappings for later use
                import json
                label_mapping = {
                    "num_labels": num_labels,
                    "label_to_text_label": label_to_text_label,
                    "text_label_to_label": text_label_to_label,
                }
                with open(best_dir / "label_mapping.json", "w", encoding="utf-8") as f:
                    json.dump(label_mapping, f, indent=2, ensure_ascii=False)

                # Save metrics
                with open(best_dir / "metrics.txt", "w", encoding="utf-8") as f:
                    f.write(f"epoch={epoch}\n")
                    f.write(f"train_loss={train_loss:.4f}\n")
                    f.write(f"train_acc={train_acc:.4f}\n")
                    if val_loader:
                        f.write(f"val_loss={val_loss:.4f}\n")
                        f.write(f"val_acc={val_acc:.4f}\n")
                    f.write(f"lora_r={args.lora_r}\n")
                    f.write(f"lora_alpha={args.lora_alpha}\n")
                    f.write(f"lora_target_modules={args.lora_target_modules}\n")

                accelerator.print(
                    f"✅ Saved new best model to: {best_dir} (score={score:.4f})"
                )
                
                # Log best model info to wandb
                if use_wandb:
                    wandb.run.summary["best_val_accuracy"] = best_val_acc
                    wandb.run.summary["best_epoch"] = best_epoch
        else:
            epochs_without_improvement += 1

        # Early stopping check
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            accelerator.print(f"\n⚠️  Early stopping triggered after {epoch} epochs")
            accelerator.print(f"   No improvement for {epochs_without_improvement} epochs")
            accelerator.print(f"   Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
            break

        # Save per-epoch checkpoints (optional)
        if not args.save_best_only:
            snap_dir = output_dir / f"epoch-{epoch}"
            snap_dir.mkdir(parents=True, exist_ok=True)
            accelerator.save_state(snap_dir)

            # Keep only recent 3 epochs to save disk space
            if accelerator.is_main_process:
                existing_epochs = sorted([
                    d for d in output_dir.glob("epoch-*")
                    if d.is_dir()
                ], key=lambda x: int(x.name.split("-")[1]))
                
                if len(existing_epochs) > 3:
                    for old_dir in existing_epochs[:-3]:
                        shutil.rmtree(old_dir)
                        accelerator.print(f"🗑️  Removed old checkpoint: {old_dir.name}")

    # ========== Training Complete ==========
    accelerator.print("\n" + "=" * 70)
    accelerator.print("Training complete!")
    accelerator.print(f"Best validation score: {best_val_acc:.4f} (epoch {best_epoch})")
    accelerator.print(f"Best model saved to: {best_dir}")
    accelerator.print("=" * 70)
    
    # Print loading instructions
    accelerator.print("\n" + "=" * 70)
    accelerator.print("Loading Instructions:")
    accelerator.print("To load the trained model for inference:")
    accelerator.print("  from transformers import AutoModelForSequenceClassification, AutoTokenizer")
    accelerator.print("  from peft import PeftModel")
    accelerator.print(f"  base_model = AutoModelForSequenceClassification.from_pretrained('{args.model_path}', num_labels={num_labels})")
    accelerator.print(f"  model = PeftModel.from_pretrained(base_model, '{best_dir}')")
    accelerator.print(f"  tokenizer = AutoTokenizer.from_pretrained('{best_dir}')")
    accelerator.print("  # For faster inference, merge LoRA weights:")
    accelerator.print("  model = model.merge_and_unload()")
    accelerator.print("=" * 70 + "\n")
    
    # Finalize wandb
    if use_wandb and accelerator.is_main_process:
        # Log final summary
        wandb.run.summary["final_best_val_accuracy"] = best_val_acc
        wandb.run.summary["best_epoch"] = best_epoch
        wandb.run.summary["total_epochs"] = epoch
        wandb.run.summary["best_model_path"] = str(best_dir)
        
        # Calculate efficiency metrics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.run.summary["parameter_efficiency"] = 100.0 * trainable_params / total_params
        wandb.run.summary["trainable_params"] = trainable_params
        wandb.run.summary["total_params"] = total_params
        
        # Save best model as artifact (optional but recommended)
        if best_dir.exists():
            try:
                artifact = wandb.Artifact(
                    name=f"model-{wandb.run.id}",
                    type="model",
                    description=f"Best BERT LoRA model on {args.dataset}",
                    metadata={
                        "dataset": args.dataset,
                        "val_accuracy": float(best_val_acc),
                        "lora_r": args.lora_r,
                        "lora_alpha": args.lora_alpha,
                        "lora_target_modules": args.lora_target_modules,
                        "parameter_efficiency": 100.0 * trainable_params / total_params,
                    }
                )
                artifact.add_dir(str(best_dir))
                wandb.log_artifact(artifact)
                accelerator.print("Model artifact uploaded to W&B")
            except Exception as e:
                accelerator.print(f"Failed to upload artifact: {e}")
        
        wandb.finish()
        accelerator.print("W&B run finished")


if __name__ == "__main__":
    main()
