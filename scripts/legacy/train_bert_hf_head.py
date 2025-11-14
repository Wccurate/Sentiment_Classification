from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
	AutoModelForSequenceClassification,
	AutoTokenizer,
	get_linear_schedule_with_warmup,
	set_seed,
)
from accelerate import Accelerator
from tqdm.auto import tqdm

from data.loaders.bert_dataloaders import BertTextDataset, DynamicPaddingCollator
from datasets import DatasetDict

# Import wandb with error handling
try:
	import wandb
	WANDB_AVAILABLE = True
except ImportError:
	WANDB_AVAILABLE = False
	print("wandb not installed. Install with: pip install wandb")

warnings.filterwarnings("ignore", category=FutureWarning)

# Dataset loaders
from data.loaders.load_raw_data import (
	load_raw_atis,
	load_raw_hwu64,
	load_raw_snips,
	load_raw_clinc_oos,
	load_raw_project,
)


DEFAULT_MODEL_PATH = \
	"/Users/wangshibo/Documents/Academic/Course/6405Project/Intent_Recognition_Exp/models/huggingface_models/bert_base_cased_google_bert"
DEFAULT_TOKENIZER_PATH = DEFAULT_MODEL_PATH
DEFAULT_OUTPUT_DIR = "outputs/checkpoints/bert_hf_native_train"


def get_loader_by_name(name: str):
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
	Split a HF Dataset into train/val stratified by 'label' without requiring
	ClassLabel dtype. Works with integer Value labels.

	Returns:
		(new_train_ds, val_ds)
	"""
	if val_ratio <= 0 or val_ratio >= 1:
		return train_ds, None

	labels = train_ds["label"]
	n = len(labels)
	rng = np.random.default_rng(seed)

	# Group indices by label value
	idx_by_label: Dict[int, list] = {}
	for idx, y in enumerate(labels):
		idx_by_label.setdefault(int(y), []).append(idx)

	val_indices = []
	for y, idxs in idx_by_label.items():
		m = len(idxs)
		if m <= 1:
			# Too few samples for this label to split; keep all in train
			continue
		# Desired count per class; ensure at least 1 and leave at least 1 for train
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

	# Edge cases: if nothing selected for val, return original as train only
	if len(val_indices) == 0:
		return train_ds, None

	new_train = train_ds.select(train_indices)
	new_val = train_ds.select(val_indices)
	return new_train, new_val


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
	preds = torch.argmax(logits, dim=-1)
	correct = (preds == labels).sum().item()
	total = labels.numel()
	return correct / max(1, total)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Accelerate training for Transformer models (BERT, RoBERTa, DistilBERT, etc.) with HF Native"
	)
	parser.add_argument("--dataset", type=str, default="atis",
						choices=["atis", "hwu64", "snips", "clinc_oos", "project"],
						help="Which dataset loader to use")
	parser.add_argument("--clinc-version", type=str, default="plus",
						choices=["small", "plus", "imbalanced"],
						help="Version for clinc_oos dataset")
	parser.add_argument("--dataset-dir", type=str, default=
		"/Users/wangshibo/Documents/Academic/Course/6405Project/Intent_Recognition_Exp/data/raw",
		help="Base directory for raw datasets")
	parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH,
						help="Base encoder model path or HF name (supports BERT, RoBERTa, DistilBERT, DistilRoBERTa, etc.)")
	parser.add_argument("--tokenizer-path", type=str, default=DEFAULT_TOKENIZER_PATH,
						help="Tokenizer path or HF name (auto-detected from model if not specified)")
	parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
						help="Output directory for checkpoints and state")
	parser.add_argument("--epochs", type=int, default=3)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--max-length", type=int, default=128)
	
	# Learning rate arguments
	parser.add_argument("--head-lr", type=float, default=3e-4,
						help="Learning rate for classification head")
	parser.add_argument("--encoder-lr", type=float, default=3e-5,
						help="Learning rate for encoder (only used if --train-encoder is set)")
	parser.add_argument("--train-encoder", action="store_true",
						help="Enable training encoder (BERT body). If not set, only train head.")
	
	parser.add_argument("--weight-decay", type=float, default=0.01)
	parser.add_argument("--warmup-ratio", type=float, default=0.06)
	parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction of train used for validation")
	parser.add_argument("--mixed-precision", type=str, default="no",
						choices=["no", "fp16", "bf16"], help="Accelerate mixed precision mode")
	parser.add_argument("--grad-accum-steps", type=int, default=1)
	parser.add_argument("--log-interval", type=int, default=50)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--resume-from", type=str, default="",
						help="Path to a directory containing accelerate state to resume from")
	parser.add_argument("--save-best-only", action="store_true", help="Only save best head checkpoint")
	
	# Wandb arguments
	parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
	parser.add_argument("--wandb-api-key", type=str, default=None, 
						help="W&B API key (if not set, will use WANDB_API_KEY env var or saved login)")
	parser.add_argument("--wandb-project", type=str, default="intent-recognition-bert-hf-native",
						help="W&B project name")
	parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity (username or team)")
	parser.add_argument("--wandb-run-name", type=str, default=None,
						help="W&B run name (auto-generated if not specified)")
	parser.add_argument("--wandb-tags", type=str, default="", help="Comma-separated W&B tags")
	
	return parser.parse_args()


def main():
	args = parse_args()

	# Initialize wandb availability flag
	use_wandb = args.use_wandb and WANDB_AVAILABLE
	if args.use_wandb and not WANDB_AVAILABLE:
		print("--use-wandb specified but wandb not available. Continuing without wandb.")

	accelerator = Accelerator(
		gradient_accumulation_steps=args.grad_accum_steps,
		mixed_precision=args.mixed_precision,
		log_with=None,
		project_dir=args.output_dir,
	)

	# Set global seed for reproducibility across processes
	set_seed(args.seed)

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	# Load dataset
	loader_fn = get_loader_by_name(args.dataset)
	if args.dataset == "clinc_oos":
		raw_dataset, stats = loader_fn(data_dir=args.dataset_dir, version=args.clinc_version)
	else:
		raw_dataset, stats = loader_fn(data_dir=args.dataset_dir)

	# Work on the train split and carve out a validation subset
	assert "train" in raw_dataset, "Expected a 'train' split in dataset"
	base_train = raw_dataset["train"]
	train_split, val_split = stratified_split(base_train, val_ratio=args.val_ratio, seed=args.seed)

	# Build tokenizer and dataset objects
	# Use AutoTokenizer to automatically detect the correct tokenizer class
	tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
	
	# Detect model type from config
	from transformers import AutoConfig
	model_config = AutoConfig.from_pretrained(args.model_path)
	model_type = model_config.model_type.lower()
	accelerator.print(f"Detected model type: {model_type}")

	train_ds = BertTextDataset(train_split, tokenizer, max_length=args.max_length)
	val_ds = BertTextDataset(val_split, tokenizer, max_length=args.max_length) if val_split is not None else None

	collator = DynamicPaddingCollator(tokenizer)

	train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
	val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator) if val_ds else None

	num_labels = int(len(set(train_split["label"])))
	
	# Initialize wandb on main process only
	if use_wandb and accelerator.is_main_process:
		# Login with API key if provided
		if args.wandb_api_key:
			try:
				wandb.login(key=args.wandb_api_key)
				accelerator.print("Logged in to W&B with provided API key")
			except Exception as e:
				accelerator.print(f"W&B login failed: {e}")
		
		# Auto-generate run name if not specified
		run_name = args.wandb_run_name
		if run_name is None:
			run_name = f"{args.dataset}_transformer_hf_native_head_lr{args.head_lr}"
		
		# Parse tags
		tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
		tags.extend([args.dataset, "transformer", "hf-native", "classifier-head"])
		
		wandb.init(
			project=args.wandb_project,
			entity=args.wandb_entity,
			name=run_name,
			tags=tags,
			config={
				"dataset": args.dataset,
				"model_path": args.model_path,
				"num_labels": num_labels,
				"epochs": args.epochs,
				"batch_size": args.batch_size,
				"max_length": args.max_length,
				"head_learning_rate": args.head_lr,
				"encoder_learning_rate": args.encoder_lr if args.train_encoder else None,
				"train_encoder": args.train_encoder,
				"weight_decay": args.weight_decay,
				"warmup_ratio": args.warmup_ratio,
				"val_ratio": args.val_ratio,
				"mixed_precision": args.mixed_precision,
				"grad_accum_steps": args.grad_accum_steps,
				"max_grad_norm": 1.0,
				"freeze_encoder": not args.train_encoder,
				"seed": args.seed,
			},
		)
		accelerator.print("Weights & Biases initialized")
	
	# Build model using HuggingFace AutoModelForSequenceClassification
	# This automatically detects and loads the correct model class (BERT, RoBERTa, DistilBERT, etc.)
	model = AutoModelForSequenceClassification.from_pretrained(
		args.model_path,
		num_labels=num_labels,
		problem_type="single_label_classification",
	)
	
	# Conditionally freeze encoder parameters based on --train-encoder flag
	if not args.train_encoder:
		# Freeze encoder, only train classification head
		for name, param in model.named_parameters():
			if "classifier" not in name:
				param.requires_grad = False
		accelerator.print("Encoder frozen. Only training classification head.")
	else:
		# Train both encoder and head
		accelerator.print("Encoder unfrozen. Training both encoder and classification head.")
	
	# Print trainable parameter statistics
	total_params = sum(p.numel() for p in model.parameters())
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	accelerator.print(f"\n{'='*60}")
	accelerator.print(f"Model Configuration:")
	accelerator.print(f"  Model type: {model.__class__.__name__} ({model_type.upper()})")
	accelerator.print(f"  Total parameters: {total_params:,}")
	accelerator.print(f"  Trainable parameters: {trainable_params:,}")
	accelerator.print(f"  Frozen parameters: {total_params - trainable_params:,}")
	accelerator.print(f"  Trainable ratio: {100.0 * trainable_params / total_params:.2f}%")
	accelerator.print(f"{'='*60}\n")

	# Optimizer with differential learning rates
	# Separate parameters into head and encoder groups
	head_params = [p for n, p in model.named_parameters() 
	               if p.requires_grad and "classifier" in n]
	encoder_params = [p for n, p in model.named_parameters() 
	                  if p.requires_grad and "classifier" not in n]
	
	optimizer_grouped_parameters = []
	
	# Always add head parameters
	if head_params:
		optimizer_grouped_parameters.append({
			"params": head_params,
			"lr": args.head_lr,
			"weight_decay": args.weight_decay,
		})
		accelerator.print(f"Head parameters: {sum(p.numel() for p in head_params):,} (lr={args.head_lr:.2e})")
	
	# Add encoder parameters only if training encoder
	if encoder_params and args.train_encoder:
		optimizer_grouped_parameters.append({
			"params": encoder_params,
			"lr": args.encoder_lr,
			"weight_decay": args.weight_decay,
		})
		accelerator.print(f"Encoder parameters: {sum(p.numel() for p in encoder_params):,} (lr={args.encoder_lr:.2e})")
	
	optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

	# Prepare with accelerator
	model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

	# Scheduler after prepare (use total steps post-accumulation)
	num_update_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
	max_train_steps = args.epochs * num_update_steps_per_epoch
	num_warmup_steps = int(args.warmup_ratio * max_train_steps)
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=max_train_steps)
	scheduler = accelerator.prepare(scheduler)
	
	# Print training configuration
	accelerator.print(f"{'='*60}")
	accelerator.print(f"Training Configuration:")
	accelerator.print(f"  Dataset: {args.dataset}")
	accelerator.print(f"  Train samples: {len(train_ds)}")
	accelerator.print(f"  Val samples: {len(val_ds) if val_ds else 0}")
	accelerator.print(f"  Num labels: {num_labels}")
	accelerator.print(f"  Training mode: {'Full (Encoder + Head)' if args.train_encoder else 'Head Only'}")
	accelerator.print(f"  Epochs: {args.epochs}")
	accelerator.print(f"  Batch size: {args.batch_size}")
	accelerator.print(f"  Gradient accumulation steps: {args.grad_accum_steps}")
	accelerator.print(f"  Effective batch size: {args.batch_size * args.grad_accum_steps * accelerator.num_processes}")
	accelerator.print(f"  Head learning rate: {args.head_lr:.2e}")
	if args.train_encoder:
		accelerator.print(f"  Encoder learning rate: {args.encoder_lr:.2e}")
	accelerator.print(f"  Weight decay: {args.weight_decay}")
	accelerator.print(f"  Warmup ratio: {args.warmup_ratio}")
	accelerator.print(f"  Warmup steps: {num_warmup_steps}")
	accelerator.print(f"  Total training steps: {max_train_steps}")
	accelerator.print(f"  Max gradient norm: 1.0")
	accelerator.print(f"  Mixed precision: {args.mixed_precision}")
	accelerator.print(f"  Num processes: {accelerator.num_processes}")
	accelerator.print(f"  Output directory: {args.output_dir}")
	accelerator.print(f"{'='*60}\n")

	# Optionally resume
	if args.resume_from:
		resume_dir = Path(args.resume_from)
		if resume_dir.exists():
			accelerator.print(f"Resuming from: {resume_dir}")
			accelerator.load_state(resume_dir)
		else:
			accelerator.print(f"--resume-from path does not exist: {resume_dir}")

	best_val_acc = -1.0
	best_dir = output_dir / "best_model"
	last_dir = output_dir / "last_state"
	
	# Gradient clipping threshold
	max_grad_norm = 1.0

	for epoch in range(1, args.epochs + 1):
		model.train()
		running_loss = 0.0
		running_acc = 0.0
		seen = 0

		# Create progress bar for training
		progress_bar = tqdm(
			train_loader,
			desc=f"Epoch {epoch}/{args.epochs}",
			disable=not accelerator.is_local_main_process,
		)

		for step, batch in enumerate(progress_bar, start=1):
			with accelerator.accumulate(model):
				# BertForSequenceClassification expects 'labels' as keyword argument
				outputs = model(
					input_ids=batch["input_ids"],
					attention_mask=batch["attention_mask"],
					labels=batch["labels"],
				)
				loss = outputs.loss
				logits = outputs.logits

				accelerator.backward(loss)
				# Gradient clipping for training stability
				accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
				optimizer.step()
				scheduler.step()
				optimizer.zero_grad()

			# Metrics (on local process tensors)
			batch_size = batch["labels"].size(0)
			running_loss += loss.detach().float().item() * batch_size
			running_acc += compute_accuracy(logits.detach(), batch["labels"]) * batch_size
			seen += batch_size

			# Update progress bar with current metrics and learning rate
			avg_loss = running_loss / max(1, seen)
			avg_acc = running_acc / max(1, seen)
			current_lr = scheduler.get_last_lr()[0]
			progress_bar.set_postfix({
				"loss": f"{avg_loss:.4f}",
				"acc": f"{avg_acc:.4f}",
				"lr": f"{current_lr:.2e}",
			})
			
			# Log to wandb (every step)
			if use_wandb and accelerator.is_main_process:
				wandb.log({
					"train/step_loss": loss.detach().float().item(),
					"train/step_accuracy": compute_accuracy(logits.detach(), batch["labels"]),
					"train/learning_rate": current_lr,
					"train/epoch": epoch,
					"train/step": (epoch - 1) * len(train_loader) + step,
				})

		# Epoch end: compute train metrics
		train_loss = running_loss / max(1, seen)
		train_acc = running_acc / max(1, seen)

		# Validation
		val_acc = None
		val_loss = None
		if val_loader is not None:
			model.eval()
			v_loss = 0.0
			v_acc = 0.0
			v_seen = 0
			
			# Create progress bar for validation
			val_progress_bar = tqdm(
				val_loader,
				desc=f"Validation",
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
					
					# Update validation progress bar
					val_progress_bar.set_postfix({
						"val_loss": f"{v_loss / max(1, v_seen):.4f}",
						"val_acc": f"{v_acc / max(1, v_seen):.4f}",
					})
			
			val_loss = v_loss / max(1, v_seen)
			val_acc = v_acc / max(1, v_seen)

		accelerator.print(
			f"Epoch {epoch} done | train_loss={train_loss:.4f} train_acc={train_acc:.4f}"
			+ (" " if val_loader is None else f" | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
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

		# Save last state (for resume)
		accelerator.wait_for_everyone()
		# Use a temporary dir to avoid partial writes on multi-process
		last_dir.mkdir(parents=True, exist_ok=True)
		accelerator.save_state(last_dir)

		# Save best model
		if val_loader is not None:
			score = float(val_acc)
		else:
			score = float(train_acc)  # fallback when no val split

		if score > best_val_acc:
			best_val_acc = score
			if accelerator.is_main_process:
				best_dir.mkdir(parents=True, exist_ok=True)
				# unwrap model to access save_pretrained
				unwrapped = accelerator.unwrap_model(model)
				# Use HuggingFace's native save_pretrained
				unwrapped.save_pretrained(best_dir)
				# Also save tokenizer for easy loading
				tokenizer.save_pretrained(best_dir)
				# also record metrics
				with open(best_dir / "metrics.txt", "w", encoding="utf-8") as f:
					f.write(f"epoch={epoch}\ntrain_loss={train_loss}\ntrain_acc={train_acc}\n")
					if val_loader is not None:
						f.write(f"val_loss={val_loss}\nval_acc={val_acc}\n")
				accelerator.print(f"Saved new best model to: {best_dir} (score={score:.4f})")

		# Optionally per-epoch full state snapshot
		if not args.save_best_only:
			snap_dir = output_dir / f"epoch-{epoch}"
			snap_dir.mkdir(parents=True, exist_ok=True)
			accelerator.save_state(snap_dir)
			
			# Keep only the latest 3 epoch checkpoints
			if accelerator.is_main_process:
				existing_epochs = sorted([
					d for d in output_dir.glob("epoch-*")
					if d.is_dir()
				], key=lambda x: int(x.name.split("-")[1]))
				
				if len(existing_epochs) > 3:
					for old_dir in existing_epochs[:-3]:
						shutil.rmtree(old_dir)
						accelerator.print(f"🗑️  Removed old checkpoint: {old_dir.name}")

	accelerator.print("\n" + "=" * 60)
	accelerator.print("Training complete!")
	accelerator.print(f"Best validation score: {best_val_acc:.4f}")
	accelerator.print(f"Best model saved to: {best_dir}")
	accelerator.print("=" * 60)
	
	# Finalize wandb
	if use_wandb and accelerator.is_main_process:
		# Log final summary
		wandb.run.summary["final_best_val_accuracy"] = best_val_acc
		wandb.run.summary["total_epochs"] = args.epochs
		wandb.run.summary["best_model_path"] = str(best_dir)
		
		# Save best model as artifact (optional but recommended)
		if best_dir.exists():
			try:
				artifact = wandb.Artifact(
					name=f"{args.dataset}_transformer_hf_native_best",
					type="model",
					description=f"Best {model_type.upper()} model (HF native) for {args.dataset} dataset",
					metadata={
						"val_accuracy": best_val_acc,
						"dataset": args.dataset,
						"freeze_encoder": not args.train_encoder,
						"model_type": model.__class__.__name__,
						"model_architecture": model_type,
					}
				)
				artifact.add_dir(str(best_dir))
				wandb.log_artifact(artifact)
				accelerator.print("✅ Best model uploaded to W&B as artifact")
			except Exception as e:
				accelerator.print(f"⚠️ Failed to upload model artifact: {e}")
		
		wandb.finish()
		accelerator.print("✅ W&B run finished")


if __name__ == "__main__":
	main()
