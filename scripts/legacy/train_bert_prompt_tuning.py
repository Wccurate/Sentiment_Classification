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
	get_cosine_schedule_with_warmup,
	set_seed,
)
from accelerate import Accelerator
from tqdm.auto import tqdm
from peft import (
	PromptTuningConfig,
	PromptTuningInit,
	TaskType,
	get_peft_model,
	PeftModel,
)

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
DEFAULT_OUTPUT_DIR = "outputs/checkpoints/bert_prompt_tuning_train"


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


def print_trainable_parameters(model, accelerator):
	"""
	Prints the number of trainable parameters in the model.
	"""
	trainable_params = 0
	all_param = 0
	for _, param in model.named_parameters():
		all_param += param.numel()
		if param.requires_grad:
			trainable_params += param.numel()
	
	accelerator.print(f"\n{'='*60}")
	accelerator.print(f"Model Configuration:")
	accelerator.print(f"  Model type: BERT with Prompt Tuning (PEFT)")
	accelerator.print(f"  Total parameters: {all_param:,}")
	accelerator.print(f"  Trainable parameters: {trainable_params:,}")
	accelerator.print(f"  Frozen parameters: {all_param - trainable_params:,}")
	accelerator.print(f"  Trainable ratio: {100.0 * trainable_params / all_param:.4f}%")
	accelerator.print(f"{'='*60}\n")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Accelerate training for BERT with Prompt Tuning (PEFT)"
	)
	
	# Dataset arguments
	parser.add_argument("--dataset", type=str, default="atis",
						choices=["atis", "hwu64", "snips", "clinc_oos", "project"],
						help="Which dataset loader to use")
	parser.add_argument("--clinc-version", type=str, default="plus",
						choices=["small", "plus", "imbalanced"],
						help="Version for clinc_oos dataset")
	parser.add_argument("--dataset-dir", type=str, default=
		"/Users/wangshibo/Documents/Academic/Course/6405Project/Intent_Recognition_Exp/data/raw",
		help="Base directory for raw datasets")
	
	# Model arguments
	parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH,
						help="Base encoder model path or HF name")
	parser.add_argument("--tokenizer-path", type=str, default=DEFAULT_TOKENIZER_PATH,
						help="Tokenizer path or HF name")
	parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
						help="Output directory for checkpoints and state")
	
	# Prompt Tuning specific arguments
	parser.add_argument("--num-virtual-tokens", type=int, default=20,
						help="Number of virtual tokens for prompt tuning (typically 8-100)")
	parser.add_argument("--prompt-tuning-init", type=str, default="random",
						choices=["random", "text"],
						help="Initialization method for prompt embeddings")
	parser.add_argument("--prompt-tuning-init-text", type=str, default=None,
						help="Initialization text for prompt (if using text init)")
	parser.add_argument("--token-dim", type=int, default=None,
						help="Token embedding dimension (auto-detected if not specified)")
	
	# Training arguments
	parser.add_argument("--epochs", type=int, default=20,
						help="Number of training epochs (prompt tuning typically needs more)")
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--max-length", type=int, default=128)
	parser.add_argument("--val-ratio", type=float, default=0.1, 
						help="Fraction of train used for validation")
	
	# Optimizer arguments with differential learning rates
	parser.add_argument("--prompt-lr", type=float, default=3e-2,
						help="Learning rate for prompt embeddings (typically 1e-2 to 5e-2)")
	parser.add_argument("--classifier-lr", type=float, default=1e-3,
						help="Learning rate for classifier head (typically 1e-4 to 1e-3)")
	parser.add_argument("--weight-decay", type=float, default=0.01)
	parser.add_argument("--warmup-ratio", type=float, default=0.1,
						help="Warmup ratio (prompt tuning benefits from longer warmup)")
	parser.add_argument("--scheduler-type", type=str, default="cosine",
						choices=["linear", "cosine"],
						help="Learning rate scheduler type (cosine often works better)")
	
	# Regularization arguments
	parser.add_argument("--mixed-precision", type=str, default="no",
						choices=["no", "fp16", "bf16"], 
						help="Accelerate mixed precision mode")
	parser.add_argument("--grad-accum-steps", type=int, default=1)
	parser.add_argument("--max-grad-norm", type=float, default=1.0,
						help="Max gradient norm for clipping")
	parser.add_argument("--dropout", type=float, default=0.1,
						help="Dropout rate for classifier head")
	
	# Other training arguments
	parser.add_argument("--log-interval", type=int, default=50)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--resume-from", type=str, default="",
						help="Path to a directory containing accelerate state to resume from")
	parser.add_argument("--save-best-only", action="store_true", 
						help="Only save best checkpoint")
	parser.add_argument("--early-stopping-patience", type=int, default=5,
						help="Early stopping patience (0 to disable)")
	
	# Wandb arguments
	parser.add_argument("--use-wandb", action="store_true", 
						help="Enable Weights & Biases logging")
	parser.add_argument("--wandb-api-key", type=str, default=None, 
						help="W&B API key (if not set, will use WANDB_API_KEY env var or saved login)")
	parser.add_argument("--wandb-project", type=str, 
						default="intent-recognition-bert-prompt-tuning",
						help="W&B project name")
	parser.add_argument("--wandb-entity", type=str, default=None, 
						help="W&B entity (username or team)")
	parser.add_argument("--wandb-run-name", type=str, default=None,
						help="W&B run name (auto-generated if not specified)")
	parser.add_argument("--wandb-tags", type=str, default="", 
						help="Comma-separated W&B tags")
	
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
	tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

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
			run_name = f"{args.dataset}_prompt_tuning_v{args.num_virtual_tokens}_plr{args.prompt_lr}"
		
		# Parse tags
		tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
		tags.extend([args.dataset, "bert", "prompt-tuning", "peft"])
		
		wandb.init(
			project=args.wandb_project,
			entity=args.wandb_entity,
			name=run_name,
			tags=tags,
			config={
				"dataset": args.dataset,
				"model_path": args.model_path,
				"num_labels": num_labels,
				"num_virtual_tokens": args.num_virtual_tokens,
				"prompt_tuning_init": args.prompt_tuning_init,
				"epochs": args.epochs,
				"batch_size": args.batch_size,
				"max_length": args.max_length,
				"prompt_learning_rate": args.prompt_lr,
				"classifier_learning_rate": args.classifier_lr,
				"weight_decay": args.weight_decay,
				"warmup_ratio": args.warmup_ratio,
				"scheduler_type": args.scheduler_type,
				"val_ratio": args.val_ratio,
				"mixed_precision": args.mixed_precision,
				"grad_accum_steps": args.grad_accum_steps,
				"max_grad_norm": args.max_grad_norm,
				"dropout": args.dropout,
				"early_stopping_patience": args.early_stopping_patience,
				"seed": args.seed,
			},
		)
		accelerator.print("Weights & Biases initialized")
	
	# Build base model
	base_model = AutoModelForSequenceClassification.from_pretrained(
		args.model_path,
		num_labels=num_labels,
		problem_type="single_label_classification",
		hidden_dropout_prob=args.dropout,
		attention_probs_dropout_prob=args.dropout,
	)
	
	# Configure Prompt Tuning
	peft_config = PromptTuningConfig(
		task_type=TaskType.SEQ_CLS,
		prompt_tuning_init=(
			PromptTuningInit.TEXT if args.prompt_tuning_init == "text" 
			else PromptTuningInit.RANDOM
		),
		num_virtual_tokens=args.num_virtual_tokens,
		prompt_tuning_init_text=args.prompt_tuning_init_text,
		tokenizer_name_or_path=args.tokenizer_path,
	)
	
	# Apply PEFT to create prompt tuning model
	model = get_peft_model(base_model, peft_config)
	
	# Print trainable parameters
	print_trainable_parameters(model, accelerator)
	
	# Separate parameters for differential learning rates
	# Prompt embeddings get higher learning rate, classifier gets lower
	prompt_params = []
	classifier_params = []
	other_params = []
	
	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue
		if "prompt_embeddings" in name or "prompt_encoder" in name:
			prompt_params.append(param)
		elif "classifier" in name or "score" in name:
			classifier_params.append(param)
		else:
			other_params.append(param)
	
	# Build optimizer with differential learning rates
	optimizer_grouped_parameters = []
	
	if prompt_params:
		optimizer_grouped_parameters.append({
			"params": prompt_params,
			"lr": args.prompt_lr,
			"weight_decay": 0.0,  # No weight decay for prompt embeddings
		})
		accelerator.print(f"Prompt parameters: {sum(p.numel() for p in prompt_params):,} (lr={args.prompt_lr})")
	
	if classifier_params:
		optimizer_grouped_parameters.append({
			"params": classifier_params,
			"lr": args.classifier_lr,
			"weight_decay": args.weight_decay,
		})
		accelerator.print(f"Classifier parameters: {sum(p.numel() for p in classifier_params):,} (lr={args.classifier_lr})")
	
	if other_params:
		optimizer_grouped_parameters.append({
			"params": other_params,
			"lr": args.classifier_lr,
			"weight_decay": args.weight_decay,
		})
		accelerator.print(f"Other trainable parameters: {sum(p.numel() for p in other_params):,} (lr={args.classifier_lr})")
	
	optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

	# Prepare with accelerator
	model, optimizer, train_loader, val_loader = accelerator.prepare(
		model, optimizer, train_loader, val_loader
	)

	# Scheduler after prepare
	num_update_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
	max_train_steps = args.epochs * num_update_steps_per_epoch
	num_warmup_steps = int(args.warmup_ratio * max_train_steps)
	
	if args.scheduler_type == "cosine":
		scheduler = get_cosine_schedule_with_warmup(
			optimizer, 
			num_warmup_steps=num_warmup_steps, 
			num_training_steps=max_train_steps
		)
	else:
		scheduler = get_linear_schedule_with_warmup(
			optimizer, 
			num_warmup_steps=num_warmup_steps, 
			num_training_steps=max_train_steps
		)
	scheduler = accelerator.prepare(scheduler)
	
	# Print training configuration
	accelerator.print(f"\n{'='*60}")
	accelerator.print(f"Training Configuration:")
	accelerator.print(f"  Dataset: {args.dataset}")
	accelerator.print(f"  Train samples: {len(train_ds)}")
	accelerator.print(f"  Val samples: {len(val_ds) if val_ds else 0}")
	accelerator.print(f"  Num labels: {num_labels}")
	accelerator.print(f"  Num virtual tokens: {args.num_virtual_tokens}")
	accelerator.print(f"  Prompt init method: {args.prompt_tuning_init}")
	accelerator.print(f"  Epochs: {args.epochs}")
	accelerator.print(f"  Batch size: {args.batch_size}")
	accelerator.print(f"  Gradient accumulation steps: {args.grad_accum_steps}")
	accelerator.print(f"  Effective batch size: {args.batch_size * args.grad_accum_steps * accelerator.num_processes}")
	accelerator.print(f"  Prompt learning rate: {args.prompt_lr:.2e}")
	accelerator.print(f"  Classifier learning rate: {args.classifier_lr:.2e}")
	accelerator.print(f"  Weight decay: {args.weight_decay}")
	accelerator.print(f"  Scheduler type: {args.scheduler_type}")
	accelerator.print(f"  Warmup ratio: {args.warmup_ratio}")
	accelerator.print(f"  Warmup steps: {num_warmup_steps}")
	accelerator.print(f"  Total training steps: {max_train_steps}")
	accelerator.print(f"  Max gradient norm: {args.max_grad_norm}")
	accelerator.print(f"  Dropout: {args.dropout}")
	accelerator.print(f"  Mixed precision: {args.mixed_precision}")
	accelerator.print(f"  Num processes: {accelerator.num_processes}")
	accelerator.print(f"  Early stopping patience: {args.early_stopping_patience if args.early_stopping_patience > 0 else 'Disabled'}")
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
	best_epoch = 0
	epochs_without_improvement = 0
	best_dir = output_dir / "best_model"
	last_dir = output_dir / "last_state"

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
				outputs = model(
					input_ids=batch["input_ids"],
					attention_mask=batch["attention_mask"],
					labels=batch["labels"],
				)
				loss = outputs.loss
				logits = outputs.logits

				accelerator.backward(loss)
				# Gradient clipping for training stability
				accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
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
			# Get prompt lr (first param group)
			current_prompt_lr = scheduler.get_last_lr()[0]
			progress_bar.set_postfix({
				"loss": f"{avg_loss:.4f}",
				"acc": f"{avg_acc:.4f}",
				"prompt_lr": f"{current_prompt_lr:.2e}",
			})
			
			# Log to wandb (every step)
			if use_wandb and accelerator.is_main_process:
				log_dict = {
					"train/step_loss": loss.detach().float().item(),
					"train/step_accuracy": compute_accuracy(logits.detach(), batch["labels"]),
					"train/prompt_learning_rate": current_prompt_lr,
					"train/epoch": epoch,
					"train/step": (epoch - 1) * len(train_loader) + step,
				}
				# Log all learning rates
				for i, lr in enumerate(scheduler.get_last_lr()):
					log_dict[f"train/lr_group_{i}"] = lr
				wandb.log(log_dict)

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
		last_dir.mkdir(parents=True, exist_ok=True)
		accelerator.save_state(last_dir)

		# Save best model
		if val_loader is not None:
			score = float(val_acc)
		else:
			score = float(train_acc)  # fallback when no val split

		if score > best_val_acc:
			best_val_acc = score
			best_epoch = epoch
			epochs_without_improvement = 0
			
			if accelerator.is_main_process:
				best_dir.mkdir(parents=True, exist_ok=True)
				# Unwrap model to access PEFT methods
				unwrapped = accelerator.unwrap_model(model)
				
				# Save PEFT adapter (only prompt embeddings + config)
				unwrapped.save_pretrained(best_dir)
				
				# Also save tokenizer
				tokenizer.save_pretrained(best_dir)
				
				# Save metrics
				with open(best_dir / "metrics.txt", "w", encoding="utf-8") as f:
					f.write(f"epoch={epoch}\n")
					f.write(f"train_loss={train_loss:.6f}\n")
					f.write(f"train_acc={train_acc:.6f}\n")
					if val_loader is not None:
						f.write(f"val_loss={val_loss:.6f}\n")
						f.write(f"val_acc={val_acc:.6f}\n")
					f.write(f"num_virtual_tokens={args.num_virtual_tokens}\n")
					f.write(f"prompt_lr={args.prompt_lr}\n")
					f.write(f"classifier_lr={args.classifier_lr}\n")
				
				accelerator.print(f"Saved new best model to: {best_dir} (score={score:.4f})")
		else:
			epochs_without_improvement += 1
		
		# Early stopping check
		if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
			accelerator.print(f"\nEarly stopping triggered after {epoch} epochs")
			accelerator.print(f"   No improvement for {epochs_without_improvement} epochs")
			accelerator.print(f"   Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
			break

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
	accelerator.print(f"Best validation score: {best_val_acc:.4f} (epoch {best_epoch})")
	accelerator.print(f"Best model saved to: {best_dir}")
	accelerator.print("=" * 60)
	
	# Print loading instructions
	accelerator.print("\n" + "=" * 60)
	accelerator.print("Loading Instructions:")
	accelerator.print("To load the trained model:")
	accelerator.print("  from transformers import AutoModelForSequenceClassification")
	accelerator.print("  from peft import PeftModel")
	accelerator.print(f"  base_model = AutoModelForSequenceClassification.from_pretrained('{args.model_path}')")
	accelerator.print(f"  model = PeftModel.from_pretrained(base_model, '{best_dir}')")
	accelerator.print("=" * 60 + "\n")
	
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
		
		# Save best model as artifact
		if best_dir.exists():
			try:
				artifact = wandb.Artifact(
					name=f"{args.dataset}_bert_prompt_tuning_best",
					type="model",
					description=f"Best BERT Prompt Tuning model for {args.dataset} dataset",
					metadata={
						"val_accuracy": best_val_acc,
						"best_epoch": best_epoch,
						"dataset": args.dataset,
						"num_virtual_tokens": args.num_virtual_tokens,
						"prompt_tuning_init": args.prompt_tuning_init,
						"parameter_efficiency": 100.0 * trainable_params / total_params,
						"model_type": "BERT-PromptTuning-PEFT",
					}
				)
				artifact.add_dir(str(best_dir))
				wandb.log_artifact(artifact)
				accelerator.print("Best model uploaded to W&B as artifact")
			except Exception as e:
				accelerator.print(f"Failed to upload model artifact: {e}")
		
		wandb.finish()
		accelerator.print("W&B run finished")


if __name__ == "__main__":
	main()
