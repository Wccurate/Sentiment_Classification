from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, set_seed
from accelerate import Accelerator
from tqdm.auto import tqdm

from models.common.common_bert_wrapper import BertWithClassifierImproved
from data.loaders.bert_dataloaders import BertTextDataset, DynamicPaddingCollator
from datasets import DatasetDict

# Dataset loaders
from data.loaders.load_raw_data import (
	load_raw_atis,
	load_raw_hwu64,
	load_raw_snips,
	load_raw_clinc_oos,
)


DEFAULT_MODEL_PATH = \
	"/Users/wangshibo/Documents/Academic/Course/6405Project/Intent_Recognition_Exp/models/huggingface_models/bert_base_cased_google_bert"
DEFAULT_TOKENIZER_PATH = DEFAULT_MODEL_PATH
DEFAULT_OUTPUT_DIR = "outputs/checkpoints/bert_accelerate_train"


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
	parser = argparse.ArgumentParser(description="Accelerate training for BertWithClassifier")
	parser.add_argument("--dataset", type=str, default="atis",
						choices=["atis", "hwu64", "snips", "clinc_oos"],
						help="Which dataset loader to use")
	parser.add_argument("--clinc-version", type=str, default="plus",
						choices=["small", "plus", "imbalanced"],
						help="Version for clinc_oos dataset")
	parser.add_argument("--dataset-dir", type=str, default=
		"/Users/wangshibo/Documents/Academic/Course/6405Project/Intent_Recognition_Exp/data/raw",
		help="Base directory for raw datasets")
	parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH,
						help="Base encoder model path or HF name")
	parser.add_argument("--tokenizer-path", type=str, default=DEFAULT_TOKENIZER_PATH,
						help="Tokenizer path or HF name")
	parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
						help="Output directory for checkpoints and state")
	parser.add_argument("--epochs", type=int, default=3)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--max-length", type=int, default=128)
	parser.add_argument("--lr", type=float, default=3e-5, help="Default learning rate (deprecated, use --encoder-lr and --head-lr)")
	parser.add_argument("--encoder-lr", type=float, default=2e-5, help="Learning rate for BERT encoder")
	parser.add_argument("--head-lr", type=float, default=1e-4, help="Learning rate for classifier head")
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
	return parser.parse_args()


def main():
	args = parse_args()

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
		raw_dataset, stats, textlabel2label, label2textlabel = loader_fn(data_dir=args.dataset_dir, version=args.clinc_version,return_dicts=True)
	else:
		raw_dataset, stats, textlabel2label, label2textlabel = loader_fn(data_dir=args.dataset_dir, return_dicts=True)

	# Work on the train split and carve out a validation subset
	assert "train" in raw_dataset, "Expected a 'train' split in dataset"
	base_train = raw_dataset["train"]
	train_split, val_split = stratified_split(base_train, val_ratio=args.val_ratio, seed=args.seed)

	# Build tokenizer and dataset objects
	from transformers import BertTokenizer
	tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)

	train_ds = BertTextDataset(train_split, tokenizer, max_length=args.max_length)
	val_ds = BertTextDataset(val_split, tokenizer, max_length=args.max_length) if val_split is not None else None

	collator = DynamicPaddingCollator(tokenizer)

	train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
	val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator) if val_ds else None

	# Build model
	num_labels = int(len(set(train_split["label"])))
	if args.dataset == "clinc_oos":
		oos_label_num=textlabel2label["out of scope"]
		w = torch.ones(num_labels)
		w[oos_label_num] = 3.0
		model=BertWithClassifierImproved(
			model_name=args.model_path,
			num_labels=num_labels,
			dropout=0.1,
			freeze_encoder=False,
			pooling_mode="mean",
			scalar_mix_last_k=4,
			head_type="mlp",
			ms_dropout=4,
			class_weight_mode="manual", 
    		class_weights=w             
		)
	elif args.dataset == "hwu64":
		# 1) Count samples per class in train_split (length=num_labels)
		y_train = np.array(train_split["label"], dtype=np.int64)
		counts = np.bincount(y_train, minlength=num_labels)

		# 2) Recommended: class-balanced weighting (effective number) with beta≈0.999
		#    Switch class_weight_mode to "auto_inv_freq" for inverse-frequency weights
		model = BertWithClassifierImproved(
			model_name=args.model_path,
			num_labels=num_labels,
			dropout=0.1,
			freeze_encoder=False,
			pooling_mode="mean",
			scalar_mix_last_k=4,
			head_type="mlp",
			ms_dropout=4,
			class_weight_mode="auto_effective",   # "auto_effective" | "auto_inv_freq" | "manual" | "none"
			class_counts=counts,                  # Needed for automatic weighting
			cb_beta=0.999,                        # Effective-number beta (closer to 1 => smoother)
			# Optional: combine with label smoothing/focal if necessary
			# label_smoothing=0.05,
			# focal_gamma=0.0,  # Enable 1.0~2.0 later if needed
		)
	else:
		model = BertWithClassifierImproved(
			model_name=args.model_path,
			num_labels=num_labels,
			dropout=0.1,
			freeze_encoder=False,
			pooling_mode="mean",
			scalar_mix_last_k=4,
			head_type="mlp",
			ms_dropout=4,
		)

	# Optimizer with different learning rates for encoder and head
	# Encoder (BERT): lower learning rate for fine-tuning
	# Classifier head: higher learning rate for faster adaptation
	optimizer = torch.optim.AdamW([
		{'params': model.encoder.parameters(), 'lr': args.encoder_lr},
		{'params': model.classifier.parameters(), 'lr': args.head_lr},
		{'params': model.dropout.parameters(), 'lr': args.head_lr},
	], weight_decay=args.weight_decay)
	
	accelerator.print(f"Optimizer configured with layered learning rates:")
	accelerator.print(f"   - Encoder LR: {args.encoder_lr:.2e}")
	accelerator.print(f"   - Head LR: {args.head_lr:.2e}")

	# Prepare with accelerator
	model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

	# Scheduler after prepare (use total steps post-accumulation)
	num_update_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
	max_train_steps = args.epochs * num_update_steps_per_epoch
	num_warmup_steps = int(args.warmup_ratio * max_train_steps)
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=max_train_steps)
	scheduler = accelerator.prepare(scheduler)
	
	accelerator.print(f"📊 Training configuration:")
	accelerator.print(f"   - Total epochs: {args.epochs}")
	accelerator.print(f"   - Steps per epoch: {num_update_steps_per_epoch}")
	accelerator.print(f"   - Total training steps: {max_train_steps}")
	accelerator.print(f"   - Warmup steps: {num_warmup_steps} ({args.warmup_ratio:.1%})")

	# Optionally resume
	if args.resume_from:
		resume_dir = Path(args.resume_from)
		if resume_dir.exists():
			accelerator.print(f"Resuming from: {resume_dir}")
			accelerator.load_state(resume_dir)
		else:
			accelerator.print(f"--resume-from path does not exist: {resume_dir}")

	best_val_acc = -1.0
	best_dir = output_dir / "best_head"
	last_dir = output_dir / "last_state"
	
	# Keep track of recent epoch checkpoints (max 3)
	recent_epoch_dirs = []

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
				loss = outputs["loss"]
				logits = outputs["logits"]

				accelerator.backward(loss)
				optimizer.step()
				scheduler.step()
				optimizer.zero_grad()

			# Metrics (on local process tensors)
			batch_size = batch["labels"].size(0)
			running_loss += loss.detach().float().item() * batch_size
			running_acc += compute_accuracy(logits.detach(), batch["labels"]) * batch_size
			seen += batch_size

			# Update progress bar with current metrics and learning rates
			avg_loss = running_loss / max(1, seen)
			avg_acc = running_acc / max(1, seen)
			# Get learning rates for both parameter groups
			lr_values = scheduler.get_last_lr()
			encoder_lr = lr_values[0] if len(lr_values) > 0 else 0
			head_lr = lr_values[1] if len(lr_values) > 1 else 0
			progress_bar.set_postfix({
				"loss": f"{avg_loss:.4f}",
				"acc": f"{avg_acc:.4f}",
				"enc_lr": f"{encoder_lr:.2e}",
				"head_lr": f"{head_lr:.2e}",
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
					loss = outputs["loss"]
					logits = outputs["logits"]
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

		# Save last state (for resume)
		accelerator.wait_for_everyone()
		# Use a temporary dir to avoid partial writes on multi-process
		last_dir.mkdir(parents=True, exist_ok=True)
		accelerator.save_state(last_dir)

		# Save best head only
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
				unwrapped.save_pretrained(best_dir)
				# also record metrics
				with open(best_dir / "metrics.txt", "w", encoding="utf-8") as f:
					f.write(f"epoch={epoch}\ntrain_loss={train_loss}\ntrain_acc={train_acc}\n")
					if val_loader is not None:
						f.write(f"val_loss={val_loss}\nval_acc={val_acc}\n")
				accelerator.print(f"Saved new best head to: {best_dir} (score={score:.4f})")

		# Optionally per-epoch full state snapshot (keep only recent 3 epochs)
		if not args.save_best_only:
			snap_dir = output_dir / f"epoch-{epoch}"
			snap_dir.mkdir(parents=True, exist_ok=True)
			accelerator.save_state(snap_dir)
			
			# Track this checkpoint
			recent_epoch_dirs.append(snap_dir)
			
			# Remove old checkpoints if we have more than 3
			if len(recent_epoch_dirs) > 3:
				old_dir = recent_epoch_dirs.pop(0)
				if old_dir.exists() and accelerator.is_main_process:
					import shutil
					shutil.rmtree(old_dir)
					accelerator.print(f"Removed old checkpoint: {old_dir.name}")

	accelerator.print("Training complete.")
	accelerator.print(f"Best validation score: {best_val_acc:.4f}")
	accelerator.print(f"Saved checkpoints:")
	accelerator.print(f"   - Best model: {best_dir}")
	if not args.save_best_only:
		accelerator.print(f"   - Recent epochs: {[d.name for d in recent_epoch_dirs]}")


if __name__ == "__main__":
	main()
