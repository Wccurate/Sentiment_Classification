from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, set_seed
from tqdm.auto import tqdm

from models.common.common_bert_wrapper import BertWithClassifier,BertWithClassifierImproved
from data.loaders.bert_dataloaders import BertTextDataset, DynamicPaddingCollator

# Dataset loaders
from data.loaders.load_raw_data import (
	load_raw_atis,
	load_raw_hwu64,
	load_raw_snips,
	load_raw_clinc_oos,
)


def get_loader_by_name(name: str):
	"""Get dataset loader function by name."""
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


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Detailed evaluation with metadata and visualization")
	parser.add_argument("--dataset", type=str, required=True,
						choices=["atis", "hwu64", "snips", "clinc_oos"],
						help="Which dataset to evaluate on")
	parser.add_argument("--clinc-version", type=str, default="plus",
						choices=["small", "plus", "imbalanced"],
						help="Version for clinc_oos dataset")
	parser.add_argument("--dataset-dir", type=str, 
						default=str(Path(__file__).resolve().parent.parent.parent / "data" / "raw"),
						help="Base directory for raw datasets")
	parser.add_argument("--model-path", type=str, required=True,
						help="Path to trained model checkpoint")
	parser.add_argument("--encoder-path", type=str, required=True,
						help="Path to base BERT encoder")
	parser.add_argument("--tokenizer-path", type=str, default="",
						help="Tokenizer path (defaults to encoder-path)")
	parser.add_argument("--output-dir", type=str, required=True,
						help="Output directory for results")
	parser.add_argument("--method", type=str, default="BERT-Base",
						help="Method name for metadata")
	parser.add_argument("--method-detail", type=str, default="",
						help="Additional method details")
	parser.add_argument("--batch-size", type=int, default=32,
						help="Batch size for batched evaluation")
	parser.add_argument("--max-length", type=int, default=128,
						help="Maximum sequence length")
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
						help="Device to run evaluation on")
	parser.add_argument("--seed", type=int, default=42,
						help="Random seed")
	parser.add_argument("--warmup-steps", type=int, default=10,
						help="Number of warmup steps before timing")
	return parser.parse_args()


def single_sample_inference(
	model: BertWithClassifierImproved,
	test_split,
	tokenizer: BertTokenizer,
	device: torch.device,
	max_length: int,
	warmup_steps: int = 10,
) -> Dict:
	"""
	Run single-sample inference to measure per-sample latency.
	
	Returns:
		Dictionary with predictions, latencies, and metadata
	"""
	model.eval()
	model.to(device)
	
	predictions = []
	latencies = []
	
	print(f"\n🔍 Single-sample inference (warmup={warmup_steps})...")
	
	# Warmup
	for i in range(min(warmup_steps, len(test_split))):
		sample = test_split[i]
		text = sample["text"]
		
		encoded = tokenizer(
			text,
			max_length=max_length,
			padding="max_length",
			truncation=True,
			return_tensors="pt"
		)
		
		input_ids = encoded["input_ids"].to(device)
		attention_mask = encoded["attention_mask"].to(device)
		
		with torch.no_grad():
			_ = model(input_ids=input_ids, attention_mask=attention_mask)
	
	# Actual inference with timing
	total_start = time.time()
	
	for idx, sample in enumerate(tqdm(test_split, desc="Single-sample eval")):
		text = sample["text"]
		true_label = int(sample["label"])
		
		# Encode
		encoded = tokenizer(
			text,
			max_length=max_length,
			padding="max_length",
			truncation=True,
			return_tensors="pt"
		)
		
		input_ids = encoded["input_ids"].to(device)
		attention_mask = encoded["attention_mask"].to(device)
		
		# Time inference
		start_time = time.time()
		with torch.no_grad():
			outputs = model(input_ids=input_ids, attention_mask=attention_mask)
		end_time = time.time()
		
		latency_ms = (end_time - start_time) * 1000
		latencies.append(latency_ms)
		
		# Get predictions
		logits = outputs["logits"].cpu().numpy()[0]  # (num_classes,)
		probabilities = torch.softmax(torch.tensor(logits), dim=0).numpy()
		pred_label = int(np.argmax(logits))
		confidence = float(probabilities[pred_label])
		
		predictions.append({
			"id": idx,
			"text": text,
			"true_label": true_label,
			"pred_label": pred_label,
			"confidence": confidence,
			"probabilities": probabilities.tolist(),
			"correct": pred_label == true_label,
			"latency_ms": latency_ms,
		})
	
	total_time = time.time() - total_start
	avg_latency = np.mean(latencies)
	
	return {
		"predictions": predictions,
		"average_latency_ms": float(avg_latency),
		"total_time_s": float(total_time),
	}


def batched_inference(
	model: BertWithClassifierImproved,
	test_loader: DataLoader,
	test_split,
	device: torch.device,
	warmup_steps: int = 10,
) -> Dict:
	"""
	Run batched inference to measure batched latency.
	
	Returns:
		Dictionary with predictions, latencies, and metadata
	"""
	model.eval()
	model.to(device)
	
	predictions = []
	batch_latencies = []
	
	print(f"\n🔍 Batched inference (batch_size={test_loader.batch_size}, warmup={warmup_steps})...")
	
	# Warmup
	warmup_batches = 0
	for batch in test_loader:
		input_ids = batch["input_ids"].to(device)
		attention_mask = batch["attention_mask"].to(device)
		
		with torch.no_grad():
			_ = model(input_ids=input_ids, attention_mask=attention_mask)
		
		warmup_batches += 1
		if warmup_batches >= warmup_steps:
			break
	
	# Actual inference with timing
	total_start = time.time()
	sample_idx = 0
	
	for batch in tqdm(test_loader, desc="Batched eval"):
		input_ids = batch["input_ids"].to(device)
		attention_mask = batch["attention_mask"].to(device)
		labels = batch["labels"]
		
		# Time batch inference
		batch_start = time.time()
		with torch.no_grad():
			outputs = model(input_ids=input_ids, attention_mask=attention_mask)
		batch_end = time.time()
		
		batch_latency = batch_end - batch_start
		batch_latencies.append(batch_latency)
		
		# Process batch results
		logits = outputs["logits"].cpu().numpy()  # (batch_size, num_classes)
		probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()
		pred_labels = np.argmax(logits, axis=1)
		
		batch_size = logits.shape[0]
		per_sample_latency = (batch_latency * 1000) / batch_size  # ms
		
		for i in range(batch_size):
			text = test_split[sample_idx]["text"]
			true_label = int(labels[i])
			pred_label = int(pred_labels[i])
			confidence = float(probabilities[i, pred_label])
			
			predictions.append({
				"id": sample_idx,
				"text": text,
				"true_label": true_label,
				"pred_label": pred_label,
				"confidence": confidence,
				"probabilities": probabilities[i].tolist(),
				"correct": pred_label == true_label,
				"latency_ms": per_sample_latency,
			})
			
			sample_idx += 1
	
	total_time = time.time() - total_start
	
	# Calculate average per-sample latency
	total_samples = sum(len(batch["labels"]) for batch in test_loader)
	avg_latency_per_sample = (sum(batch_latencies) * 1000) / total_samples
	
	return {
		"predictions": predictions,
		"average_latency_ms": float(avg_latency_per_sample),
		"total_time_s": float(total_time),
		"batch_latencies_s": [float(x) for x in batch_latencies],
	}


def generate_wordcloud(error_texts: List[str], output_path: Path):
	"""Generate word cloud from error samples."""
	try:
		from wordcloud import WordCloud
		import matplotlib.pyplot as plt
		
		if not error_texts:
			print("⚠️  No error samples to generate word cloud")
			return
		
		# Concatenate all error texts
		combined_text = " ".join(error_texts)
		
		# Generate word cloud
		wordcloud = WordCloud(
			width=1600,
			height=800,
			background_color="white",
			colormap="Reds",
			max_words=100,
			relative_scaling=0.5,
			min_font_size=10,
		).generate(combined_text)
		
		# Plot
		plt.figure(figsize=(16, 8))
		plt.imshow(wordcloud, interpolation="bilinear")
		plt.axis("off")
		plt.title("Word Cloud of Misclassified Samples", fontsize=20, pad=20)
		plt.tight_layout(pad=0)
		
		# Save
		plt.savefig(output_path, dpi=150, bbox_inches="tight")
		plt.close()
		
		print(f"📊 Saved word cloud to: {output_path}")
		
	except ImportError:
		print("⚠️  wordcloud not installed. Skipping word cloud generation.")
		print("   Install with: pip install wordcloud matplotlib")


def save_results(
	single_results: Dict,
	batched_results: Dict,
	output_dir: Path,
	metadata: Dict,
):
	"""Save all results to JSON and generate visualizations."""
	
	output_dir.mkdir(parents=True, exist_ok=True)
	
	# Calculate accuracy from single-sample results
	correct = sum(1 for p in single_results["predictions"] if p["correct"])
	total = len(single_results["predictions"])
	accuracy = correct / total if total > 0 else 0.0
	
	# Prepare final JSON structure
	result_json = {
		"experiment_metadata": {
			"method": metadata["method"],
			"other_method_detail": metadata["method_detail"],
			"dataset_name": metadata["dataset_name"],
			"num_samples": total,
			"num_classes": metadata["num_classes"],
			"accuracy": round(accuracy, 4),
			"correct_samples": correct,
			"error_samples": total - correct,
			# Single-sample metrics
			"average_latency_ms": round(single_results["average_latency_ms"], 2),
			"total_time_s": round(single_results["total_time_s"], 2),
			# Batched metrics
			"batched_average_latency_ms": round(batched_results["average_latency_ms"], 2),
			"batched_total_time_s": round(batched_results["total_time_s"], 2),
			"batch_size": metadata["batch_size"],
			# Model info
			"model_path": metadata["model_path"],
			"encoder_path": metadata["encoder_path"],
			"device": metadata["device"],
		},
		"predictions": single_results["predictions"]  # Use single-sample predictions with individual timings
	}
	
	# Save main results JSON
	results_file = output_dir / "evaluation_results.json"
	with open(results_file, "w", encoding="utf-8") as f:
		json.dump(result_json, f, indent=2, ensure_ascii=False)
	print(f"💾 Saved results to: {results_file}")
	
	# Save batched timing details separately
	batched_timing_file = output_dir / "batched_timing_details.json"
	with open(batched_timing_file, "w", encoding="utf-8") as f:
		json.dump({
			"batch_size": metadata["batch_size"],
			"num_batches": len(batched_results.get("batch_latencies_s", [])),
			"batch_latencies_s": batched_results.get("batch_latencies_s", []),
			"average_batch_latency_s": round(np.mean(batched_results.get("batch_latencies_s", [0])), 4),
			"total_time_s": round(batched_results["total_time_s"], 2),
			"average_per_sample_latency_ms": round(batched_results["average_latency_ms"], 2),
		}, f, indent=2)
	print(f"💾 Saved batched timing details to: {batched_timing_file}")
	
	# Extract error samples for word cloud
	error_texts = [p["text"] for p in single_results["predictions"] if not p["correct"]]
	
	if error_texts:
		wordcloud_path = output_dir / "error_samples_wordcloud.png"
		generate_wordcloud(error_texts, wordcloud_path)
		
		# Also save error samples as separate JSON
		error_file = output_dir / "error_samples.json"
		error_samples = [p for p in single_results["predictions"] if not p["correct"]]
		with open(error_file, "w", encoding="utf-8") as f:
			json.dump({
				"num_errors": len(error_samples),
				"error_rate": round((len(error_samples) / total), 4),
				"errors": error_samples
			}, f, indent=2, ensure_ascii=False)
		print(f"💾 Saved error samples to: {error_file}")
	
	# Save summary text file
	summary_file = output_dir / "summary.txt"
	with open(summary_file, "w", encoding="utf-8") as f:
		f.write("=" * 60 + "\n")
		f.write("EVALUATION SUMMARY\n")
		f.write("=" * 60 + "\n\n")
		f.write(f"Method: {metadata['method']}\n")
		f.write(f"Dataset: {metadata['dataset_name']}\n")
		f.write(f"Model: {metadata['model_path']}\n\n")
		f.write(f"Total Samples: {total}\n")
		f.write(f"Correct: {correct}\n")
		f.write(f"Errors: {total - correct}\n")
		f.write(f"Accuracy: {accuracy*100:.2f}%\n\n")
		f.write("-" * 60 + "\n")
		f.write("LATENCY METRICS\n")
		f.write("-" * 60 + "\n\n")
		f.write("Single-Sample Inference:\n")
		f.write(f"  Average Latency: {single_results['average_latency_ms']:.2f} ms\n")
		f.write(f"  Total Time: {single_results['total_time_s']:.2f} s\n\n")
		f.write(f"Batched Inference (batch_size={metadata['batch_size']}):\n")
		f.write(f"  Average Per-Sample Latency: {batched_results['average_latency_ms']:.2f} ms\n")
		f.write(f"  Total Time: {batched_results['total_time_s']:.2f} s\n")
		f.write(f"  Speedup: {single_results['total_time_s'] / batched_results['total_time_s']:.2f}x\n\n")
	print(f"💾 Saved summary to: {summary_file}")


def main():
	args = parse_args()
	
	# Set seed
	set_seed(args.seed)
	
	# Set device
	device = torch.device(args.device)
	print(f"🖥️  Using device: {device}")
	
	# Load dataset
	print(f"📂 Loading dataset: {args.dataset}")
	loader_fn = get_loader_by_name(args.dataset)
	if args.dataset == "clinc_oos":
		raw_dataset, stats = loader_fn(data_dir=args.dataset_dir, version=args.clinc_version)
	else:
		raw_dataset, stats = loader_fn(data_dir=args.dataset_dir)
	
	assert "test" in raw_dataset, f"Dataset {args.dataset} does not have a 'test' split"
	test_split = raw_dataset["test"]
	num_labels = stats["num_labels"]
	
	print(f"✅ Loaded {len(test_split)} test samples")
	print(f"✅ Number of labels: {num_labels}")
	
	# Load tokenizer
	tokenizer_path = args.tokenizer_path if args.tokenizer_path else args.encoder_path
	print(f"🔤 Loading tokenizer from: {tokenizer_path}")
	tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
	
	# Load model
	print(f"🤖 Loading model from: {args.model_path}")
	model = BertWithClassifierImproved.from_pretrained(
		model_name=args.encoder_path,
		extra_path=args.model_path,
		freeze_encoder=False,
	)
	print(f"✅ Model loaded successfully\n")
	
	# Run single-sample inference
	single_results = single_sample_inference(
		model=model,
		test_split=test_split,
		tokenizer=tokenizer,
		device=device,
		max_length=args.max_length,
		warmup_steps=args.warmup_steps,
	)
	
	# Create batched dataloader
	test_ds = BertTextDataset(test_split, tokenizer, max_length=args.max_length)
	collator = DynamicPaddingCollator(tokenizer)
	test_loader = DataLoader(
		test_ds,
		batch_size=args.batch_size,
		shuffle=False,
		collate_fn=collator
	)
	
	# Run batched inference
	batched_results = batched_inference(
		model=model,
		test_loader=test_loader,
		test_split=test_split,
		device=device,
		warmup_steps=args.warmup_steps,
	)
	
	# Calculate and print accuracy
	correct = sum(1 for p in single_results["predictions"] if p["correct"])
	total = len(single_results["predictions"])
	accuracy = correct / total if total > 0 else 0.0
	
	print("\n" + "=" * 60)
	print("📊 EVALUATION RESULTS")
	print("=" * 60)
	print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
	print(f"Correct: {correct}/{total}")
	print(f"Errors: {total - correct}")
	print("=" * 60 + "\n")
	
	# Save results
	metadata = {
		"method": args.method,
		"method_detail": args.method_detail,
		"dataset_name": args.dataset,
		"num_classes": num_labels,
		"batch_size": args.batch_size,
		"model_path": args.model_path,
		"encoder_path": args.encoder_path,
		"device": str(device),
	}
	
	output_dir = Path(args.output_dir)
	save_results(single_results, batched_results, output_dir, metadata)
	
	print("\n✅ Evaluation complete!")
	print(f"📁 Results saved to: {output_dir}")


if __name__ == "__main__":
	main()
