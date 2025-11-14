from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed
from peft import PeftModel
from tqdm.auto import tqdm

# Dataset loaders
from data.loaders.load_raw_data import (
    load_raw_atis,
    load_raw_hwu64,
    load_raw_snips,
    load_raw_clinc_oos,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ========== Dataset Utilities ==========

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


class Qwen3TextDataset(torch.utils.data.Dataset):
    """Text classification dataset for Qwen3."""

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

        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": label,
        }


class DynamicPaddingCollator:
    """Dynamic padding collator."""

    def __init__(self, tokenizer, padding_side: str = "right"):
        self.tokenizer = tokenizer
        self.padding_side = padding_side

    def __call__(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]

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


# ========== Argument Parser ==========

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3 LoRA model")

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["atis", "hwu64", "snips", "clinc_oos"],
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--clinc-version",
        type=str,
        default="plus",
        choices=["small", "plus", "imbalanced"],
        help="CLINC OOS version",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/raw",
        help="Base directory for datasets",
    )

    # Model
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="/projects/wangshibohdd/models/model_from_hf/qwen317b",
        help="Path to base Qwen3 model",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        required=True,
        help="Path to trained LoRA adapter",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="",
        help="Tokenizer path (defaults to lora-path or base-model-path)",
    )

    # Evaluation
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--max-length", type=int, default=256, help="Max sequence length")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--warmup-steps", type=int, default=10, help="Warmup steps before timing"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="Qwen3-1.7B-LoRA",
        help="Method name for metadata",
    )

    return parser.parse_args()


# ========== Evaluation Functions ==========

def single_sample_inference(
    model: PeftModel,
    test_split,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_length: int,
    warmup_steps: int = 10,
) -> Dict:
    """Run single-sample inference with timing."""
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
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

    # Actual inference
    total_start = time.time()

    for idx, sample in enumerate(tqdm(test_split, desc="Single-sample eval")):
        text = sample["text"]
        true_label = int(sample["label"])

        encoded = tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        start_time = time.time()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        end_time = time.time()

        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)

        logits = outputs.logits.cpu().to(dtype=torch.float32).numpy()[0]
        probabilities = torch.softmax(torch.tensor(logits), dim=0).numpy()
        pred_label = int(np.argmax(logits))
        confidence = float(probabilities[pred_label])

        predictions.append(
            {
                "id": idx,
                "text": text,
                "true_label": true_label,
                "pred_label": pred_label,
                "confidence": confidence,
                "probabilities": probabilities.tolist(),
                "correct": pred_label == true_label,
                "latency_ms": latency_ms,
            }
        )

    total_time = time.time() - total_start
    avg_latency = float(np.mean(latencies))

    return {
        "predictions": predictions,
        "average_latency_ms": avg_latency,
        "total_time_s": float(total_time),
    }


def batched_inference(
    model: PeftModel,
    test_loader: DataLoader,
    test_split,
    device: torch.device,
    warmup_steps: int = 10,
) -> Dict:
    """Run batched inference with timing."""
    model.eval()
    model.to(device)

    predictions = []
    batch_latencies = []

    print(
        f"\n🔍 Batched inference (batch_size={test_loader.batch_size}, warmup={warmup_steps})..."
    )

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

    # Actual inference
    total_start = time.time()
    sample_idx = 0

    for batch in tqdm(test_loader, desc="Batched eval"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        batch_start = time.time()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        batch_end = time.time()

        batch_latency = batch_end - batch_start
        batch_latencies.append(batch_latency)

        logits = outputs.logits.cpu().to(dtype=torch.float32).numpy()
        probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()
        pred_labels = np.argmax(logits, axis=1)

        batch_size = logits.shape[0]
        per_sample_latency = (batch_latency * 1000) / batch_size

        for i in range(batch_size):
            text = test_split[sample_idx]["text"]
            true_label = int(labels[i])
            pred_label = int(pred_labels[i])
            confidence = float(probabilities[i, pred_label])

            predictions.append(
                {
                    "id": sample_idx,
                    "text": text,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "confidence": confidence,
                    "probabilities": probabilities[i].tolist(),
                    "correct": pred_label == true_label,
                    "latency_ms": per_sample_latency,
                }
            )

            sample_idx += 1

    total_time = time.time() - total_start
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
            print("⚠️  No error samples for word cloud")
            return

        combined_text = " ".join(error_texts)

        wordcloud = WordCloud(
            width=1600,
            height=800,
            background_color="white",
            colormap="Reds",
            max_words=100,
            relative_scaling=0.5,
            min_font_size=10,
        ).generate(combined_text)

        plt.figure(figsize=(16, 8))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud of Misclassified Samples", fontsize=20, pad=20)
        plt.tight_layout(pad=0)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"📊 Saved word cloud to: {output_path}")

    except ImportError:
        print("⚠️  wordcloud not installed. Skipping.")


def save_results(
    single_results: Dict,
    batched_results: Dict,
    output_dir: Path,
    metadata: Dict,
):
    """Save evaluation results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    correct = sum(1 for p in single_results["predictions"] if p["correct"])
    total = len(single_results["predictions"])
    accuracy = correct / total if total > 0 else 0.0

    result_json = {
        "experiment_metadata": {
            "method": metadata["method"],
            "dataset_name": metadata["dataset_name"],
            "num_samples": total,
            "num_classes": metadata["num_classes"],
            "accuracy": round(accuracy, 4),
            "correct_samples": correct,
            "error_samples": total - correct,
            "average_latency_ms": round(single_results["average_latency_ms"], 2),
            "total_time_s": round(single_results["total_time_s"], 2),
            "batched_average_latency_ms": round(batched_results["average_latency_ms"], 2),
            "batched_total_time_s": round(batched_results["total_time_s"], 2),
            "batch_size": metadata["batch_size"],
            "base_model_path": metadata["base_model_path"],
            "lora_path": metadata["lora_path"],
            "device": metadata["device"],
        },
        "predictions": single_results["predictions"],
    }

    results_file = output_dir / "evaluation_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved results to: {results_file}")

    batched_timing_file = output_dir / "batched_timing_details.json"
    with open(batched_timing_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "batch_size": metadata["batch_size"],
                "num_batches": len(batched_results.get("batch_latencies_s", [])),
                "batch_latencies_s": batched_results.get("batch_latencies_s", []),
                "average_batch_latency_s": round(
                    np.mean(batched_results.get("batch_latencies_s", [0])), 4
                ),
                "total_time_s": round(batched_results["total_time_s"], 2),
                "average_per_sample_latency_ms": round(
                    batched_results["average_latency_ms"], 2
                ),
            },
            f,
            indent=2,
        )
    print(f"💾 Saved batched timing details to: {batched_timing_file}")

    error_texts = [p["text"] for p in single_results["predictions"] if not p["correct"]]

    if error_texts:
        wordcloud_path = output_dir / "error_samples_wordcloud.png"
        generate_wordcloud(error_texts, wordcloud_path)

        error_file = output_dir / "error_samples.json"
        error_samples = [p for p in single_results["predictions"] if not p["correct"]]
        with open(error_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "num_errors": len(error_samples),
                    "error_rate": round((len(error_samples) / total), 4),
                    "errors": error_samples,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"💾 Saved error samples to: {error_file}")

    summary_file = output_dir / "summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("EVALUATION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Method: {metadata['method']}\n")
        f.write(f"Dataset: {metadata['dataset_name']}\n")
        f.write(f"Base Model: {metadata['base_model_path']}\n")
        f.write(f"LoRA Adapter: {metadata['lora_path']}\n\n")
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
        f.write(
            f"  Average Per-Sample Latency: {batched_results['average_latency_ms']:.2f} ms\n"
        )
        f.write(f"  Total Time: {batched_results['total_time_s']:.2f} s\n")
        if batched_results["total_time_s"] > 0:
            speedup = single_results["total_time_s"] / batched_results["total_time_s"]
            f.write(f"  Speedup: {speedup:.2f}x\n\n")
    print(f"💾 Saved summary to: {summary_file}")


# ========== Main ==========

def main():
    args = parse_args()

    set_seed(args.seed)

    device = torch.device(args.device)
    print(f"🖥️  Using device: {device}")

    # Load dataset
    print(f"📂 Loading dataset: {args.dataset}")
    loader_fn = get_loader_by_name(args.dataset)

    if args.dataset == "clinc_oos":
        raw_dataset, stats = loader_fn(data_dir=args.dataset_dir, version=args.clinc_version)
    else:
        raw_dataset, stats = loader_fn(data_dir=args.dataset_dir)

    assert "test" in raw_dataset, "Expected 'test' split"
    test_split = raw_dataset["test"]
    num_labels = stats["num_labels"]

    print(f"✅ Loaded {len(test_split)} test samples")
    print(f"✅ Number of labels: {num_labels}")

    # Load tokenizer
    tokenizer_path = args.tokenizer_path if args.tokenizer_path else args.base_model_path
    print(f"\n🔤 Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    print(f"\n🤖 Loading base model from: {args.base_model_path}")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model_path,
        num_labels=num_labels,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Load LoRA adapter
    print(f"🔧 Loading LoRA adapter from: {args.lora_path}")
    model = PeftModel.from_pretrained(base_model, args.lora_path)
    model = model.merge_and_unload()  # Merge for faster inference
    print("✅ Model loaded successfully\n")
    model.config.pad_token_id = tokenizer.pad_token_id
    # accelerator.print(f"⚙️  Set model.config.pad_token_id to: {model.config.pad_token_id}")

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
    test_ds = Qwen3TextDataset(test_split, tokenizer, max_length=args.max_length)
    collator = DynamicPaddingCollator(tokenizer)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator
    )

    # Run batched inference
    batched_results = batched_inference(
        model=model,
        test_loader=test_loader,
        test_split=test_split,
        device=device,
        warmup_steps=args.warmup_steps,
    )

    # Calculate accuracy
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
        "dataset_name": args.dataset,
        "num_classes": num_labels,
        "batch_size": args.batch_size,
        "base_model_path": args.base_model_path,
        "lora_path": args.lora_path,
        "device": str(device),
    }

    output_dir = Path(args.output_dir)
    save_results(single_results, batched_results, output_dir, metadata)

    print("\n✅ Evaluation complete!")
    print(f"📁 Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
