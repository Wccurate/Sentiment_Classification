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
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    set_seed,
)
from peft import PeftModel
from tqdm.auto import tqdm

# Dataset loaders
from data.loaders.load_raw_data import (
    load_raw_atis,
    load_raw_hwu64,
    load_raw_snips,
    load_raw_clinc_oos,
    load_raw_project,
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
    if name == "project":
        return load_raw_project
    raise ValueError(f"Unknown dataset name: {name}")


class UnifiedTextDataset(torch.utils.data.Dataset):
    """Unified text classification dataset."""

    def __init__(self, raw_data, tokenizer, max_length: int = 512):
        self.data = raw_data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = sample["text"]
        
        # Check if label exists (for test sets without labels)
        label = int(sample["label"]) if "label" in sample and sample["label"] is not None else -1

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


# ========== Model Loading ==========

def load_model_and_tokenizer(args, num_labels: int, device: torch.device):
    """Load model and tokenizer based on model type."""
    
    if args.model_type == "bert-head":
        # Transformer Head Only or Full Fine-tuning (saved with save_pretrained)
        # Supports BERT, RoBERTa, DistilBERT, DistilRoBERTa, and any compatible encoder
        print(f"Loading transformer model from: {args.model_path}")
        
        # Use AutoTokenizer and AutoModelForSequenceClassification for flexibility
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        
        # Detect model type for logging
        config = AutoConfig.from_pretrained(args.model_path)
        model_type = config.model_type if hasattr(config, 'model_type') else 'unknown'
        print(f"Detected model type: {model_type}")
        
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path,
            num_labels=num_labels,
        )
        
    elif args.model_type == "bert-lora":
        # BERT LoRA (PEFT)
        print(f"Loading base model from: {args.base_model_path}")
        print(f"Loading LoRA adapter from: {args.adapter_path}")
        
        # Try to load tokenizer from adapter path first (saved during training)
        tokenizer_path = args.adapter_path if Path(args.adapter_path).exists() else args.base_model_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Detect model type
        config = AutoConfig.from_pretrained(args.base_model_path)
        model_type = config.model_type if hasattr(config, 'model_type') else 'unknown'
        print(f"Detected base model type: {model_type}")
        
        # Load base model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model_path,
            num_labels=num_labels,
        )
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
        if args.merge_adapter:
            print("Merging LoRA adapter for faster inference...")
            model = model.merge_and_unload()
        
    elif args.model_type == "bert-prompt":
        # BERT Prompt Tuning (PEFT)
        print(f"Loading base model from: {args.base_model_path}")
        print(f"Loading Prompt Tuning adapter from: {args.adapter_path}")

        # Try to load tokenizer from adapter path first (saved during training)
        tokenizer_path = args.adapter_path if Path(args.adapter_path).exists() else args.base_model_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Detect model type
        config = AutoConfig.from_pretrained(args.base_model_path)
        model_type = config.model_type if hasattr(config, 'model_type') else 'unknown'
        print(f"Detected base model type: {model_type}")
        
        # Load base model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model_path,
            num_labels=num_labels,
        )
        
        # Load Prompt Tuning adapter
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
        # Note: Prompt Tuning does NOT support merge_and_unload() as it uses virtual tokens
        # The model is already optimized for inference with prompt embeddings
        if args.merge_adapter:
            print("    Note: Prompt Tuning does not support adapter merging (uses virtual tokens)")
            print("    Using model with prompt adapter attached (already efficient)")
        
    elif args.model_type == "qwen3-lora":
        # Qwen3 LoRA (PEFT)
        print(f"  Loading base Qwen3 model from: {args.base_model_path}")
        print(f"  Loading LoRA adapter from: {args.adapter_path}")
        
        tokenizer_path = args.adapter_path if Path(args.adapter_path).exists() else args.base_model_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        base_model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model_path,
            num_labels=num_labels,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if args.use_bf16 else torch.float32,
        )
        
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
        if args.merge_adapter:
            print("  Merging adapter for faster inference...")
            model = model.merge_and_unload()
        
        model.config.pad_token_id = tokenizer.pad_token_id
        
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Set pad_token_id if not set
    if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
        if hasattr(model.config, 'pad_token_id'):
            model.config.pad_token_id = tokenizer.pad_token_id
    
    model.eval()
    model.to(device)
    
    print(" Model and tokenizer loaded successfully\n")
    
    return model, tokenizer


# ========== Evaluation Functions ==========

def single_sample_inference(
    model,
    test_split,
    tokenizer,
    device: torch.device,
    max_length: int,
    warmup_steps: int = 10,
) -> Dict:
    """Run single-sample inference with timing."""
    model.eval()

    predictions = []
    latencies = []

    print(f"\n Single-sample inference (warmup={warmup_steps})...")

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
        # Check if label exists
        true_label = int(sample["label"]) if "label" in sample and sample["label"] is not None else None

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

        pred_dict = {
            "id": idx,
            "text": text,
            "pred_label": pred_label,
            "confidence": confidence,
            "probabilities": probabilities.tolist(),
            "latency_ms": latency_ms,
        }
        
        # Only add true_label and correct if label exists
        if true_label is not None:
            pred_dict["true_label"] = true_label
            pred_dict["correct"] = pred_label == true_label
        
        predictions.append(pred_dict)

    total_time = time.time() - total_start
    avg_latency = float(np.mean(latencies))

    return {
        "predictions": predictions,
        "average_latency_ms": avg_latency,
        "total_time_s": float(total_time),
    }


def batched_inference(
    model,
    test_loader: DataLoader,
    test_split,
    device: torch.device,
    warmup_steps: int = 10,
) -> Dict:
    """Run batched inference with timing."""
    model.eval()

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
            sample = test_split[sample_idx]
            
            pred_dict = {
                "id": sample_idx,
                "text": sample["text"],
                "pred_label": int(pred_labels[i]),
                "confidence": float(probabilities[i][pred_labels[i]]),
                "probabilities": probabilities[i].tolist(),
                "latency_ms": per_sample_latency,
            }
            
            # Only add true_label and correct if label exists (labels[i] != -1)
            if int(labels[i]) != -1:
                pred_dict["true_label"] = int(labels[i])
                pred_dict["correct"] = int(pred_labels[i]) == int(labels[i])
            
            predictions.append(pred_dict)
            sample_idx += 1

    total_time = time.time() - total_start
    total_samples = len(predictions)
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
            print("  No errors to generate word cloud")
            return

        combined_text = " ".join(error_texts)

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            colormap="viridis",
            max_words=100,
        ).generate(combined_text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Error Sample Word Cloud", fontsize=16)
        plt.tight_layout(pad=0)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f" Saved word cloud to: {output_path}")

    except ImportError:
        print("  wordcloud not installed. Skipping word cloud generation.")


def save_results(
    single_results: Dict,
    batched_results: Dict,
    output_dir: Path,
    metadata: Dict,
):
    """Save evaluation results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(single_results["predictions"])
    
    # Check if labels exist by checking if any prediction has 'correct' field
    has_labels = any("correct" in p for p in single_results["predictions"])
    
    # Calculate accuracy only if labels exist
    if has_labels:
        correct = sum(1 for p in single_results["predictions"] if p.get("correct", False))
        accuracy = correct / total if total > 0 else 0.0
    else:
        correct = None
        accuracy = None

    # Build metadata
    exp_metadata = {
        "model_type": metadata["model_type"],
        "method": metadata["method"],
        "dataset_name": metadata["dataset_name"],
        "num_samples": total,
        "num_classes": metadata["num_classes"],
        "has_labels": has_labels,
        "average_latency_ms": round(single_results["average_latency_ms"], 2),
        "total_time_s": round(single_results["total_time_s"], 2),
        "batched_average_latency_ms": round(batched_results["average_latency_ms"], 2),
        "batched_total_time_s": round(batched_results["total_time_s"], 2),
        "batch_size": metadata["batch_size"],
        "model_path": metadata.get("model_path", "N/A"),
        "base_model_path": metadata.get("base_model_path", "N/A"),
        "adapter_path": metadata.get("adapter_path", "N/A"),
        "device": metadata["device"],
    }
    
    # Add accuracy metrics only if labels exist
    if has_labels:
        exp_metadata["accuracy"] = round(accuracy, 4)
        exp_metadata["correct_samples"] = correct
        exp_metadata["error_samples"] = total - correct

    result_json = {
        "experiment_metadata": exp_metadata,
        "predictions": single_results["predictions"],
    }

    results_file = output_dir / "evaluation_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)
    print(f" Saved results to: {results_file}")

    batched_timing_file = output_dir / "batched_timing_details.json"
    with open(batched_timing_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "batch_size": metadata["batch_size"],
                "total_batches": len(batched_results["batch_latencies_s"]),
                "average_latency_per_sample_ms": batched_results["average_latency_ms"],
                "total_time_s": batched_results["total_time_s"],
                "batch_latencies_s": batched_results["batch_latencies_s"],
                "predictions": batched_results["predictions"],
            },
            f,
            indent=2,
        )
    print(f" Saved batched timing details to: {batched_timing_file}")

    # Only save error samples if labels exist
    if has_labels:
        error_texts = [p["text"] for p in single_results["predictions"] if not p.get("correct", True)]

        if error_texts:
            error_samples_file = output_dir / "error_samples.json"
            error_samples = [p for p in single_results["predictions"] if not p.get("correct", True)]
            with open(error_samples_file, "w", encoding="utf-8") as f:
                json.dump(error_samples, f, indent=2, ensure_ascii=False)
            print(f" Saved {len(error_samples)} error samples to: {error_samples_file}")

            wordcloud_path = output_dir / "error_wordcloud.png"
            generate_wordcloud(error_texts, wordcloud_path)

    summary_file = output_dir / "summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("EVALUATION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model Type: {metadata['model_type']}\n")
        f.write(f"Method: {metadata['method']}\n")
        f.write(f"Dataset: {metadata['dataset_name']}\n")
        f.write(f"Device: {metadata['device']}\n\n")
        f.write(f"Model Path: {metadata.get('model_path', 'N/A')}\n")
        f.write(f"Base Model Path: {metadata.get('base_model_path', 'N/A')}\n")
        f.write(f"Adapter Path: {metadata.get('adapter_path', 'N/A')}\n\n")
        f.write(f"Total Samples: {total}\n")
        
        # Only write accuracy metrics if labels exist
        if has_labels:
            f.write(f"Correct: {correct}\n")
            f.write(f"Errors: {total - correct}\n")
            f.write(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        else:
            f.write("Labels: Not available (prediction only mode)\n\n")
        
        f.write("Single-sample Inference:\n")
        f.write(f"  Average Latency: {single_results['average_latency_ms']:.2f} ms\n")
        f.write(f"  Total Time: {single_results['total_time_s']:.2f} s\n\n")
        f.write(f"Batched Inference (batch_size={metadata['batch_size']}):\n")
        f.write(f"  Average Latency per Sample: {batched_results['average_latency_ms']:.2f} ms\n")
        f.write(f"  Total Time: {batched_results['total_time_s']:.2f} s\n")
        f.write("=" * 60 + "\n")
    print(f" Saved summary to: {summary_file}")


# ========== Argument Parser ==========

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified evaluation script for all models")

    # Model type
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["bert-head", "bert-lora", "bert-prompt", "qwen3-lora"],
        help="Type of model to evaluate",
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["atis", "hwu64", "snips", "clinc_oos", "project"],
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

    # Model paths (for bert-head)
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model (for bert-head only)",
    )

    # Model paths (for bert-lora, bert-prompt and qwen3-lora)
    parser.add_argument(
        "--base-model-path",
        type=str,
        default=None,
        help="Path to base model (for bert-lora, bert-prompt and qwen3-lora)",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to PEFT adapter (for bert-lora, bert-prompt and qwen3-lora)",
    )

    # Evaluation settings
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
        default=None,
        help="Method name for metadata (auto-generated if not specified)",
    )
    parser.add_argument(
        "--merge-adapter",
        action="store_true",
        help="Merge PEFT adapter for faster inference (for PEFT models)",
    )
    parser.add_argument(
        "--use-bf16",
        action="store_true",
        help="Use bfloat16 for Qwen3 (recommended)",
    )
    parser.add_argument(
        "--use-train-split",
        action="store_true",
        help="Use training split for evaluation (if available)",
    )

    return parser.parse_args()


# ========== Main ==========

def main():
    args = parse_args()

    # print("Model Path: " + args.model_path)

    # Validate arguments based on model type
    if args.model_type == "bert-head":
        if not args.model_path:
            raise ValueError("--model-path is required for bert-head")
    else:  # bert-lora, bert-prompt or qwen3-lora
        if not args.base_model_path or not args.adapter_path:
            raise ValueError("--base-model-path and --adapter-path are required for PEFT models")

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
    # TODO: change the split
    test_split = raw_dataset["test"]
    # test_split=raw_dataset["train"]
    num_labels = stats["num_labels"]

    print(f"✅ Loaded {len(test_split)} test samples")
    print(f"✅ Number of labels: {num_labels}")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args, num_labels, device)

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
    test_ds = UnifiedTextDataset(test_split, tokenizer, max_length=args.max_length)
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

    # Calculate accuracy only if labels exist
    total = len(single_results["predictions"])
    has_labels = any("correct" in p for p in single_results["predictions"])
    
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model Type: {args.model_type}")
    print(f"Total Samples: {total}")
    
    if has_labels:
        correct = sum(1 for p in single_results["predictions"] if p.get("correct", False))
        accuracy = correct / total if total > 0 else 0.0
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Correct: {correct}/{total}")
        print(f"Errors: {total - correct}")
    else:
        print("Mode: Prediction only (no labels available)")
    
    print("=" * 60 + "\n")

    # Prepare metadata
    method = args.method
    if method is None:
        method_map = {
            "bert-head": "BERT-Head",
            "bert-lora": "BERT-LoRA",
            "bert-prompt": "BERT-PromptTuning",
            "qwen3-lora": "Qwen3-1.7B-LoRA",
        }
        method = method_map.get(args.model_type, args.model_type)

    metadata = {
        "model_type": args.model_type,
        "method": method,
        "dataset_name": args.dataset,
        "num_classes": num_labels,
        "batch_size": args.batch_size,
        "device": str(device),
    }

    if args.model_type == "bert-head":
        metadata["model_path"] = args.model_path
    else:
        metadata["base_model_path"] = args.base_model_path
        metadata["adapter_path"] = args.adapter_path

    # Save results
    output_dir = Path(args.output_dir)
    save_results(single_results, batched_results, output_dir, metadata)

    print("\n✅ Evaluation complete!")
    print(f"📁 Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
