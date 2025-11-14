from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, AutoConfig, set_seed
from tqdm.auto import tqdm

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
    parser = argparse.ArgumentParser(
        description="Zero-shot evaluation using pretrained BERT + cosine similarity"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["atis", "hwu64", "snips", "clinc_oos"],
        help="Which dataset to evaluate on"
    )
    parser.add_argument(
        "--clinc-version",
        type=str,
        default="plus",
        choices=["small", "plus", "imbalanced"],
        help="Version for clinc_oos dataset"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/raw",
        help="Base directory for raw datasets"
    )
    
    # Model arguments
    parser.add_argument(
        "--encoder-path",
        type=str,
        required=True,
        help="Path to pretrained BERT encoder"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="",
        help="Tokenizer path (defaults to encoder-path)"
    )
    
    # Evaluation settings
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="BERT-Pretrained-CLS-Cosine",
        help="Method name for metadata"
    )
    
    return parser.parse_args()


class TextDataset(Dataset):
    """Simple dataset for text encoding."""
    
    def __init__(self, texts: List[str]):
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]


def collate_fn(batch: List[str], tokenizer, max_length: int):
    """Collate function for batching texts."""
    encoded = tokenizer(
        batch,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    return encoded


def sum_pool_normalise(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Sum token vectors (mask-aware) and apply L2 length normalisation."""
    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    summed = (hidden_states * mask).sum(dim=1)
    return F.normalize(summed, p=2, dim=-1, eps=1e-6)


def encode_texts_batched(
    texts: List[str],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    batch_size: int,
    max_length: int,
    desc: str = "Encoding",
) -> np.ndarray:
    """
    Encode a list of texts using length-normalised sum-pooled token representations.
    
    Args:
        texts: List of text strings to encode
        model: Pretrained BERT model
        tokenizer: Tokenizer
        device: Device to run on
        batch_size: Batch size for encoding
        max_length: Maximum sequence length
        desc: Description for progress bar
        
    Returns:
        numpy array of shape (len(texts), hidden_size)
    """
    model.eval()
    model.to(device)
    
    dataset = TextDataset(texts)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, max_length)
    )
    
    all_embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

            pooled = sum_pool_normalise(outputs.last_hidden_state, attention_mask)
            
            # Move to CPU and store
            all_embeddings.append(pooled.cpu().numpy())
    
    # Concatenate all batches
    embeddings = np.concatenate(all_embeddings, axis=0)
    
    return embeddings


def compute_cosine_similarity(
    query_embeddings: np.ndarray,
    label_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between query embeddings and label embeddings.
    
    Args:
        query_embeddings: (num_queries, hidden_size)
        label_embeddings: (num_labels, hidden_size)
        
    Returns:
        similarity matrix of shape (num_queries, num_labels)
    """
    # Normalize embeddings
    query_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
    label_norm = label_embeddings / (np.linalg.norm(label_embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Compute cosine similarity: (num_queries, num_labels)
    similarity = np.matmul(query_norm, label_norm.T)
    
    return similarity


def run_single_sample_evaluation(
    test_split,
    label_names: List[str],
    text_label_to_label: Dict[str, int],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_length: int,
) -> Dict:
    """
    Run single-sample evaluation (encode each test sample individually).
    
    Returns:
        Dictionary with predictions and timing info
    """
    print("\n🔍 Single-sample evaluation...")
    
    # Step 1: Encode all label names (once)
    print(f"📝 Encoding {len(label_names)} label names...")
    label_embeddings = encode_texts_batched(
        texts=label_names,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=len(label_names),  # Encode all labels at once
        max_length=max_length,
        desc="Encoding labels"
    )
    print(f"✅ Label embeddings shape: {label_embeddings.shape}")
    
    # Step 2: Encode and predict each test sample
    predictions = []
    latencies = []
    
    model.eval()
    model.to(device)
    
    total_start = time.time()
    
    for idx, sample in enumerate(tqdm(test_split, desc="Single-sample eval")):
        text = sample["text"]
        true_text_label = sample["text_label"]
        true_label = int(sample["label"])
        
        # Encode the utterance
        start_time = time.time()
        
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
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            pooled = sum_pool_normalise(outputs.last_hidden_state, attention_mask)
            utterance_embedding = pooled.cpu().numpy()
        
        end_time = time.time()
        
    # Compute similarity
        similarity = compute_cosine_similarity(utterance_embedding, label_embeddings)
        similarity_scores = similarity[0]  # (num_labels,)
        
        # Get prediction
        pred_label_idx = int(np.argmax(similarity_scores))
        pred_text_label = label_names[pred_label_idx]
        pred_label = text_label_to_label[pred_text_label]
        confidence = float(similarity_scores[pred_label_idx])
        
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
        
        predictions.append({
            "id": idx,
            "text": text,
            "true_label": true_label,
            "true_text_label": true_text_label,
            "pred_label": pred_label,
            "pred_text_label": pred_text_label,
            "confidence": confidence,
            "similarity_scores": similarity_scores.tolist(),
            "correct": pred_label == true_label,
            "latency_ms": latency_ms,
        })
    
    total_time = time.time() - total_start
    
    return {
        "predictions": predictions,
        "average_latency_ms": float(np.mean(latencies)),
        "total_time_s": float(total_time),
    }


def run_batched_evaluation(
    test_split,
    label_names: List[str],
    text_label_to_label: Dict[str, int],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> Dict:
    """
    Run batched evaluation (encode test samples in batches).
    
    Returns:
        Dictionary with predictions and timing info
    """
    print("\n🔍 Batched evaluation...")
    
    # Step 1: Encode all label names (once)
    print(f"📝 Encoding {len(label_names)} label names...")
    label_embeddings = encode_texts_batched(
        texts=label_names,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=len(label_names),
        max_length=max_length,
        desc="Encoding labels"
    )
    print(f"✅ Label embeddings shape: {label_embeddings.shape}")
    
    # Step 2: Encode all test utterances in batches
    test_texts = [sample["text"] for sample in test_split]
    
    start_time = time.time()
    utterance_embeddings = encode_texts_batched(
        texts=test_texts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        desc="Encoding test samples"
    )
    encoding_time = time.time() - start_time
    
    print(f"✅ Utterance embeddings shape: {utterance_embeddings.shape}")
    
    # Step 3: Compute similarities for all samples at once
    print("🔢 Computing cosine similarities...")
    similarity_matrix = compute_cosine_similarity(utterance_embeddings, label_embeddings)
    print(f"✅ Similarity matrix shape: {similarity_matrix.shape}")
    
    # Step 4: Get predictions
    pred_label_indices = np.argmax(similarity_matrix, axis=1)
    
    predictions = []
    
    for idx, sample in enumerate(test_split):
        text = sample["text"]
        true_text_label = sample["text_label"]
        true_label = int(sample["label"])
        
        pred_label_idx = int(pred_label_indices[idx])
        pred_text_label = label_names[pred_label_idx]
        pred_label = text_label_to_label[pred_text_label]
        confidence = float(similarity_matrix[idx, pred_label_idx])
        
        # Calculate per-sample latency (approximation)
        per_sample_latency = (encoding_time * 1000) / len(test_split)
        
        predictions.append({
            "id": idx,
            "text": text,
            "true_label": true_label,
            "true_text_label": true_text_label,
            "pred_label": pred_label,
            "pred_text_label": pred_text_label,
            "confidence": confidence,
            "similarity_scores": similarity_matrix[idx].tolist(),
            "correct": pred_label == true_label,
            "latency_ms": per_sample_latency,
        })
    
    return {
        "predictions": predictions,
        "average_latency_ms": float((encoding_time * 1000) / len(test_split)),
        "total_time_s": float(encoding_time),
    }


def save_results(
    single_results: Dict,
    batched_results: Dict,
    output_dir: Path,
    metadata: Dict,
):
    """Save all results to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate accuracy from single-sample results
    correct = sum(1 for p in single_results["predictions"] if p["correct"])
    total = len(single_results["predictions"])
    accuracy = correct / total if total > 0 else 0.0
    
    # Prepare final JSON structure
    result_json = {
        "experiment_metadata": {
            "method": metadata["method"],
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
            "encoder_path": metadata["encoder_path"],
            "device": metadata["device"],
        },
        "predictions": single_results["predictions"]
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
            "total_time_s": round(batched_results["total_time_s"], 2),
            "average_per_sample_latency_ms": round(batched_results["average_latency_ms"], 2),
        }, f, indent=2)
    print(f"💾 Saved batched timing details to: {batched_timing_file}")
    
    # Save error samples
    error_samples = [p for p in single_results["predictions"] if not p["correct"]]
    if error_samples:
        error_file = output_dir / "error_samples.json"
        with open(error_file, "w", encoding="utf-8") as f:
            json.dump({
                "num_errors": len(error_samples),
                "error_rate": round(len(error_samples) / total, 4),
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
        f.write(f"Encoder: {metadata['encoder_path']}\n\n")
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
        if batched_results['total_time_s'] > 0:
            f.write(f"  Speedup: {single_results['total_time_s'] / batched_results['total_time_s']:.2f}x\n\n")
    print(f"💾 Saved summary to: {summary_file}")


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device(args.device)
    print(f"🖥️  Using device: {device}")
    
    # Load dataset with label mappings
    print(f"📂 Loading dataset: {args.dataset}")
    loader_fn = get_loader_by_name(args.dataset)
    
    if args.dataset == "clinc_oos":
        raw_dataset, stats, text_label_to_label, label_to_text_label = loader_fn(
            data_dir=args.dataset_dir,
            version=args.clinc_version,
            return_dicts=True
        )
    else:
        raw_dataset, stats, text_label_to_label, label_to_text_label = loader_fn(
            data_dir=args.dataset_dir,
            return_dicts=True
        )
    
    test_split = raw_dataset["test"]
    # label_names = stats["label_names"]
    label_names=list(text_label_to_label.keys())
    num_labels = len(label_names)
    
    print(f"✅ Loaded {len(test_split)} test samples")
    print(f"✅ Number of labels: {num_labels}")
    print(f"✅ Label names: {label_names[:5]}... (showing first 5)")
    
    # Load tokenizer
    tokenizer_path = args.tokenizer_path if args.tokenizer_path else args.encoder_path
    print(f"\n🔤 Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Load model
    print(f"🤖 Loading pretrained BERT from: {args.encoder_path}")
    config = AutoConfig.from_pretrained(args.encoder_path)
    model = AutoModel.from_pretrained(args.encoder_path, config=config)
    print(f"✅ Model loaded successfully (hidden_size={config.hidden_size})\n")
    
    # Run single-sample evaluation
    single_results = run_single_sample_evaluation(
        test_split=test_split,
        label_names=label_names,
        text_label_to_label=text_label_to_label,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=args.max_length,
    )
    
    # Run batched evaluation
    batched_results = run_batched_evaluation(
        test_split=test_split,
        label_names=label_names,
        text_label_to_label=text_label_to_label,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
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
        "dataset_name": args.dataset,
        "num_classes": num_labels,
        "batch_size": args.batch_size,
        "encoder_path": args.encoder_path,
        "device": str(device),
    }
    
    output_dir = Path(args.output_dir)
    save_results(single_results, batched_results, output_dir, metadata)
    
    print("\n✅ Evaluation complete!")
    print(f"📁 Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
