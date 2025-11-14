from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import set_seed
from tqdm.auto import tqdm

# Dataset loaders
from data.loaders.load_raw_data import (  # type: ignore
    load_raw_atis,
    load_raw_clinc_oos,
    load_raw_hwu64,
    load_raw_snips,
)


def get_loader_by_name(name: str):
    """Return dataset loader function by dataset name."""
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
        description="Zero-shot evaluation using SentenceTransformer embeddings + cosine similarity",
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["atis", "hwu64", "snips", "clinc_oos"],
        help="Dataset to evaluate",
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

    # Model arguments
    parser.add_argument(
        "--encoder-path",
        type=str,
        default="/projects/wangshibohdd/models/model_from_hf/qwen3embedding06b",
        help="Path to pretrained SentenceTransformer checkpoint",
    )

    # Evaluation settings
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for evaluation outputs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for encoding",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length for the encoder",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference (cuda or cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="SentenceTransformer-Cosine",
        help="Method name saved in metadata",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable half-precision inference if supported",
    )
    parser.add_argument(
        "--normalize-embeddings",
        dest="normalize_embeddings",
        action="store_true",
        help="Apply L2 normalization to embeddings during encoding",
    )
    parser.add_argument(
        "--no-normalize-embeddings",
        dest="normalize_embeddings",
        action="store_false",
        help="Disable L2 normalization during encoding",
    )
    parser.set_defaults(normalize_embeddings=True)

    return parser.parse_args()


def encode_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int,
    normalize: bool,
    desc: str,
) -> np.ndarray:
    """Encode texts into embeddings using the SentenceTransformer."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    return embeddings


def compute_cosine_similarity(
    query_embeddings: np.ndarray,
    label_embeddings: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity matrix between queries and labels."""
    query_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
    label_norm = label_embeddings / (np.linalg.norm(label_embeddings, axis=1, keepdims=True) + 1e-8)
    return np.matmul(query_norm, label_norm.T)


def run_single_sample_evaluation(
    test_split,
    label_names: List[str],
    text_label_to_label: Dict[str, int],
    model: SentenceTransformer,
    batch_size: int,
    normalize: bool,
) -> Dict:
    """Run single-sample evaluation loop."""
    print("\n🔍 Single-sample evaluation...")

    print(f"📝 Encoding {len(label_names)} label names...")
    label_embeddings = encode_texts(
        model=model,
        texts=label_names,
        batch_size=len(label_names),
        normalize=normalize,
        desc="Encoding labels",
    )
    print(f"✅ Label embeddings shape: {label_embeddings.shape}")

    predictions = []
    latencies: List[float] = []

    total_start = time.time()
    for idx, sample in enumerate(tqdm(test_split, desc="Single-sample eval")):
        text = sample["text"]
        true_text_label = sample["text_label"]
        true_label = int(sample["label"])

        start_time = time.time()
        utterance_embedding = model.encode(
            text,
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        if utterance_embedding.ndim == 1:
            utterance_embedding = utterance_embedding.reshape(1, -1)
        similarity_scores = compute_cosine_similarity(utterance_embedding, label_embeddings)[0]
        end_time = time.time()

        pred_label_idx = int(np.argmax(similarity_scores))
        pred_text_label = label_names[pred_label_idx]
        pred_label = text_label_to_label[pred_text_label]
        confidence = float(similarity_scores[pred_label_idx])

        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)

        predictions.append(
            {
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
            }
        )

    total_time = time.time() - total_start
    avg_latency = float(np.mean(latencies)) if latencies else 0.0

    return {
        "predictions": predictions,
        "average_latency_ms": avg_latency,
        "total_time_s": float(total_time),
    }


def run_batched_evaluation(
    test_split,
    label_names: List[str],
    text_label_to_label: Dict[str, int],
    model: SentenceTransformer,
    batch_size: int,
    normalize: bool,
) -> Dict:
    """Run batched evaluation for the full test set."""
    print("\n🔍 Batched evaluation...")

    print(f"📝 Encoding {len(label_names)} label names...")
    label_embeddings = encode_texts(
        model=model,
        texts=label_names,
        batch_size=len(label_names),
        normalize=normalize,
        desc="Encoding labels",
    )
    print(f"✅ Label embeddings shape: {label_embeddings.shape}")

    test_texts = [sample["text"] for sample in test_split]

    start_time = time.time()
    utterance_embeddings = encode_texts(
        model=model,
        texts=test_texts,
        batch_size=batch_size,
        normalize=normalize,
        desc="Encoding test samples",
    )
    encoding_time = time.time() - start_time

    print(f"✅ Utterance embeddings shape: {utterance_embeddings.shape}")
    print("🔢 Computing cosine similarities...")
    similarity_matrix = compute_cosine_similarity(utterance_embeddings, label_embeddings)
    print(f"✅ Similarity matrix shape: {similarity_matrix.shape}")

    pred_label_indices = np.argmax(similarity_matrix, axis=1)
    per_sample_latency = (encoding_time * 1000) / len(test_split) if test_split else 0.0

    predictions = []
    for idx, sample in enumerate(test_split):
        text = sample["text"]
        true_text_label = sample["text_label"]
        true_label = int(sample["label"])

        pred_label_idx = int(pred_label_indices[idx])
        pred_text_label = label_names[pred_label_idx]
        pred_label = text_label_to_label[pred_text_label]
        confidence = float(similarity_matrix[idx, pred_label_idx])

        predictions.append(
            {
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
            }
        )

    return {
        "predictions": predictions,
        "average_latency_ms": float(per_sample_latency),
        "total_time_s": float(encoding_time),
    }


def save_results(
    single_results: Dict,
    batched_results: Dict,
    output_dir: Path,
    metadata: Dict,
) -> None:
    """Persist evaluation outputs to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(single_results["predictions"])
    correct = sum(1 for pred in single_results["predictions"] if pred["correct"])
    accuracy = correct / total if total else 0.0

    final_payload = {
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
            "encoder_path": metadata["encoder_path"],
            "device": metadata["device"],
            "normalize_embeddings": metadata["normalize_embeddings"],
            "fp16": metadata["fp16"],
        },
        "predictions": single_results["predictions"],
    }

    results_file = output_dir / "evaluation_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved results to: {results_file}")

    batched_timing_file = output_dir / "batched_timing_details.json"
    with open(batched_timing_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "batch_size": metadata["batch_size"],
                "total_time_s": round(batched_results["total_time_s"], 2),
                "average_per_sample_latency_ms": round(batched_results["average_latency_ms"], 2),
            },
            f,
            indent=2,
        )
    print(f"💾 Saved batched timing details to: {batched_timing_file}")

    error_samples = [pred for pred in single_results["predictions"] if not pred["correct"]]
    if error_samples:
        error_file = output_dir / "error_samples.json"
        with open(error_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "num_errors": len(error_samples),
                    "error_rate": round(len(error_samples) / total, 4) if total else 0.0,
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
        f.write(f"Encoder: {metadata['encoder_path']}\n\n")
        f.write(f"Total Samples: {total}\n")
        f.write(f"Correct: {correct}\n")
        f.write(f"Errors: {total - correct}\n")
        f.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
        f.write("-" * 60 + "\n")
        f.write("LATENCY METRICS\n")
        f.write("-" * 60 + "\n\n")
        f.write("Single-Sample Inference:\n")
        f.write(f"  Average Latency: {single_results['average_latency_ms']:.2f} ms\n")
        f.write(f"  Total Time: {single_results['total_time_s']:.2f} s\n\n")
        f.write(f"Batched Inference (batch_size={metadata['batch_size']}):\n")
        f.write(f"  Average Per-Sample Latency: {batched_results['average_latency_ms']:.2f} ms\n")
        f.write(f"  Total Time: {batched_results['total_time_s']:.2f} s\n")
        if batched_results["total_time_s"] > 0:
            speedup = single_results["total_time_s"] / batched_results["total_time_s"]
            f.write(f"  Speedup: {speedup:.2f}x\n\n")
    print(f"💾 Saved summary to: {summary_file}")


def main() -> None:
    args = parse_args()

    set_seed(args.seed)
    device = args.device
    print(f"🖥️  Using device: {device}")

    print(f"📂 Loading dataset: {args.dataset}")
    loader_fn = get_loader_by_name(args.dataset)
    if args.dataset == "clinc_oos":
        raw_dataset, stats, text_label_to_label, label_to_text_label = loader_fn(  # type: ignore
            data_dir=args.dataset_dir,
            version=args.clinc_version,
            return_dicts=True,
        )
    else:
        raw_dataset, stats, text_label_to_label, label_to_text_label = loader_fn(  # type: ignore
            data_dir=args.dataset_dir,
            return_dicts=True,
        )

    test_split = raw_dataset["test"]
    label_names = list(text_label_to_label.keys())
    num_labels = len(label_names)

    print(f"✅ Loaded {len(test_split)} test samples")
    print(f"✅ Number of labels: {num_labels}")
    print(f"✅ Label names: {label_names[:5]}... (showing first 5)")

    print(f"\n🔤 Loading SentenceTransformer from: {args.encoder_path}")
    model = SentenceTransformer(args.encoder_path, device=device)
    model.max_seq_length = args.max_length
    if args.fp16:
        try:
            model = model.half()
            print("⚙️  Enabled half-precision inference")
        except AttributeError:
            print("⚠️  Half-precision not supported by this model; continuing in full precision")

    single_results = run_single_sample_evaluation(
        test_split=test_split,
        label_names=label_names,
        text_label_to_label=text_label_to_label,
        model=model,
        batch_size=args.batch_size,
        normalize=args.normalize_embeddings,
    )

    batched_results = run_batched_evaluation(
        test_split=test_split,
        label_names=label_names,
        text_label_to_label=text_label_to_label,
        model=model,
        batch_size=args.batch_size,
        normalize=args.normalize_embeddings,
    )

    correct = sum(1 for pred in single_results["predictions"] if pred["correct"])
    total = len(single_results["predictions"])
    accuracy = correct / total if total else 0.0

    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Correct: {correct}/{total}")
    print(f"Errors: {total - correct}")
    print("=" * 60 + "\n")

    metadata = {
        "method": args.method,
        "dataset_name": args.dataset,
        "num_classes": num_labels,
        "batch_size": args.batch_size,
        "encoder_path": args.encoder_path,
        "device": device,
        "normalize_embeddings": args.normalize_embeddings,
        "fp16": args.fp16,
    }

    output_dir = Path(args.output_dir)
    save_results(single_results, batched_results, output_dir, metadata)

    print("\n✅ Evaluation complete!")
    print(f"📁 Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
