#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compute detailed metrics from evaluation_results.json outputs."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from evaluation.metrics.compute_metrics import (
    compute_classification_metrics,
    compute_topk_accuracy,
    compute_latency_stats,
    get_classification_report_dict,
    format_metrics_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute detailed metrics from evaluation_results.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to evaluation_results.json"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory (default: alongside input file)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON filename or path (default: detailed_metrics.json)"
    )
    parser.add_argument(
        "--output-txt",
        type=str,
        default=None,
        help="Output text report filename or path (default: metrics_report.txt)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K accuracy value K (default: 5)"
    )
    parser.add_argument(
        "--no-print",
        action="store_true",
        help="Disable console summary"
    )
    
    return parser.parse_args()


def extract_predictions(results: Dict) -> Dict:
    """Extract prediction details from an evaluation JSON payload."""
    predictions = results.get("predictions", [])
    
    if not predictions:
        raise ValueError("No predictions found in results file")
    
    # Determine whether labels are present (prediction-only mode supported)
    metadata = results.get("experiment_metadata", {})
    has_labels = metadata.get("has_labels", True)
    
    # If metadata is ambiguous, inspect the first prediction
    if has_labels and predictions:
        first_pred = predictions[0]
        has_labels = "true_label" in first_pred or "correct" in first_pred
    
    # Filter predictions that include pred_label
    valid_preds = [
        p for p in predictions
        if p.get("pred_label") is not None
    ]
    
    if not valid_preds:
        raise ValueError("No valid predictions with pred_label available")
    
    # Collect labels when available
    if has_labels:
        y_true = [p.get("true_label") for p in valid_preds if "true_label" in p]
        if not y_true:
            has_labels = False  # Override when labels are actually missing
    else:
        y_true = []
    
    y_pred = [p["pred_label"] for p in valid_preds]
    
    # Collect text labels for reporting
    pred_text_labels = [p.get("pred_text_label", "") for p in valid_preds]
    true_text_labels = []
    if has_labels:
        true_text_labels = [p.get("true_text_label", "") for p in valid_preds if "true_text_label" in p]
    
    # Build label-name mapping
    label_names_dict = {}
    if has_labels:
        for p in predictions:
            label = p.get("true_label")
            text_label = p.get("true_text_label")
            if label is not None and text_label:
                label_names_dict[label] = text_label
    else:
        # Prediction-only mode: derive names from predictions
        for p in predictions:
            label = p.get("pred_label")
            text_label = p.get("pred_text_label")
            if label is not None and text_label:
                label_names_dict[label] = text_label
    
    label_names = [label_names_dict[i] for i in sorted(label_names_dict.keys())] if label_names_dict else []
    
    # Latency info
    latencies = []
    if valid_preds and "latency_s" in valid_preds[0]:
        latencies = [p.get("latency_s", 0) for p in valid_preds]
    
    # Probability info
    has_probs = False
    y_probs = None
    if valid_preds:
        has_probs = "probabilities" in valid_preds[0] or "probs" in valid_preds[0] or "similarity_scores" in valid_preds[0]
        if has_probs:
            # Attempt to build probability vectors
            y_probs = []
            for p in valid_preds:
                probs = p.get("probabilities") or p.get("probs") or p.get("similarity_scores")
                if probs:
                    y_probs.append(probs)
            
            if not y_probs:
                has_probs = False
                y_probs = None
    
    return {
        "has_labels": has_labels,
        "y_true": y_true,
        "y_pred": y_pred,
        "true_text_labels": true_text_labels,
        "pred_text_labels": pred_text_labels,
        "label_names": label_names,
        "latencies": latencies,
        "has_probs": has_probs,
        "y_probs": y_probs,
        "num_valid": len(valid_preds),
        "num_total": len(predictions),
    }


def compute_all_metrics(
    data: Dict,
    top_k: int = 5,
) -> Dict:
    """Compute all metrics from extracted prediction data."""
    metrics = {}
    
    # Classification metrics when labels exist
    if data["has_labels"] and data["y_true"]:
        classification_metrics = compute_classification_metrics(
            y_true=data["y_true"],
            y_pred=data["y_pred"],
            num_classes=len(data["label_names"]) if data["label_names"] else max(max(data["y_true"], default=0), max(data["y_pred"], default=0)) + 1,
        )
        metrics.update(classification_metrics)
        
        # Top-K accuracy if probability outputs are available
        if data["has_probs"] and data["y_probs"]:
            try:
                topk_acc = compute_topk_accuracy(
                    y_true=data["y_true"],
                    y_probs=data["y_probs"],
                    k=top_k,
                )
                metrics[f"top{top_k}_accuracy"] = topk_acc
            except Exception as e:
                print(f"⚠️  Failed to compute Top-{top_k} accuracy: {e}")
    else:
        # Prediction-only mode: add basic note
        metrics["mode"] = "prediction_only"
        metrics["note"] = "Prediction-only dataset (no labels)"

    # Latency stats
    if data["latencies"]:
        latency_stats = compute_latency_stats(data["latencies"])
        metrics["latency_stats"] = latency_stats
    else:
        metrics["latency_stats"] = {
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "std_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "p50_ms": 0.0,
            "p90_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
        }
    
    # Sample counts
    metrics["num_samples"] = {
        "total": data["num_total"],
        "valid": data["num_valid"],
        "failed": data["num_total"] - data["num_valid"],
    }
    
    # Detailed classification report when labels exist
    if data["has_labels"] and data["y_true"]:
        try:
            report = get_classification_report_dict(
                y_true=data["y_true"],
                y_pred=data["y_pred"],
                target_names=data["label_names"],
            )
            metrics["classification_report"] = report
        except Exception as e:
            print(f"⚠️  Failed to generate classification report: {e}")
    
    return metrics


def save_metrics(
    metrics: Dict,
    original_results: Dict,
    output_json: Path,
    output_txt: Path,
    label_names: List[str],
    no_print: bool = False,
):
    """Persist metrics and reports to disk."""
    # Merge computed metrics with original metadata
    output_data = {
        "original_metadata": original_results.get("experiment_metadata", {}),
        "computed_metrics": metrics,
    }
    
    # Save JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    if not no_print:
        print(f"💾 Detailed metrics saved to: {output_json}")
    
    # Save text report
    report_text = format_metrics_report(metrics, label_names)
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(report_text)
    if not no_print:
        print(f"💾 Text report saved to: {output_txt}")


def main():
    args = parse_args()
    
    input_path = Path(args.input_file)
    
    if not input_path.exists():
        print(f"❌ Error: file not found: {input_path}")
        return

    if not args.no_print:
        print(f"📂 Loading evaluation results: {input_path}")
    
    # Load results JSON
    with open(input_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    if not args.no_print:
        print(f"✅ Results file loaded\n")
    
    # Extract prediction data
    if not args.no_print:
        print("🔍 Extracting prediction data...")
    data = extract_predictions(results)
    if not args.no_print:
        print(f"✅ Parsed {data['num_valid']}/{data['num_total']} valid predictions")
        if data["has_labels"]:
            print(f"✅ Number of labels: {len(data['label_names'])}")
        else:
            print(f"ℹ️  Prediction mode: unlabeled dataset")
        print()
    
    # Compute metrics
    if not args.no_print:
        print("📊 Computing metrics...")
    metrics = compute_all_metrics(data, top_k=args.top_k)
    if not args.no_print:
        print(f"✅ Metrics computed\n")
    
    # Prepare output paths
    # Priority: --output > --output-dir + filename > input file parent
    if args.output:
        output_json = Path(args.output)
        if not output_json.is_absolute():
            # If relative and --output-dir is provided, resolve against it
            if args.output_dir:
                output_json = Path(args.output_dir) / output_json
            else:
                # Otherwise resolve relative to the input file directory
                output_json = input_path.parent / output_json
    elif args.output_dir:
        output_json = Path(args.output_dir) / "detailed_metrics.json"
    else:
        output_json = input_path.parent / "detailed_metrics.json"
    
    if args.output_txt:
        output_txt = Path(args.output_txt)
        if not output_txt.is_absolute():
            # Same resolution logic for the text report
            if args.output_dir:
                output_txt = Path(args.output_dir) / output_txt
            else:
                # Otherwise use the input file directory
                output_txt = input_path.parent / output_txt
    elif args.output_dir:
        output_txt = Path(args.output_dir) / "metrics_report.txt"
    else:
        output_txt = input_path.parent / "metrics_report.txt"
    
    # Ensure output directories exist
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    
    # Save results
    save_metrics(
        metrics=metrics,
        original_results=results,
        output_json=output_json,
        output_txt=output_txt,
        label_names=data["label_names"],
        no_print=args.no_print,
    )
    
    # Print summary (only when labels exist and printing enabled)
    if not args.no_print:
        print("\n" + "=" * 70)
        print("📊 METRICS SUMMARY")
        print("=" * 70)
        
        if data["has_labels"]:
            print(f"Accuracy:          {metrics.get('accuracy', 0):.4f} ({metrics.get('accuracy', 0)*100:.2f}%)")
            print(f"Precision (Macro): {metrics.get('precision_macro', 0):.4f}")
            print(f"Precision (Micro): {metrics.get('precision_micro', 0):.4f}")
            print(f"Recall (Macro):    {metrics.get('recall_macro', 0):.4f}")   
            print(f"Recall (Micro):    {metrics.get('recall_micro', 0):.4f}")
            print(f"F1-Score (Macro):  {metrics.get('f1_macro', 0):.4f}")
            print(f"F1-Score (Micro):  {metrics.get('f1_micro', 0):.4f}")
            
            if f"top{args.top_k}_accuracy" in metrics:
                topk_acc = metrics[f"top{args.top_k}_accuracy"]
                print(f"Top-{args.top_k} Accuracy:    {topk_acc:.4f} ({topk_acc*100:.2f}%)")
        else:
            print("Mode: prediction-only (no labels)")
            print(f"Samples predicted: {metrics['num_samples']['valid']}/{metrics['num_samples']['total']}")
        
        if metrics["latency_stats"]["mean_ms"] > 0:
            lat = metrics["latency_stats"]
            print(f"\nLatency (Mean):    {lat['mean_ms']:.2f} ms")
            print(f"Latency (Median):  {lat['median_ms']:.2f} ms")
            print(f"Latency (P95):     {lat['p95_ms']:.2f} ms")
        
        print("=" * 70)
        print(f"\n✅ Analysis complete!")
        print(f"📁 Detailed outputs stored in: {output_json.parent}")


if __name__ == "__main__":
    main()
