#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Utility functions to compute classification metrics and summaries."""

from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    num_classes: Optional[int] = None,
) -> Dict:
    """Compute standard classification metrics."""
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1 (Micro & Macro)
    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    
    recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    return {
        "accuracy": float(accuracy),
        "precision_micro": float(precision_micro),
        "precision_macro": float(precision_macro),
        "recall_micro": float(recall_micro),
        "recall_macro": float(recall_macro),
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro),
        "confusion_matrix": cm.tolist(),
        "precision_per_class": precision_per_class.tolist(),
        "recall_per_class": recall_per_class.tolist(),
        "f1_per_class": f1_per_class.tolist(),
    }


def compute_topk_accuracy(
    y_true: List[int],
    y_probs: List[List[float]],
    k: int = 5,
) -> float:
    """Compute top-k accuracy from probability outputs."""
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    
    # Take top-k predictions
    topk_preds = np.argsort(y_probs, axis=1)[:, -k:]
    
    # Check membership of true labels in top-k set
    correct = np.array([y_true[i] in topk_preds[i] for i in range(len(y_true))])
    
    return float(correct.mean())


def compute_latency_stats(latencies: List[float]) -> Dict:
    """Compute latency stats from per-sample timings (seconds -> ms)."""
    if not latencies:
        return {
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
    
    latencies_ms = np.array(latencies) * 1000  # convert to milliseconds
    
    return {
        "mean_ms": float(np.mean(latencies_ms)),
        "median_ms": float(np.median(latencies_ms)),
        "std_ms": float(np.std(latencies_ms)),
        "min_ms": float(np.min(latencies_ms)),
        "max_ms": float(np.max(latencies_ms)),
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p90_ms": float(np.percentile(latencies_ms, 90)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
    }


def get_classification_report_dict(
    y_true: List[int],
    y_pred: List[int],
    target_names: Optional[List[str]] = None,
) -> Dict:
    """Return sklearn classification_report as a dict."""
    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    return report


def format_metrics_report(metrics: Dict, label_names: Optional[List[str]] = None) -> str:
    """Format metrics into a human-readable report."""
    lines = []
    lines.append("=" * 70)
    lines.append("EVALUATION METRICS REPORT")
    lines.append("=" * 70)
    lines.append("")
    
    # Prediction-only mode (no ground truth)
    is_prediction_only = metrics.get("mode") == "prediction_only"
    
    if is_prediction_only:
        lines.append("MODE: Prediction Only (no labels)")
        lines.append("-" * 70)
        lines.append(metrics.get("note", "Test set has no labels; predictions only"))
        lines.append("")
    else:
        # Overall metrics (only when labels exist)
        lines.append("OVERALL METRICS")
        lines.append("-" * 70)
        if "accuracy" in metrics:
            lines.append(f"Accuracy:          {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
            lines.append("")
        
        if "precision_micro" in metrics:
            lines.append(f"Precision (Micro): {metrics['precision_micro']:.4f}")
            lines.append(f"Precision (Macro): {metrics.get('precision_macro', 0):.4f}")
            lines.append("")
        
        if "recall_micro" in metrics:
            lines.append(f"Recall (Micro):    {metrics['recall_micro']:.4f}")
            lines.append(f"Recall (Macro):    {metrics.get('recall_macro', 0):.4f}")
            lines.append("")
        
        if "f1_micro" in metrics:
            lines.append(f"F1-Score (Micro):  {metrics['f1_micro']:.4f}")
            lines.append(f"F1-Score (Macro):  {metrics.get('f1_macro', 0):.4f}")
            lines.append("")
        
        # Top-K accuracy if available
        for key in metrics:
            if key.startswith("top") and key.endswith("_accuracy"):
                k = key.replace("top", "").replace("_accuracy", "")
                lines.append(f"Top-{k} Accuracy:  {metrics[key]:.4f} ({metrics[key]*100:.2f}%)")
                lines.append("")
                break
    
    # Latency stats if available
    if 'latency_stats' in metrics and metrics['latency_stats']['mean_ms'] > 0:
        lines.append("-" * 70)
        lines.append("LATENCY STATISTICS (milliseconds)")
        lines.append("-" * 70)
        lat = metrics['latency_stats']
        lines.append(f"Mean:      {lat['mean_ms']:.2f} ms")
        lines.append(f"Median:    {lat['median_ms']:.2f} ms")
        lines.append(f"Std Dev:   {lat['std_ms']:.2f} ms")
        lines.append(f"Min:       {lat['min_ms']:.2f} ms")
        lines.append(f"Max:       {lat['max_ms']:.2f} ms")
        lines.append(f"P50:       {lat['p50_ms']:.2f} ms")
        lines.append(f"P90:       {lat['p90_ms']:.2f} ms")
        lines.append(f"P95:       {lat['p95_ms']:.2f} ms")
        lines.append(f"P99:       {lat['p99_ms']:.2f} ms")
        lines.append("")
    
    # Per-class metrics
    if label_names and 'precision_per_class' in metrics:
        lines.append("-" * 70)
        lines.append("PER-CLASS METRICS")
        lines.append("-" * 70)
        lines.append(f"{'Class':<20} {'Precision':>12} {'Recall':>12} {'F1-Score':>12}")
        lines.append("-" * 70)
        
        for i, name in enumerate(label_names):
            if i < len(metrics['precision_per_class']):
                prec = metrics['precision_per_class'][i]
                rec = metrics['recall_per_class'][i]
                f1 = metrics['f1_per_class'][i]
                lines.append(f"{name:<20} {prec:>12.4f} {rec:>12.4f} {f1:>12.4f}")
        lines.append("")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)
