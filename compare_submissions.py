#!/usr/bin/env python3
"""
Compare label differences between two submissions

- sample_submission.csv: text column `reviews`, label column `sentiments`
- submission.csv: 1-based id column `id`, label column `label`
"""

import csv
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import json


def load_sample_submission(csv_path: str) -> List[Dict]:
    """
    Load sample_submission.csv (with text)
    
    Returns:
        List of dict with 'text' and 'label' keys
    """
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            # sample_submission.csv has columns: reviews, sentiments
            text = row.get('reviews', '')
            label = int(row.get('sentiments', -1))
            data.append({
                'id': idx,
                'text': text,
                'label': label
            })
    return data


def load_prediction_submission(csv_path: str) -> List[Dict]:
    """
    Load submission.csv (id + label)
    
    Returns:
        List of dict with 'id' and 'label' keys
    """
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'id': int(row['id']),
                'label': int(row['label'])
            })
    return data


def compare_submissions(
    sample_data: List[Dict],
    pred_data: List[Dict],
    output_dir: str = "."
) -> Dict:
    """
    Compare two submissions and compute metrics
    
    Args:
        sample_data: ground truth rows (text + label)
        pred_data: predicted rows (id + label)
        output_dir: output directory
        
    Returns:
        stats dict
    """
    # Check size consistency
    if len(sample_data) != len(pred_data):
        print(f"Warning: size mismatch")
        print(f"   sample_submission.csv: {len(sample_data)} rows")
        print(f"   submission.csv: {len(pred_data)} rows")
    
    # Build prediction map
    pred_dict = {item['id']: item['label'] for item in pred_data}
    
    # Stats
    total = len(sample_data)
    correct = 0
    wrong = 0
    
    # Confusion matrix
    confusion = {
        'TP': 0,
        'TN': 0,
        'FP': 0,
        'FN': 0
    }
    
    # Misclassified examples
    error_samples = []
    
    # Iterate and compare
    for sample in sample_data:
        idx = sample['id']
        true_label = sample['label']
        text = sample['text']
        
        if idx not in pred_dict:
            print(f"Warning: ID {idx} not found in predictions")
            continue
        
        pred_label = pred_dict[idx]
        
        # Accuracy
        if true_label == pred_label:
            correct += 1
        else:
            wrong += 1
            error_samples.append({
                'id': idx,
                'text': text,
                'true_label': true_label,
                'pred_label': pred_label
            })
        
        # Update confusion
        if true_label == 1 and pred_label == 1:
            confusion['TP'] += 1
        elif true_label == 0 and pred_label == 0:
            confusion['TN'] += 1
        elif true_label == 0 and pred_label == 1:
            confusion['FP'] += 1
        elif true_label == 1 and pred_label == 0:
            confusion['FN'] += 1
    
    # Compute metrics
    accuracy = correct / total if total > 0 else 0
    precision = confusion['TP'] / (confusion['TP'] + confusion['FP']) if (confusion['TP'] + confusion['FP']) > 0 else 0
    recall = confusion['TP'] / (confusion['TP'] + confusion['FN']) if (confusion['TP'] + confusion['FN']) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Print summary
    print("\n" + "="*60)
    print("Metrics on predictions vs truth")
    print("="*60)
    print(f"\nTotal: {total}")
    print(f"Correct: {correct} ({accuracy*100:.2f}%)")
    print(f"Wrong: {wrong} ({(1-accuracy)*100:.2f}%)")
    
    print(f"\nConfusion matrix:")
    print(f"              Pred 0    Pred 1")
    print(f"  True 0        {confusion['TN']:6d}     {confusion['FP']:6d}")
    print(f"  True 1        {confusion['FN']:6d}     {confusion['TP']:6d}")
    
    print(f"\nMetrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Label distribution
    sample_label_dist = {}
    pred_label_dist = {}
    for sample in sample_data:
        label = sample['label']
        sample_label_dist[label] = sample_label_dist.get(label, 0) + 1
    
    for pred in pred_data:
        label = pred['label']
        pred_label_dist[label] = pred_label_dist.get(label, 0) + 1
    
    print(f"\nLabel distribution:")
    print(f"  Truth - 0: {sample_label_dist.get(0, 0)} ({sample_label_dist.get(0, 0)/total*100:.2f}%)")
    print(f"  Truth - 1: {sample_label_dist.get(1, 0)} ({sample_label_dist.get(1, 0)/total*100:.2f}%)")
    print(f"  Pred  - 0: {pred_label_dist.get(0, 0)} ({pred_label_dist.get(0, 0)/total*100:.2f}%)")
    print(f"  Pred  - 1: {pred_label_dist.get(1, 0)} ({pred_label_dist.get(1, 0)/total*100:.2f}%)")
    
    # Save mismatched samples
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Reformat mismatch items
    differences = []
    for err in error_samples:
        differences.append({
            'id': err['id'],
            'text': err['text'],
            'sample_submission_label': err['true_label'],
            'submission_label': err['pred_label']
        })
    
    difference_file = output_path / "label_differences.json"
    with open(difference_file, 'w', encoding='utf-8') as f:
        json.dump(differences, f, indent=2, ensure_ascii=False)
    
    print(f"\nMismatched samples:")
    print(f"  Count: {len(differences)}")
    print(f"  Saved: {difference_file}")
    
    # Show first 10 mismatches
    if differences:
        print(f"\nPreview first 10 mismatches:")
        print("-" * 60)
        for i, diff in enumerate(differences[:10], 1):
            text_preview = diff['text'][:80] + "..." if len(diff['text']) > 80 else diff['text']
            print(f"{i}. ID={diff['id']}")
            print(f"   sample_submission: {diff['sample_submission_label']}")
            print(f"   submission:        {diff['submission_label']}")
            print(f"   text: {text_preview}")
            print()
    
    print("="*60)
    
    return {
        'total_samples': total,
        'correct_predictions': correct,
        'wrong_predictions': wrong,
        'accuracy': accuracy
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare labels between sample_submission.csv and submission.csv"
    )
    parser.add_argument(
        'sample_csv',
        type=str,
        help='Path to sample_submission.csv (text + true label)'
    )
    parser.add_argument(
        'pred_csv',
        type=str,
        help='Path to submission.csv (id + predicted label)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='.',
        help='Output directory (default: current)'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.sample_csv).exists():
        print(f"Error: File not found: {args.sample_csv}")
        return
    
    if not Path(args.pred_csv).exists():
        print(f"Error: File not found: {args.pred_csv}")
        return
    
    # Load files
    print("Loading data...")
    sample_data = load_sample_submission(args.sample_csv)
    pred_data = load_prediction_submission(args.pred_csv)
    
    print(f"Loaded sample_submission.csv: {len(sample_data)} rows")
    print(f"Loaded submission.csv: {len(pred_data)} rows")
    
    # Compare
    compare_submissions(sample_data, pred_data, args.output)


if __name__ == '__main__':
    main()
