#!/usr/bin/env python3
"""
Convert evaluation_results.json to submission.csv format

Output format:
- Column 1: id (1-based)
- Column 2: label (0 or 1 prediction)
- Row 1 header: id,label
"""

import json
import csv
import argparse
from pathlib import Path


def convert_evaluation_to_submission(
    json_path: str,
    output_path: str = "submission.csv"
) -> None:
    """
    Convert evaluation results JSON to competition submission CSV
    
    Args:
        json_path: path to evaluation_results.json
        output_path: output submission.csv path
    """
    # Read JSON file
    print(f"Reading evaluation results: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract predictions
    predictions = data.get('predictions', [])
    
    if not predictions:
        print("Warning: No predictions found in JSON")
        return
    
    print(f"Found {len(predictions)} predictions")
    
    # Write CSV file
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['id', 'label'])
        
        # Write predictions, id starts from 1
        for idx, pred in enumerate(predictions, start=1):
            pred_label = pred['pred_label']
            writer.writerow([idx, pred_label])
    
    print(f"Submission written: {output_path}")
    print(f"Total {len(predictions)} predictions")
    
    # Print distribution
    label_counts = {}
    for pred in predictions:
        label = pred['pred_label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        percentage = count / len(predictions) * 100
        print(f"  label {label}: {count} ({percentage:.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert evaluation_results.json to submission.csv"
    )
    parser.add_argument(
        'json_path',
        type=str,
        help='Path to evaluation_results.json'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='submission.csv',
        help='Output submission.csv path (default: submission.csv)'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.json_path).exists():
        print(f"Error: File not found: {args.json_path}")
        return
    
    # Execute conversion
    convert_evaluation_to_submission(args.json_path, args.output)


if __name__ == '__main__':
    main()
