from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

import numpy as np
from tqdm.auto import tqdm
from openai import OpenAI

# Dataset loaders
from data.loaders.load_raw_data import (
    load_raw_atis,
    load_raw_hwu64,
    load_raw_snips,
    load_raw_clinc_oos,
    build_label_mappings,
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
        description="LLM-based intent recognition evaluation"
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
    
    # LLM arguments
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model name (e.g., gpt-4o-mini, gpt-4, deepseek-chat)"
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default="https://api.openai.com/v1",
        help="OpenAI-compatible API base URL"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="API key for authentication"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 for deterministic)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum tokens in response"
    )
    
    # Prompting strategy
    parser.add_argument(
        "--mode",
        type=str,
        default="zero-shot",
        choices=["zero-shot", "few-shot"],
        help="Prompting mode"
    )
    parser.add_argument(
        "--num-shots",
        type=int,
        default=5,
        help="Number of examples for few-shot (ignored for zero-shot)"
    )
    parser.add_argument(
        "--shot-selection",
        type=str,
        default="random",
        choices=["random", "balanced"],
        help="How to select few-shot examples"
    )
    
    # Evaluation settings
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of test samples (None for all)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--retry-limit",
        type=int,
        default=3,
        help="Number of retries for API calls"
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=1.0,
        help="Delay between retries (seconds)"
    )
    
    return parser.parse_args()


def build_zero_shot_prompt(text: str, label_names: List[str]) -> str:
    """Build a zero-shot prompt for LLM classification."""
    label_list_str = "\n".join([f"- {label}" for label in label_names])
    
    prompt = f"""You are an intent classification system. Given a user utterance, classify it into one of the predefined intent categories.

Available Intent Categories:
{label_list_str}

User Utterance: "{text}"

Instructions:
1. Analyze the user's utterance carefully
2. Select the most appropriate intent category from the list above
3. Respond with ONLY the intent category name, exactly as listed above
4. Do not include any explanation or additional text

Your Answer:"""
    
    return prompt


def build_few_shot_prompt(
    text: str,
    label_names: List[str],
    examples: List[Dict],
) -> str:
    """Build a few-shot prompt with labeled examples."""
    label_list_str = "\n".join([f"- {label}" for label in label_names])
    
    # Assemble few-shot examples
    examples_str = ""
    for i, ex in enumerate(examples, 1):
        examples_str += f'\nExample {i}:\nUtterance: "{ex["text"]}"\nIntent: {ex["text_label"]}\n'
    
    prompt = f"""You are an intent classification system. Given a user utterance, classify it into one of the predefined intent categories.

Available Intent Categories:
{label_list_str}

Here are some examples:
{examples_str}

Now classify this utterance:
User Utterance: "{text}"

Instructions:
1. Analyze the user's utterance carefully
2. Select the most appropriate intent category from the list above
3. Respond with ONLY the intent category name, exactly as listed above
4. Do not include any explanation or additional text

Your Answer:"""
    
    return prompt


def select_few_shot_examples(
    train_split,
    num_shots: int,
    selection_mode: str,
    seed: int,
) -> List[Dict]:
    """Select few-shot examples from the training split."""
    random.seed(seed)
    np.random.seed(seed)
    
    if selection_mode == "random":
        # Random sampling
        indices = random.sample(range(len(train_split)), min(num_shots, len(train_split)))
        examples = [train_split[i] for i in indices]
    
    elif selection_mode == "balanced":
        # Sample evenly across labels
        label_to_indices = {}
        for idx, sample in enumerate(train_split):
            label = sample["text_label"]
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)
        
        # Determine allocation per label
        num_labels = len(label_to_indices)
        samples_per_label = max(1, num_shots // num_labels)
        
        examples = []
        for label, indices in label_to_indices.items():
            selected = random.sample(indices, min(samples_per_label, len(indices)))
            examples.extend([train_split[i] for i in selected])
            
            if len(examples) >= num_shots:
                break
        
        # Randomly add remaining samples if needed
        if len(examples) < num_shots:
            remaining_indices = [
                i for i in range(len(train_split))
                if i not in [train_split.index(ex) for ex in examples]
            ]
            additional = random.sample(
                remaining_indices,
                min(num_shots - len(examples), len(remaining_indices))
            )
            examples.extend([train_split[i] for i in additional])
        
        # Trim to requested count
        examples = examples[:num_shots]
    
    else:
        raise ValueError(f"Unknown selection_mode: {selection_mode}")
    
    return examples


def extract_intent_from_response(
    response: str,
    label_names: List[str],
    text_label_to_label: Dict[str, int],
) -> Tuple[Optional[str], Optional[int]]:
    """Extract an intent label from an LLM response string."""
    if not response or not response.strip():
        return None, None
    
    # Normalize response
    cleaned = response.strip().lower()
    
    # Strategy 1: exact match (case-insensitive)
    for label in label_names:
        if label.lower() == cleaned:
            return label, text_label_to_label.get(label)
    
    # Strategy 2: substring match (Intent: xxx, etc.)
    for label in label_names:
        if label.lower() in cleaned:
            return label, text_label_to_label.get(label)
    
    # Strategy 3: regex extraction for common patterns
    patterns = [
        r'(?:intent|answer|category|class|label):\s*["\']?([^"\'\n]+)["\']?',
        r'^["\']?([^"\'\n]+)["\']?$',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            # Retry matching after regex extraction
            for label in label_names:
                if label.lower() == extracted or label.lower() in extracted:
                    return label, text_label_to_label.get(label)
    
    # Strategy 4: Partial match via simple overlap
    best_match = None
    best_score = 0
    
    for label in label_names:
        # Simple overlap-based similarity
        label_lower = label.lower()
        if label_lower in cleaned or cleaned in label_lower:
            score = len(label_lower) / (len(cleaned) + 1)
            if score > best_score:
                best_score = score
                best_match = label
    
    if best_match and best_score > 0.3:
        return best_match, text_label_to_label.get(best_match)
    
    return None, None


def call_llm_api(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    retry_limit: int,
    retry_delay: float,
) -> Tuple[Optional[str], float, bool]:
    """Call the LLM API with retries and return the response."""
    for attempt in range(retry_limit):
        try:
            start_time = time.time()
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            end_time = time.time()
            latency = end_time - start_time
            
            # Extract response text
            if response.choices and len(response.choices) > 0:
                answer = response.choices[0].message.content
                return answer, latency, True
            else:
                return None, latency, False
        
        except Exception as e:
            print(f"⚠️  API call failed (attempt {attempt + 1}/{retry_limit}): {e}")
            if attempt < retry_limit - 1:
                time.sleep(retry_delay)
            else:
                return None, 0.0, False
    
    return None, 0.0, False


def run_evaluation(
    client: OpenAI,
    args: argparse.Namespace,
    test_split,
    train_split,
    label_names: List[str],
    text_label_to_label: Dict[str, int],
    label_to_text_label: Dict[int, str],
) -> Dict:
    """Run the full evaluation loop and return metrics."""
    predictions = []
    latencies = []
    
    # Prepare few-shot examples if needed
    few_shot_examples = None
    if args.mode == "few-shot":
        print(f"🎯 Selecting {args.num_shots} few-shot examples (mode={args.shot_selection})...")
        few_shot_examples = select_few_shot_examples(
            train_split,
            num_shots=args.num_shots,
            selection_mode=args.shot_selection,
            seed=args.seed,
        )
        print(f"✅ Selected {len(few_shot_examples)} examples")
    
    # Optional subsampling
    test_samples = list(test_split)
    if args.max_samples is not None and args.max_samples < len(test_samples):
        random.seed(args.seed)
        test_samples = random.sample(test_samples, args.max_samples)
        print(f"📊 Limiting evaluation to {args.max_samples} samples")
    
    print(f"\n🚀 Evaluating {len(test_samples)} samples...")
    print(f"📝 Mode: {args.mode}")
    print(f"🤖 Model: {args.model}\n")
    
    total_start = time.time()
    
    for idx, sample in enumerate(tqdm(test_samples, desc="LLM Evaluation")):
        text = sample["text"]
        true_text_label = sample["text_label"]
        true_label = sample["label"]
        
        # Build prompt
        if args.mode == "zero-shot":
            prompt = build_zero_shot_prompt(text, label_names)
        else:  # few-shot
            prompt = build_few_shot_prompt(text, label_names, few_shot_examples)
        
        # Call API
        response, latency, success = call_llm_api(
            client=client,
            model=args.model,
            prompt=prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            retry_limit=args.retry_limit,
            retry_delay=args.retry_delay,
        )
        
        latencies.append(latency)
        
        # Extract intent
        if success and response:
            pred_text_label, pred_label = extract_intent_from_response(
                response, label_names, text_label_to_label
            )
        else:
            pred_text_label = None
            pred_label = None
        
        # Store result
        prediction = {
            "id": idx,
            "text": text,
            "true_label": int(true_label),
            "true_text_label": true_text_label,
            "pred_label": int(pred_label) if pred_label is not None else None,
            "pred_text_label": pred_text_label,
            "raw_response": response if response else "",
            "correct": (pred_label == true_label) if pred_label is not None else False,
            "latency_s": latency,
            "api_success": success,
        }
        
        predictions.append(prediction)
    
    total_time = time.time() - total_start
    
    return {
        "predictions": predictions,
        "total_time_s": total_time,
        "average_latency_s": np.mean(latencies) if latencies else 0.0,
    }


def save_results(
    results: Dict,
    output_dir: Path,
    metadata: Dict,
):
    """Save evaluation results to JSON, plus auxiliary files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    predictions = results["predictions"]
    
    # Accuracy stats
    valid_predictions = [p for p in predictions if p["api_success"] and p["pred_label"] is not None]
    correct = sum(1 for p in valid_predictions if p["correct"])
    total = len(predictions)
    valid_total = len(valid_predictions)
    
    accuracy = correct / valid_total if valid_total > 0 else 0.0
    coverage = valid_total / total if total > 0 else 0.0
    
    # Build JSON payload
    result_json = {
        "experiment_metadata": {
            "method": metadata["method"],
            "model": metadata["model"],
            "mode": metadata["mode"],
            "num_shots": metadata.get("num_shots", 0),
            "shot_selection": metadata.get("shot_selection", "N/A"),
            "dataset_name": metadata["dataset_name"],
            "num_samples": total,
            "num_classes": metadata["num_classes"],
            "valid_predictions": valid_total,
            "api_success_rate": coverage,
            "accuracy": round(accuracy, 4),
            "correct_samples": correct,
            "error_samples": valid_total - correct,
            "failed_samples": total - valid_total,
            "average_latency_s": round(results["average_latency_s"], 4),
            "total_time_s": round(results["total_time_s"], 2),
            "api_base": metadata["api_base"],
            "temperature": metadata["temperature"],
        },
        "predictions": predictions,
    }
    
    # Persist summary
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)
    print(f"💾 Results saved to: {results_file}")
    
    # Persist misclassified samples
    error_samples = [p for p in valid_predictions if not p["correct"]]
    if error_samples:
        error_file = output_dir / "error_samples.json"
        with open(error_file, "w", encoding="utf-8") as f:
            json.dump({
                "num_errors": len(error_samples),
                "error_rate": round(len(error_samples) / valid_total, 4) if valid_total > 0 else 0.0,
                "errors": error_samples
            }, f, indent=2, ensure_ascii=False)
        print(f"💾 Error samples saved to: {error_file}")
    
    # Persist API failures
    failed_samples = [p for p in predictions if not p["api_success"] or p["pred_label"] is None]
    if failed_samples:
        failed_file = output_dir / "failed_samples.json"
        with open(failed_file, "w", encoding="utf-8") as f:
            json.dump({
                "num_failed": len(failed_samples),
                "failed_samples": failed_samples
            }, f, indent=2, ensure_ascii=False)
        print(f"💾 Failed samples saved to: {failed_file}")
    
    # Save human-readable summary
    summary_file = output_dir / "summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("LLM EVALUATION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Method: {metadata['method']}\n")
        f.write(f"Model: {metadata['model']}\n")
        f.write(f"Mode: {metadata['mode']}\n")
        if metadata["mode"] == "few-shot":
            f.write(f"Num Shots: {metadata.get('num_shots', 0)}\n")
            f.write(f"Shot Selection: {metadata.get('shot_selection', 'N/A')}\n")
        f.write(f"Dataset: {metadata['dataset_name']}\n\n")
        f.write(f"Total Samples: {total}\n")
        f.write(f"Valid Predictions: {valid_total}\n")
        f.write(f"API Success Rate: {coverage*100:.2f}%\n\n")
        f.write(f"Correct: {correct}\n")
        f.write(f"Errors: {valid_total - correct}\n")
        f.write(f"Accuracy: {accuracy*100:.2f}%\n\n")
        f.write("-" * 60 + "\n")
        f.write("LATENCY METRICS\n")
        f.write("-" * 60 + "\n\n")
        f.write(f"Average Latency: {results['average_latency_s']:.4f} s\n")
        f.write(f"Total Time: {results['total_time_s']:.2f} s\n\n")
    print(f"💾 Summary saved to: {summary_file}")


def main():
    args = parse_args()
    
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize OpenAI client
    print(f"🔗 Connecting to API: {args.api_base}")
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.api_base,
    )
    print(f"✅ API client initialized\n")
    
    # Load dataset
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
    train_split = raw_dataset["train"]
    label_names = stats["label_names"]
    num_labels = stats["num_labels"]
    
    print(f"✅ Loaded {len(test_split)} test samples")
    print(f"✅ Loaded {len(train_split)} train samples")
    print(f"✅ Number of labels: {num_labels}\n")
    
    # Run evaluation
    results = run_evaluation(
        client=client,
        args=args,
        test_split=test_split,
        train_split=train_split,
        label_names=label_names,
        text_label_to_label=text_label_to_label,
        label_to_text_label=label_to_text_label,
    )
    
    # Accuracy summary
    predictions = results["predictions"]
    valid_predictions = [p for p in predictions if p["api_success"] and p["pred_label"] is not None]
    correct = sum(1 for p in valid_predictions if p["correct"])
    valid_total = len(valid_predictions)
    total = len(predictions)
    
    accuracy = correct / valid_total if valid_total > 0 else 0.0
    coverage = valid_total / total if total > 0 else 0.0
    
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total Samples: {total}")
    print(f"Valid Predictions: {valid_total}")
    print(f"API Success Rate: {coverage:.4f} ({coverage*100:.2f}%)")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Correct: {correct}/{valid_total}")
    print(f"Errors: {valid_total - correct}")
    print(f"Failed: {total - valid_total}")
    print("=" * 60 + "\n")
    
    # Persist results
    metadata = {
        "method": f"LLM-{args.mode}",
        "model": args.model,
        "mode": args.mode,
        "num_shots": args.num_shots if args.mode == "few-shot" else 0,
        "shot_selection": args.shot_selection if args.mode == "few-shot" else "N/A",
        "dataset_name": args.dataset,
        "num_classes": num_labels,
        "api_base": args.api_base,
        "temperature": args.temperature,
    }
    
    output_dir = Path(args.output_dir)
    save_results(results, output_dir, metadata)
    
    print("\n✅ Evaluation finished!")
    print(f"📁 Artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
