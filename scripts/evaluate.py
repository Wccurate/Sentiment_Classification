#!/usr/bin/env python
import argparse
from src.utils.logging import get_logger
from src.evaluation.classical import ClassicalEvalConfig, evaluate_classical


def main():
    parser = argparse.ArgumentParser(description="Unified evaluation entry for sentiment classification")
    parser.add_argument("--method", required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model artifacts")
    parser.add_argument("--data_csv", type=str, required=True, help="Dataset CSV for evaluation")
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--output_dir", type=str, default="outputs/eval")
    args = parser.parse_args()

    logger = get_logger()
    logger.info(f"Evaluating method={args.method} checkpoint={args.checkpoint}")

    if args.method in {"tfidf_svm", "w2v_svm", "sbert_svm"}:
        evaluate_classical(ClassicalEvalConfig(
            method=args.method,
            checkpoint_dir=args.checkpoint,
            data_csv=args.data_csv,
            text_col=args.text_col,
            label_col=args.label_col,
            output_dir=args.output_dir,
        ))
        return

    logger.info("Selected method not yet wired for evaluation.")


if __name__ == "__main__":
    main()
