from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import joblib
import numpy as np

from src.data.datasets import load_csv_dataset
from src.utils.logging import get_logger


@dataclass
class ClassicalEvalConfig:
    method: str
    checkpoint_dir: str
    data_csv: str
    text_col: str = "text"
    label_col: str | None = "label"
    output_dir: str = "outputs/eval"


def evaluate_classical(cfg: ClassicalEvalConfig) -> Dict:
    logger = get_logger()
    ckpt = Path(cfg.checkpoint_dir)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    data = load_csv_dataset(cfg.data_csv, cfg.text_col, cfg.label_col)
    texts = [ex.text for ex in data]
    labels = [ex.label for ex in data]

    if cfg.method == "tfidf_svm":
        featurizer = joblib.load(ckpt / "tfidf_featurizer.joblib")
    elif cfg.method == "w2v_svm":
        featurizer = joblib.load(ckpt / "w2v_averager.joblib")
    elif cfg.method == "sbert_svm":
        featurizer = joblib.load(ckpt / "sbert_encoder.joblib")
    else:
        raise ValueError(f"Unsupported classical method: {cfg.method}")

    svm = joblib.load(ckpt / "svm_model.joblib")
    features = featurizer.transform(data)
    preds = svm.predict(features)

    metrics = {}
    if all(label is not None for label in labels):
        y_true = np.array(labels)
        metrics["accuracy"] = float((preds == y_true).mean()) if len(y_true) else 0.0
        logger.info(f"Evaluation accuracy: {metrics['accuracy']:.4f}")
    else:
        logger.info("No ground-truth labels detected; skipping accuracy computation")

    output_file = Path(cfg.output_dir) / f"predictions_{cfg.method}.csv"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("id,label\n")
        for idx, pred in enumerate(preds, start=1):
            f.write(f"{idx},{pred}\n")
    logger.info(f"Predictions written to {output_file}")

    metrics_file = Path(cfg.output_dir) / f"metrics_{cfg.method}.json"
    import json
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_file}")

    return metrics

