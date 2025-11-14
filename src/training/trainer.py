from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from src.data.datasets import Example, load_csv_dataset, split_dataset
from src.features.tfidf import TfidfConfig, TfidfFeaturizer
from src.features.word2vec import W2VConfig, Word2VecAverager
from src.features.sentence_bert import SBERTConfig, SentenceBertEncoder
from src.models.svm import SVMClassifier, SVMConfig
from src.utils.logging import get_logger


@dataclass
class ClassicalRunConfig:
    train_csv: str
    text_col: str = "text"
    label_col: str = "label"
    val_ratio: float = 0.1
    seed: int = 42
    # TF-IDF
    max_features: int = 50000
    ngram_min: int = 1
    ngram_max: int = 2
    pca_components: int | None = None
    # SVM
    C: float = 1.0
    # W2V
    w2v_path: str | None = None
    w2v_binary: bool = True
    # SBERT
    sbert_name: str = "BAAI/bge-small-en-v1.5"
    device: str | None = None


def _labels_or_fail(data: List[Example]) -> np.ndarray:
    values = []
    for ex in data:
        if ex.label is None:
            raise ValueError("Dataset entry missing label but labels required for training/eval")
        values.append(ex.label)
    return np.array(values)


def run_tfidf_svm(cfg: ClassicalRunConfig, output_dir: str = "outputs/tfidf_svm"):
    logger = get_logger()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading dataset from {cfg.train_csv}")
    data: List[Example] = load_csv_dataset(cfg.train_csv, cfg.text_col, cfg.label_col)
    train, val = split_dataset(data, cfg.val_ratio, cfg.seed)
    logger.info(f"Train={len(train)}, Val={len(val)}")

    tfidf = TfidfFeaturizer(TfidfConfig(
        max_features=cfg.max_features,
        ngram_range=(cfg.ngram_min, cfg.ngram_max),
        pca_components=cfg.pca_components,
    ))
    X_train = tfidf.fit_transform(train)
    y_train = _labels_or_fail(train)
    X_val = tfidf.transform(val)
    y_val = _labels_or_fail(val)

    svm = SVMClassifier(SVMConfig(C=cfg.C))
    svm.fit(X_train, y_train)
    preds = svm.predict(X_val)
    acc = float((preds == y_val).mean()) if len(y_val) else 0.0
    logger.info(f"Validation accuracy: {acc:.4f}")

    # Save artifacts
    import joblib
    joblib.dump(tfidf, str(Path(output_dir) / "tfidf_featurizer.joblib"))
    svm.save(str(Path(output_dir) / "svm_model.joblib"))
    with open(Path(output_dir) / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"val_accuracy={acc:.6f}\n")


def run_w2v_svm(cfg: ClassicalRunConfig, output_dir: str = "outputs/w2v_svm"):
    logger = get_logger()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    assert cfg.w2v_path, "w2v_path must be provided for w2v_svm"
    data: List[Example] = load_csv_dataset(cfg.train_csv, cfg.text_col, cfg.label_col)
    train, val = split_dataset(data, cfg.val_ratio, cfg.seed)
    logger.info(f"Train={len(train)}, Val={len(val)}")

    w2v = Word2VecAverager(W2VConfig(model_path=cfg.w2v_path, binary=cfg.w2v_binary))
    X_train = w2v.fit_transform(train)
    y_train = _labels_or_fail(train)
    X_val = w2v.transform(val)
    y_val = _labels_or_fail(val)

    svm = SVMClassifier(SVMConfig(C=cfg.C))
    svm.fit(X_train, y_train)
    preds = svm.predict(X_val)
    acc = float((preds == y_val).mean()) if len(y_val) else 0.0
    logger.info(f"Validation accuracy: {acc:.4f}")

    import joblib
    joblib.dump(w2v, str(Path(output_dir) / "w2v_averager.joblib"))
    svm.save(str(Path(output_dir) / "svm_model.joblib"))
    with open(Path(output_dir) / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"val_accuracy={acc:.6f}\n")


def run_sbert_svm(cfg: ClassicalRunConfig, output_dir: str = "outputs/sbert_svm"):
    logger = get_logger()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    data: List[Example] = load_csv_dataset(cfg.train_csv, cfg.text_col, cfg.label_col)
    train, val = split_dataset(data, cfg.val_ratio, cfg.seed)
    logger.info(f"Train={len(train)}, Val={len(val)}")

    sbert = SentenceBertEncoder(SBERTConfig(model_name=cfg.sbert_name, device=cfg.device))
    X_train = sbert.fit_transform(train)
    y_train = _labels_or_fail(train)
    X_val = sbert.transform(val)
    y_val = _labels_or_fail(val)

    svm = SVMClassifier(SVMConfig(C=cfg.C))
    svm.fit(X_train, y_train)
    preds = svm.predict(X_val)
    acc = float((preds == y_val).mean()) if len(y_val) else 0.0
    logger.info(f"Validation accuracy: {acc:.4f}")

    import joblib
    joblib.dump(sbert, str(Path(output_dir) / "sbert_encoder.joblib"))
    svm.save(str(Path(output_dir) / "svm_model.joblib"))
    with open(Path(output_dir) / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"val_accuracy={acc:.6f}\n")
