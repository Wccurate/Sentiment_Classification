from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

from src.data.datasets import Example


@dataclass
class TfidfConfig:
    max_features: Optional[int] = 50000
    ngram_range: Tuple[int, int] = (1, 2)
    pca_components: Optional[int] = None


class TfidfFeaturizer:
    def __init__(self, cfg: TfidfConfig):
        self.cfg = cfg
        self.vectorizer = TfidfVectorizer(
            max_features=cfg.max_features,
            ngram_range=cfg.ngram_range,
        )
        self.pca = PCA(n_components=cfg.pca_components) if cfg.pca_components else None

    def fit_transform(self, data: List[Example]):
        texts = [ex.text for ex in data]
        X = self.vectorizer.fit_transform(texts)
        if self.pca:
            X = self.pca.fit_transform(X.toarray())
        return X

    def transform(self, data: List[Example]):
        texts = [ex.text for ex in data]
        X = self.vectorizer.transform(texts)
        if self.pca:
            X = self.pca.transform(X.toarray())
        return X

