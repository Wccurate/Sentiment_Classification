from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from gensim.models import KeyedVectors
from src.data.datasets import Example


@dataclass
class W2VConfig:
    model_path: str
    binary: bool = True
    lowercase: bool = True


class Word2VecAverager:
    def __init__(self, cfg: W2VConfig):
        self.cfg = cfg
        self.kv: Optional[KeyedVectors] = None

    def load(self):
        if self.kv is None:
            self.kv = KeyedVectors.load_word2vec_format(self.cfg.model_path, binary=self.cfg.binary)

    def _embed(self, text: str) -> np.ndarray:
        assert self.kv is not None, "Word2Vec vectors not loaded"
        toks = text.strip().split()
        if self.cfg.lowercase:
            toks = [t.lower() for t in toks]
        vecs = [self.kv[w] for w in toks if w in self.kv]
        if not vecs:
            return np.zeros(self.kv.vector_size, dtype=np.float32)
        return np.mean(np.stack(vecs, axis=0), axis=0)

    def fit_transform(self, data: List[Example]):
        self.load()
        return np.stack([self._embed(ex.text) for ex in data], axis=0)

    def transform(self, data: List[Example]):
        return np.stack([self._embed(ex.text) for ex in data], axis=0)

