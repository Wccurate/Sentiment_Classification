from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.data.datasets import Example


@dataclass
class SBERTConfig:
    model_name: str = "BAAI/bge-small-en-v1.5"
    device: str | None = None
    batch_size: int = 64


class SentenceBertEncoder:
    def __init__(self, cfg: SBERTConfig):
        self.cfg = cfg
        self.model = SentenceTransformer(cfg.model_name, device=cfg.device)

    def fit_transform(self, data: List[Example]):
        texts = [ex.text for ex in data]
        embs = self.model.encode(texts, batch_size=self.cfg.batch_size, convert_to_numpy=True, show_progress_bar=False)
        return embs

    def transform(self, data: List[Example]):
        texts = [ex.text for ex in data]
        embs = self.model.encode(texts, batch_size=self.cfg.batch_size, convert_to_numpy=True, show_progress_bar=False)
        return embs

