from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import csv
import random


@dataclass
class Example:
    text: str
    label: int | None


def load_csv_dataset(path: str, text_col: str = "text", label_col: str | None = "label") -> List[Example]:
    data: List[Example] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get(text_col, "").strip()
            label_val = None
            if label_col is not None and label_col in row and row[label_col] != "":
                label_val = int(row[label_col])
            data.append(Example(text=text, label=label_val))
    return data


def split_dataset(data: List[Example], val_ratio: float = 0.1, seed: int = 42) -> Tuple[List[Example], List[Example]]:
    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    n_val = int(len(indices) * val_ratio)
    val_idx = set(indices[:n_val])
    train, val = [], []
    for i, ex in enumerate(data):
        (val if i in val_idx else train).append(ex)
    return train, val
