from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader

from src.data.datasets import Example


class ExampleDataset(Dataset):
    def __init__(self, data: List[Example]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


@dataclass
class TokenizingCollator:
    tokenizer: any
    max_length: int

    def __call__(self, batch: List[Example]):
        texts = [ex.text for ex in batch]
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if batch[0].label is not None:
            labels = torch.tensor([ex.label for ex in batch], dtype=torch.long)
        else:
            labels = torch.full((len(batch),), -100, dtype=torch.long)
        encodings["labels"] = labels
        return encodings


def build_dataloader(
    data: List[Example],
    tokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool = False,
):
    dataset = ExampleDataset(data)
    collator = TokenizingCollator(tokenizer=tokenizer, max_length=max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)

