from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    task: str = "sentiment"
    method: str = "tfidf_svm"
    model_name: Optional[str] = None
    tokenizer_name: Optional[str] = None
    max_len: int = 128
    batch_size: int = 32
    lr: float = 2e-5
    epochs: int = 3
    seed: int = 42
    pca_components: Optional[int] = None
    loss: str = "cross_entropy"
    lora_r: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: Optional[float] = None

