#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download embedding models from Hugging Face Hub to local directory.
"""

from pathlib import Path
from huggingface_hub import snapshot_download


# Default base directory to store the embedding models
DEFAULT_BASE_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "huggingface_models"

def download_embedder(
    repo_id: str = "Lajavaness/bilingual-embedding-small",
    local_name: str = "bilingual_embedding_small_lajavaness",
    base_dir: Path = DEFAULT_BASE_DIR,
    max_workers: int = 8,
) -> Path:
    """
    Download an embedding model from Hugging Face Hub.
    
    Args:
        repo_id: HF repo id, e.g., 'Lajavaness/bilingual-embedding-small'.
        local_name: Local directory name to store the model.
        base_dir: Base directory under which the model directory is created.
        max_workers: Max worker threads for file downloads.
        
    Returns:
        The path to the local model directory.
    """
    target_dir = base_dir / local_name
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[+] Downloading {repo_id} -> {target_dir}")
    
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision="main",
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,  # Copy files instead of using symlinks
        max_workers=max_workers,
        resume_download=True,
        ignore_patterns=None,
        allow_patterns=None,
    )
    
    print(f"[✓] Finished downloading to {target_dir}")
    return target_dir


if __name__ == "__main__":
    # Download the bilingual embedding model
    download_embedder(
        repo_id="Lajavaness/bilingual-embedding-small",
        local_name="bilingual_embedding_small_lajavaness",
    )
