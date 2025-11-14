#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download multiple datasets from Hugging Face Hub using snapshot_download,
with explicit local directory naming, no symlinks, and easy extensibility.

Usage:
    python scripts/download_hf_datasets.py \
        --base-dir /path/to/data/raw \
        --only DeepPavlov/snips,PolyAI/banking77 \
        --max-workers 8
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from huggingface_hub import snapshot_download


# Default base directory to store the raw datasets
# Dynamically set based on the current file's location: ../../data/raw
DEFAULT_BASE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

# Dataset plan: list of (repo_id, local_dir_name)
# To add a new dataset, append a (repo_id, local_name) tuple here.
DATASETS: List[Tuple[str, str]] = [
    # ("DeepPavlov/snips", "snips_deeppavlov"),
    ("benayas/snips", "snips_benayas"),
    # ("tuetschek/atis", "atis_tuetschek"),
    # ("benayas/atis", "atis_benayas"),
    ("fathyshalab/atis_intents", "atis_intents_fathyshalab"),
    # ("PolyAI/banking77", "banking77_polyai"),
    ("DeepPavlov/hwu64", "hwu64_deeppavlov"),
    ("DeepPavlov/clinc_oos", "clinc_oos_deeppavlov"),
]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download datasets from Hugging Face Hub to a fixed directory structure."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help=f"Target base directory (default: {DEFAULT_BASE_DIR})",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help=(
            "Optional filter of repo_ids to download. "
            "Comma-separated list supporting glob patterns, e.g.: "
            "'DeepPavlov/*,PolyAI/banking77'"
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Max workers for parallel file downloads inside snapshot_download.",
    )
    return parser.parse_args(argv)


def ensure_dir(path: Path) -> None:
    """Create directory tree if not exists."""
    path.mkdir(parents=True, exist_ok=True)


def filter_datasets(
    datasets: Iterable[Tuple[str, str]], only: str
) -> List[Tuple[str, str]]:
    """
    Filter dataset tuples by 'only' glob patterns on repo_id.
    If only is empty, return original list.
    """
    items = list(datasets)
    if not only:
        return items

    patterns = [p.strip() for p in only.split(",") if p.strip()]
    if not patterns:
        return items

    selected: List[Tuple[str, str]] = []
    for repo_id, local_name in items:
        if any(fnmatch.fnmatch(repo_id, pat) for pat in patterns):
            selected.append((repo_id, local_name))
    return selected


def download_one(
    repo_id: str,
    local_name: str,
    base_dir: Path,
    *,
    max_workers: int = 8,
) -> Path:
    """
    Download a single dataset repo to base_dir/local_name using snapshot_download.

    Args:
        repo_id: HF repo id, e.g., 'DeepPavlov/snips'.
        local_name: Local directory name to store the dataset.
        base_dir: Base directory under which the dataset directory is created.
        max_workers: Max worker threads for file downloads.

    Returns:
        The path to the local dataset directory.
    """
    target_dir = base_dir / local_name
    ensure_dir(target_dir)

    print(f"[+] Downloading {repo_id} -> {target_dir}")

    # snapshot_download fetches to cache and then materializes into local_dir.
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision="main",  # keep main; change if you need a specific revision
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,  # copy files, do not symlink
        max_workers=max_workers,
        # Keep defaults for patterns to download the original repo contents.
        resume_download=True,
        ignore_patterns=None,
        allow_patterns=None,
    )

    print(f"[✓] Finished {repo_id} -> {target_dir}")
    return target_dir


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    base_dir: Path = args.base_dir
    ensure_dir(base_dir)

    plan = filter_datasets(DATASETS, args.only)
    if not plan:
        print("[!] No datasets matched the '--only' filter. Nothing to do.")
        return 0

    print(f"Base directory: {base_dir}")
    print("Plan:")
    for repo_id, local_name in plan:
        print(f"  - {repo_id} -> {base_dir / local_name}")

    failures: List[str] = []
    for repo_id, local_name in plan:
        try:
            download_one(
                repo_id=repo_id,
                local_name=local_name,
                base_dir=base_dir,
                max_workers=args.max_workers,
            )
        except Exception as e:
            failures.append(f"{repo_id} ({e})")
            print(f"[x] Failed: {repo_id} -> {e}")

    if failures:
        print("\nSome downloads failed:")
        for f in failures:
            print(f"  - {f}")
        return 2

    print("\nAll downloads completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))