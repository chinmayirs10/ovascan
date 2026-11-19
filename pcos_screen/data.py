"""Dataset loading, leakage auditing and group-aware splitting.

The public PCOS ultrasound dataset this project started from ships with a
``train/`` and ``test/`` folder. Auditing showed that the test folder is a
byte-for-byte copy of the training folder, and that most images are
augmented variants (flips, zooms, shifts) of a small number of source scans.
Splitting such data at image level leaks near-identical images across
splits, so this module groups images into "families" and splits by family.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Folder names in the source dataset -> label (1 = PCOS-like, 0 = no PCOS-like features)
CLASS_FOLDERS = {"infected": 1, "notinfected": 0}
LABEL_NAMES = {1: "pcos_like", 0: "normal"}


@dataclass
class ImageRecord:
    path: Path
    label: int
    source_split: str  # folder it came from in the original dataset
    md5: str


def md5_of(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def scan_dataset(root: Path) -> list[ImageRecord]:
    """Find every image under ``root/<split>/<class>/``."""
    records = []
    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for class_name, label in CLASS_FOLDERS.items():
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                continue
            for f in sorted(class_dir.iterdir()):
                if f.suffix.lower() in IMAGE_EXTS:
                    records.append(ImageRecord(f, label, split_dir.name, md5_of(f)))
    return records


def deduplicate(records: list[ImageRecord]) -> tuple[list[ImageRecord], dict]:
    """Keep one record per unique file content and report what was dropped."""
    by_hash: dict[str, list[ImageRecord]] = {}
    for r in records:
        by_hash.setdefault(r.md5, []).append(r)

    unique, cross_split, label_conflicts = [], 0, 0
    for group in by_hash.values():
        if len({r.source_split for r in group}) > 1:
            cross_split += 1
        if len({r.label for r in group}) > 1:
            label_conflicts += 1
            continue  # same pixels with two labels: unusable
        unique.append(group[0])

    stats = {
        "files_total": len(records),
        "unique_contents": len(by_hash),
        "duplicated_across_splits": cross_split,
        "label_conflicts_dropped": label_conflicts,
        "kept": len(unique),
    }
    return unique, stats


def group_into_families(embeddings: np.ndarray, labels: np.ndarray, threshold: float) -> np.ndarray:
    """Cluster near-duplicate images (augmentations of the same scan).

    Uses agglomerative clustering on cosine distance of CNN embeddings,
    separately per class so a family never mixes labels.
    Returns a family id per image.
    """
    from sklearn.cluster import AgglomerativeClustering

    family = np.full(len(labels), -1, dtype=int)
    next_id = 0
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        if len(idx) == 1:
            family[idx] = next_id
            next_id += 1
            continue
        clus = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=threshold,
        ).fit(embeddings[idx])
        family[idx] = clus.labels_ + next_id
        next_id += clus.labels_.max() + 1
    return family


def group_split(
    families: np.ndarray,
    labels: np.ndarray,
    test_frac: float = 0.2,
    val_frac: float = 0.15,
    seed: int = 42,
) -> np.ndarray:
    """Assign whole families to train/val/test, stratified by class."""
    rng = np.random.default_rng(seed)
    split = np.empty(len(labels), dtype=object)
    for label in np.unique(labels):
        fams = np.unique(families[labels == label])
        rng.shuffle(fams)
        n = len(fams)
        n_test = max(1, round(n * test_frac))
        n_val = max(1, round(n * val_frac)) if n - n_test > 1 else 0
        assignment = {}
        for i, f in enumerate(fams):
            assignment[f] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
        for i in np.where(labels == label)[0]:
            split[i] = assignment[families[i]]
    return split


def write_split_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_split_csv(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))
