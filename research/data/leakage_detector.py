"""Perceptual Leakage and Near-Duplicate Detection for Histopathology Images.

Computes perceptual difference hashes (dHash) and average hashes (aHash) to identify
morphologically near-identical patches, detects cross-split leakage risks, and groups
suspicious clusters via multi-index candidate indexing and graph clustering.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def compute_dhash(image: Image.Image | Path | str, hash_size: int = 8) -> str:
    """Computes difference hash (dHash) based on horizontal intensity gradients.

    Produces a (hash_size * hash_size) bit hash as a hexadecimal string.
    Default hash_size=8 yields a 64-bit hash (16 hex chars).
    """
    if isinstance(image, (str, Path)):
        with Image.open(image) as img:
            return _calculate_dhash_from_img(img, hash_size)
    return _calculate_dhash_from_img(image, hash_size)


def _calculate_dhash_from_img(img: Image.Image, hash_size: int) -> str:
    # Resize to (width=hash_size + 1, height=hash_size) grayscale
    resized = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.int32)
    # Compare adjacent pixels horizontally: arr[:, 1:] > arr[:, :-1]
    diff = arr[:, 1:] > arr[:, :-1]
    # Pack 64 booleans into 8 bytes (64-bit integer)
    bit_str = "".join("1" if b else "0" for b in diff.flatten())
    int_val = int(bit_str, 2)
    # 16-character zero-padded hex
    hex_len = (hash_size * hash_size) // 4
    return f"{int_val:0{hex_len}x}"


def compute_ahash(image: Image.Image | Path | str, hash_size: int = 8) -> str:
    """Computes average hash (aHash) based on mean intensity thresholding."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as img:
            return _calculate_ahash_from_img(img, hash_size)
    return _calculate_ahash_from_img(image, hash_size)


def _calculate_ahash_from_img(img: Image.Image, hash_size: int) -> str:
    resized = img.convert("L").resize((hash_size, hash_size), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32)
    avg = arr.mean()
    diff = arr > avg
    bit_str = "".join("1" if b else "0" for b in diff.flatten())
    int_val = int(bit_str, 2)
    hex_len = (hash_size * hash_size) // 4
    return f"{int_val:0{hex_len}x}"


def hamming_distance(h1: str, h2: str) -> int:
    """Calculates bitwise Hamming distance between two hex hashes."""
    v1 = int(h1, 16)
    v2 = int(h2, 16)
    return bin(v1 ^ v2).count("1")


class UnionFind:
    """Disjoint Set Union (DSU) with path compression and union by rank."""

    def __init__(self, elements: list[str]) -> None:
        self.parent = {elem: elem for elem in elements}
        self.rank = {elem: 0 for elem in elements}

    def find(self, item: str) -> str:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            if self.rank[root_a] < self.rank[root_b]:
                self.parent[root_a] = root_b
            elif self.rank[root_a] > self.rank[root_b]:
                self.parent[root_b] = root_a
            else:
                self.parent[root_b] = root_a
                self.rank[root_a] += 1


@dataclass
class NearDuplicateCluster:
    cluster_id: int
    size: int
    classes: list[str]
    is_cross_class: bool
    paths: list[str]
    hashes: list[str]
    max_pairwise_hamming: int


@dataclass
class NearDuplicateReport:
    dataset_id: str
    scanned_at_utc: str
    total_images_evaluated: int
    hamming_threshold: int
    algorithm: str
    total_candidate_pairs: int
    total_clusters: int
    cross_class_clusters_count: int
    intra_class_clusters_count: int
    cross_class_conflicts: list[dict[str, Any]]
    scientific_limitation_disclosure: str
    clusters: list[NearDuplicateCluster]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "scanned_at_utc": self.scanned_at_utc,
            "total_images_evaluated": self.total_images_evaluated,
            "hamming_threshold": self.hamming_threshold,
            "algorithm": self.algorithm,
            "total_candidate_pairs": self.total_candidate_pairs,
            "total_clusters": self.total_clusters,
            "cross_class_clusters_count": self.cross_class_clusters_count,
            "intra_class_clusters_count": self.intra_class_clusters_count,
            "cross_class_conflicts": self.cross_class_conflicts,
            "scientific_limitation_disclosure": self.scientific_limitation_disclosure,
            "clusters": [asdict(c) for c in self.clusters],
        }

    def to_json(self, output_path: Path | str, indent: int = 2) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent)

    def to_csv(self, output_path: Path | str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "cluster_id",
                "size",
                "is_cross_class",
                "classes",
                "max_pairwise_hamming",
                "paths",
            ])
            for c in self.clusters:
                writer.writerow([
                    c.cluster_id,
                    c.size,
                    c.is_cross_class,
                    ";".join(sorted(set(c.classes))),
                    c.max_pairwise_hamming,
                    ";".join(c.paths),
                ])


class NearDuplicateDetector:
    """Finds near-duplicate images using dHash and multi-index inverted chunk lookup."""

    SCIENTIFIC_DISCLOSURE = (
        "SCIENTIFIC INTEGRITY LIMITATION: Perceptual similarity (dHash/aHash) indicates morphological "
        "and texture alignment, but does NOT prove that two patches originate from the same patient "
        "or Whole Slide Image (WSI). Histopathologically normal stroma, adipose tissue, or uniform "
        "mucinous pools from completely distinct individuals often present near-identical structural features."
    )

    def __init__(
        self,
        dataset_dir: Path | str,
        hamming_threshold: int = 3,
        hash_size: int = 8,
        dataset_id: str = "GCHTID",
    ) -> None:
        self.dataset_dir = Path(dataset_dir).resolve()
        self.hamming_threshold = hamming_threshold
        self.hash_size = hash_size
        self.dataset_id = dataset_id

    def find_near_duplicates(
        self,
        path_to_hash: dict[str, str] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[NearDuplicateReport, dict[str, str]]:
        """Identifies near-duplicate pairs and groups them into clusters.

        If path_to_hash is not provided, computes dHash for all images in dataset_dir.
        """
        if path_to_hash is None:
            if not self.dataset_dir.exists():
                raise FileNotFoundError(f"Dataset directory does not exist: {self.dataset_dir}")
            file_paths = []
            for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
                file_paths.extend(self.dataset_dir.rglob(f"*{ext}"))
                file_paths.extend(self.dataset_dir.rglob(f"*{ext.upper()}"))
            file_paths = sorted(list(set(file_paths)))

            path_to_hash = {}
            total = len(file_paths)
            for idx, p in enumerate(file_paths):
                if progress_callback and idx % 1000 == 0:
                    progress_callback(idx, total)
                rel_path = p.relative_to(self.dataset_dir).as_posix()
                try:
                    path_to_hash[rel_path] = compute_dhash(p, hash_size=self.hash_size)
                except Exception as exc:
                    logger.warning("Failed dHash for %s: %s", rel_path, exc)

        items = list(path_to_hash.items())
        total_items = len(items)

        # Multi-Index Hash Inverted Table for candidate retrieval
        # Split 64-bit hash into 8 chunks of 8 bits (1 byte each)
        # By pigeonhole principle, if hamming_dist <= 7, at least 1 of 8 chunks MUST be an exact match
        num_chunks = 8
        chunk_bits = (self.hash_size * self.hash_size) // num_chunks  # 64 // 8 = 8 bits
        mask = (1 << chunk_bits) - 1

        chunk_tables: list[dict[int, list[int]]] = [{} for _ in range(num_chunks)]
        int_hashes: list[int] = []

        for idx, (_, h_str) in enumerate(items):
            val = int(h_str, 16)
            int_hashes.append(val)
            for c_idx in range(num_chunks):
                chunk_val = (val >> (c_idx * chunk_bits)) & mask
                if chunk_val not in chunk_tables[c_idx]:
                    chunk_tables[c_idx][chunk_val] = []
                chunk_tables[c_idx][chunk_val].append(idx)

        # Candidate pair verification
        candidate_pairs: set[tuple[int, int]] = set()
        for c_idx in range(num_chunks):
            for idx_list in chunk_tables[c_idx].values():
                if len(idx_list) > 1:
                    for i in range(len(idx_list)):
                        for j in range(i + 1, len(idx_list)):
                            idx1, idx2 = idx_list[i], idx_list[j]
                            if idx1 > idx2:
                                idx1, idx2 = idx2, idx1
                            candidate_pairs.add((idx1, idx2))

        # Check exact Hamming distance on candidates
        verified_edges: list[tuple[str, str, int]] = []
        for idx1, idx2 in candidate_pairs:
            dist = bin(int_hashes[idx1] ^ int_hashes[idx2]).count("1")
            if dist <= self.hamming_threshold:
                p1 = items[idx1][0]
                p2 = items[idx2][0]
                verified_edges.append((p1, p2, dist))

        # Graph clustering via Union-Find
        all_paths = [p for p, _ in items]
        dsu = UnionFind(all_paths)
        edge_distances: dict[tuple[str, str], int] = {}

        for p1, p2, dist in verified_edges:
            dsu.union(p1, p2)
            edge_distances[(min(p1, p2), max(p1, p2))] = dist

        # Collect clusters of size >= 2
        root_to_paths: dict[str, list[str]] = {}
        for p in all_paths:
            root = dsu.find(p)
            if root not in root_to_paths:
                root_to_paths[root] = []
            root_to_paths[root].append(p)

        clusters: list[NearDuplicateCluster] = []
        cross_class_conflicts: list[dict[str, Any]] = []
        cluster_id = 0

        for root, paths in root_to_paths.items():
            if len(paths) >= 2:
                classes = [Path(p).parent.name if "/" in p else "UNKNOWN" for p in paths]
                unique_classes = sorted(list(set(classes)))
                is_cross = len(unique_classes) > 1

                # Compute maximum pairwise distance within cluster
                max_d = 0
                for i in range(len(paths)):
                    for j in range(i + 1, len(paths)):
                        d = edge_distances.get((min(paths[i], paths[j]), max(paths[i], paths[j])), None)
                        if d is not None and d > max_d:
                            max_d = d

                c = NearDuplicateCluster(
                    cluster_id=cluster_id,
                    size=len(paths),
                    classes=classes,
                    is_cross_class=is_cross,
                    paths=paths,
                    hashes=[path_to_hash[p] for p in paths],
                    max_pairwise_hamming=max_d,
                )
                clusters.append(c)
                cluster_id += 1

                if is_cross:
                    cross_class_conflicts.append({
                        "cluster_id": c.cluster_id,
                        "size": c.size,
                        "conflicting_classes": unique_classes,
                        "paths": paths,
                    })

        cross_class_count = sum(1 for c in clusters if c.is_cross_class)
        intra_class_count = len(clusters) - cross_class_count

        report = NearDuplicateReport(
            dataset_id=self.dataset_id,
            scanned_at_utc=datetime.now(timezone.utc).isoformat(),
            total_images_evaluated=total_items,
            hamming_threshold=self.hamming_threshold,
            algorithm="dHash-64-bit-multi-index",
            total_candidate_pairs=len(verified_edges),
            total_clusters=len(clusters),
            cross_class_clusters_count=cross_class_count,
            intra_class_clusters_count=intra_class_count,
            cross_class_conflicts=cross_class_conflicts,
            scientific_limitation_disclosure=self.SCIENTIFIC_DISCLOSURE,
            clusters=clusters,
        )

        return report, path_to_hash

