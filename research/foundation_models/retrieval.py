"""Lightweight Foundation Model Embedding Similarity Retrieval Engine.

Supports exact cosine similarity retrieval over L2-normalized pathology representations
using FAISS IndexFlatIP (if installed) or an exact NumPy vectorized matrix multiplication
fallback. Enforces strict split isolation, query self-match exclusion, and duplicate
hash exclusion.

DISCLOSURE: Retrieved neighbors represent morphological proximity in embedding space,
not confirmed biological identity or clinical diagnostic certainty.
"""

from __future__ import annotations

import logging
from typing import Any
import numpy as np

from research.foundation_models.metadata import ImageEmbeddingRecord, RetrievalMatch

logger = logging.getLogger(__name__)

# Check for FAISS availability
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False


class HistopathologySimilarityRetriever:
    """Exact cosine similarity retriever supporting FAISS and NumPy vectorized execution."""

    def __init__(
        self,
        embeddings: np.ndarray,
        records: list[ImageEmbeddingRecord],
        reference_split: str | None = None,
        use_faiss: bool = True,
    ) -> None:
        """Initializes the retriever with a reference gallery.

        Args:
            embeddings: Float32 matrix of shape [N, D]
            records: Metadata records corresponding to each row of embeddings
            reference_split: Optional filter (e.g. 'train' to index only training samples)
            use_faiss: Whether to use FAISS if installed
        """
        if len(embeddings) != len(records):
            raise ValueError(
                f"Mismatch: embeddings has {len(embeddings)} rows, records has {len(records)} items."
            )

        # Filter by reference_split if requested
        if reference_split is not None:
            indices = [i for i, r in enumerate(records) if r.dataset_split == reference_split]
            if not indices:
                raise ValueError(f"No records found with reference_split='{reference_split}'")
            self.embeddings = embeddings[indices].astype(np.float32)
            self.records = [records[i] for i in indices]
        else:
            self.embeddings = embeddings.astype(np.float32)
            self.records = records

        self.num_items, self.embedding_dim = self.embeddings.shape
        self.use_faiss = use_faiss and FAISS_AVAILABLE

        # Build search index
        if self.use_faiss:
            logger.info("Initializing FAISS IndexFlatIP for %d embeddings (dim=%d)", self.num_items, self.embedding_dim)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.index.add(self.embeddings)
        else:
            if use_faiss and not FAISS_AVAILABLE:
                logger.info("FAISS not installed; using exact NumPy vectorized matrix multiplication fallback.")
            self.index = None

    def query(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        query_image_id: str | None = None,
        query_relative_path: str | None = None,
        query_sha256: str | None = None,
        exclude_self: bool = True,
        exclude_duplicates: bool = True,
        filter_split: str | None = None,
    ) -> list[RetrievalMatch]:
        """Finds top-k nearest neighbors for a query embedding vector.

        Args:
            query_vector: 1D array of shape [D] or 2D array [1, D]
            top_k: Number of nearest neighbors to return
            query_image_id: Optional ID of query image (used for self-exclusion)
            query_relative_path: Optional relative path of query image
            query_sha256: Optional SHA-256 hash (used for duplicate exclusion)
            exclude_self: Whether to filter out the query image itself from results
            exclude_duplicates: Whether to filter out exact hash duplicates of the query
            filter_split: Optional filter restricting returned neighbors to a split

        Returns:
            List of RetrievalMatch dataclasses sorted by similarity descending
        """
        q = np.asarray(query_vector, dtype=np.float32).flatten()
        if len(q) != self.embedding_dim:
            raise ValueError(
                f"Query dimension mismatch: expected {self.embedding_dim}, got {len(q)}"
            )

        # Ensure query is L2-normalized
        q_norm = np.linalg.norm(q)
        if q_norm > 0 and not np.isclose(q_norm, 1.0, atol=1e-3):
            q = q / q_norm

        # Candidate pool size: request more neighbors than top_k to account for exclusions
        pool_k = min(self.num_items, top_k * 5 + 20)

        if self.use_faiss and self.index is not None:
            q_2d = np.ascontiguousarray(q.reshape(1, -1), dtype=np.float32)
            scores, indices = self.index.search(q_2d, pool_k)
            candidate_scores = scores[0]
            candidate_indices = indices[0]
        else:
            # Exact cosine similarity via vectorized dot product: S = X @ q
            scores_all = np.dot(self.embeddings, q)
            # Top-pool_k partition
            if pool_k < self.num_items:
                candidate_indices = np.argpartition(-scores_all, pool_k)[:pool_k]
                candidate_indices = candidate_indices[np.argsort(-scores_all[candidate_indices])]
            else:
                candidate_indices = np.argsort(-scores_all)
            candidate_scores = scores_all[candidate_indices]

        # Filter candidates
        matches: list[RetrievalMatch] = []
        rank = 1

        for score, idx in zip(candidate_scores, candidate_indices):
            if idx < 0 or idx >= len(self.records):
                continue
            rec = self.records[idx]

            # 1. Self exclusion check
            if exclude_self:
                if query_image_id and rec.image_id == query_image_id:
                    continue
                if query_relative_path and rec.relative_path == query_relative_path:
                    continue

            # 2. Duplicate exclusion check
            if exclude_duplicates and query_sha256:
                if rec.sha256 == query_sha256:
                    continue

            # 3. Split filter check
            if filter_split and rec.dataset_split != filter_split:
                continue

            # Check class match
            query_class = getattr(rec, "query_class", None)  # will be set if caller provides
            is_same_class = (query_class == rec.class_name) if query_class else False

            match = RetrievalMatch(
                query_image_id=query_image_id or "query",
                neighbor_image_id=rec.image_id,
                relative_path=rec.relative_path,
                sha256=rec.sha256,
                class_name=rec.class_name,
                dataset_split=rec.dataset_split,
                similarity_score=round(float(score), 4),
                rank=rank,
                is_same_class=is_same_class,
            )
            matches.append(match)
            rank += 1

            if len(matches) >= top_k:
                break

        return matches

    def get_gallery_summary(self) -> dict[str, Any]:
        """Returns summary statistics of the indexed gallery."""
        splits = {}
        classes = {}
        for r in self.records:
            splits[r.dataset_split] = splits.get(r.dataset_split, 0) + 1
            classes[r.class_name] = classes.get(r.class_name, 0) + 1

        return {
            "total_indexed": self.num_items,
            "embedding_dimension": self.embedding_dim,
            "backend": "FAISS (IndexFlatIP)" if self.use_faiss else "NumPy (Vectorized Dot Product)",
            "faiss_available": FAISS_AVAILABLE,
            "splits": splits,
            "classes": classes,
        }

