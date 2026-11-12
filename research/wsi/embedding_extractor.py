"""Foundation Model Embedding Integration Engine for Whole-Slide Histopathology.

Connects extracted WSI patch grids with Phase 4 frozen foundation encoders (Phikon ViT-B/16 / DINOv2):
- Memory-bounded batch inference.
- Exact foundation model preprocessing and L2 normalization.
- Strict provenance mapping: associates every embedding row with (slide_id, patch_id, level0_x, level0_y).
- Prohibits fabrication of artificial WSI embeddings when real whole-slide files are absent.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

from research.wsi.patch_manifest import PatchManifest, PatchRecord

logger = logging.getLogger("wsi.embeddings")


class WSIEmbeddingsBlockedError(RuntimeError):
    """Raised when embedding extraction is attempted without genuine Whole-Slide Images or patch tiles."""
    pass


# Exact ImageNet normalization standard across Phikon and DINOv2
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_wsi_patch_preprocessing_transform(target_size: int = 224) -> T.Compose:
    """Returns deterministic PyTorch transformation pipeline matching Phase 4 foundation models."""
    return T.Compose([
        T.Resize((target_size, target_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class WSIEmbeddingExtractor:
    """
    Inference interface connecting WSI patch manifests with frozen foundation encoders.
    """

    def __init__(
        self,
        model_name: str = "phikon",
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name.lower().strip()
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.transform = get_wsi_patch_preprocessing_transform()
        self._model = None

    def _load_model(self) -> Any:
        """Lazily loads the specified foundation model in frozen evaluation mode."""
        if self._model is not None:
            return self._model

        logger.info(f"Loading frozen foundation encoder '{self.model_name}' on {self.device}...")
        # Note: In Phase 4, Phikon Mean Patch (ViT-B/16, 768-D) was the certified winning representation
        # When operating in offline/test environment, loads existing cached weights or raises clear error
        if self.model_name == "phikon":
            from transformers import AutoModel
            try:
                model = AutoModel.from_pretrained("owkin/phikon", local_files_only=True)
            except Exception as e:
                # If local HuggingFace cache is unavailable
                raise RuntimeError(f"Could not load local Phikon model: {e}") from e
        else:
            raise ValueError(f"Unsupported WSI foundation model: {self.model_name}")

        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        model.to(self.device)
        self._model = model
        return self._model

    def extract_bag_embeddings(
        self,
        manifest: PatchManifest,
        slide_id: str,
        base_dir: Optional[Union[str, Path]] = None,
        is_genuine_wsi: bool = True,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Extracts 768-dimensional L2-normalized embeddings for all patches in a slide.

        Args:
            manifest: PatchManifest containing extracted patch records.
            slide_id: Target slide ID.
            base_dir: Base directory where patch images reside.
            is_genuine_wsi: Flag confirming whether source slide is real clinical data.

        Returns:
            embeddings: np.ndarray of shape (K, 768), float32.
            metadata_list: List of K dictionaries preserving spatial coordinates and patch IDs.
        """
        # Data integrity rule: Do not extract fake WSI embeddings
        if not is_genuine_wsi:
            raise WSIEmbeddingsBlockedError(
                f"Cannot generate WSI bag embeddings for slide '{slide_id}': "
                "The source slide is a synthetic test fixture or genuine WSI files are absent. "
                "Fabrication of artificial WSI bag embeddings is strictly prohibited by research protocol."
            )

        records = manifest.get_slide_records(slide_id)
        if not records:
            return np.empty((0, 768), dtype=np.float32), []

        # Verify patch image files exist
        base_p = Path(base_dir) if base_dir else Path(".")
        missing_files = []
        for r in records:
            img_p = base_p / r.relative_patch_path
            if not img_p.exists():
                missing_files.append(str(img_p))

        if missing_files:
            raise WSIEmbeddingsBlockedError(
                f"Missing {len(missing_files)} patch images on disk for slide '{slide_id}'. "
                f"First missing: {missing_files[0]}"
            )

        model = self._load_model()
        all_embeddings: List[np.ndarray] = []
        metadata_list: List[Dict[str, Any]] = []

        # Process in memory-bounded batches
        for i in range(0, len(records), self.batch_size):
            batch_records = records[i : i + self.batch_size]
            tensors = []
            for r in batch_records:
                img_p = base_p / r.relative_patch_path
                img = Image.open(img_p).convert("RGB")
                tensors.append(self.transform(img))
                metadata_list.append({
                    "slide_id": r.slide_id,
                    "patch_id": r.patch_id,
                    "level": r.level,
                    "level0_x": r.level0_x,
                    "level0_y": r.level0_y,
                    "tissue_percentage": r.tissue_percentage,
                })

            batch_t = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                out = model(batch_t)
                # Phikon Mean Patch representation (average over 196 patch tokens, index 1:)
                if hasattr(out, "last_hidden_state"):
                    patch_tokens = out.last_hidden_state[:, 1:, :]  # (B, 196, 768)
                    emb = patch_tokens.mean(dim=1)  # (B, 768)
                else:
                    emb = out[0].mean(dim=1)

                # L2 normalization
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                all_embeddings.append(emb.cpu().numpy().astype(np.float32))

        embeddings = np.vstack(all_embeddings)
        return embeddings, metadata_list

