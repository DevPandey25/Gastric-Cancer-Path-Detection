"""Extracts and caches perturbed embeddings for Phase 7 Mahalanobis proxy-OOD evaluation."""

import logging
from pathlib import Path
import numpy as np

from research.robustness.pipeline import RobustnessPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extract_proxy_embeddings")

CONDITIONS = [
    "brightness_contrast",
    "gaussian_blur",
    "jpeg_compression",
    "stain_hed_variation",
]

def main():
    ood_dir = Path("research/ood")
    ood_dir.mkdir(parents=True, exist_ok=True)
    rp = RobustnessPipeline()

    for cond in CONDITIONS:
        save_path = ood_dir / f"perturbed_embeddings_{cond}.npy"
        if save_path.exists():
            logger.info("Found existing %s, skipping extraction.", save_path.name)
            continue
        logger.info("Extracting embeddings for condition '%s'...", cond)
        emb = rp.extract_perturbed_embeddings(cond, batch_size=16)
        np.save(save_path, emb.astype(np.float32))
        logger.info("Saved %s (shape %s)", save_path.name, emb.shape)

    logger.info("All 4 perturbed embedding arrays successfully saved in %s", ood_dir)

if __name__ == "__main__":
    main()

