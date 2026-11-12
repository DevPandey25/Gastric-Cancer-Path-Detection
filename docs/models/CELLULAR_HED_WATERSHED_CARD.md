# Model Card: Classical HED-Watershed Cellular Candidate Detector

## 1. Module Overview
- **Method Name:** Ruifrok Color Deconvolution with Marker-Controlled Distance Watershed
- **Algorithmic Category:** Classical Unsupervised Computer Vision Pipeline (Zero Trainable Weights)
- **Role:** Candidate cellular object segmenter and morphological descriptor generator
- **Pilot Dataset Scope:** 64 representative GCHTID patches (8 balanced samples per class)
- **Total Candidate Objects Detected:** 3,895 candidate cellular instances ($60.86 \pm 38.4$ per tile)

---

## 2. Detection Methodology

```
Input H&E Tile (224x224x3 RGB)
  │
  ▼
Ruifrok Optical Density Deconvolution (Ruifrok & Johnston, 2001)
  │ (Separates Hematoxylin, Eosin, and Residual channels via 3x3 absorption matrix)
  ▼
Hematoxylin Optical Density Map
  │
  ▼
Otsu / Adaptive Local Thresholding & Morphological Opening
  │
  ▼
Euclidean Distance Transform
  │
  ▼
Peak Local Maxima Identification (Nuclei Seeds / Markers)
  │
  ▼
Marker-Controlled Watershed Segmentation
  │
  ▼
Connected Component Analysis & 16-D Morphological Descriptors
```

---

## 3. Extracted 16-D Morphological & Coordinate Features
For every candidate object, a standardized 16-dimensional descriptor vector is extracted:
1. Centroid X coordinate (normalized $[0, 1]$)
2. Centroid Y coordinate (normalized $[0, 1]$)
3. Object Area ($A$ in pixels)
4. Object Perimeter ($P$ in pixels)
5. Circularity / Form Factor ($4 \pi A / P^2$)
6. Major Axis Length
7. Minor Axis Length
8. Aspect Ratio (Eccentricity proxy)
9. Solidity (Area / Convex Hull Area)
10. Extent (Area / Bounding Box Area)
11. Mean Hematoxylin Optical Density
12. Standard Deviation of Hematoxylin OD
13. Mean Eosin Optical Density
14. Max Hematoxylin OD (chromatin condensation proxy)
15. Local Neighbor Density (within $R = 30\ \text{px}$)
16. Orientation Angle ($-\pi/2$ to $\pi/2$)

---

## 4. Deep Learning Detector Blockers & Scientific Disclosure
1. **Candidate Object Designation:** Objects detected by this pipeline are formally designated as **`candidate objects`**, NOT manually validated nuclei. No nuclear classification (e.g., neoplastic vs stromal vs tumor-infiltrating lymphocyte) is claimed.
2. **Deep Learning Model Blockage:** Deep learning nuclei models (Cellpose, StarDist 2D H&E, HoVer-Net) require external pretrained weights ($\approx 50\text{--}250\ \text{MB}$) that are absent from the offline local repository. To maintain strict reproducibility within the offline environment without fabricating model outputs, the classical Ruifrok HED-watershed was selected as the verified reproducible baseline.

