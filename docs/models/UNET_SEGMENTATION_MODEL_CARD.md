# Model Card: U-Net Tissue Segmentation Infrastructure

## 1. Model Status: UNTRAINED / DATA-BLOCKED
- **Operational Status:** Fully implemented software architecture; **untrained on clinical data**.
- **Blocker Classification:** **Fundamental Data Absence in GCHTID**.
- **Integrity Guarantee:** Zero artificial segmentation masks were synthesized, and zero fabricated Dice coefficients or IoU metrics are reported.

---

## 2. Architecture Specifications
- **Architecture Name:** Classical 2D U-Net (Ronneberger et al., 2015)
- **Input Channels:** 3 (RGB optical density, normalized $\mu=0.5, \sigma=0.5$)
- **Output Channels:** 9 logits ($C = 9$, Background + 8 canonical GCHTID tissue classes)
- **Encoder Stages:** 4 downsampling blocks ($64 \rightarrow 128 \rightarrow 256 \rightarrow 512$ feature channels) with double $3 \times 3$ convolutions, Batch Normalization, and ReLU activations, followed by $2 \times 2$ max pooling.
- **Bottleneck Stage:** 1,024 channels.
- **Decoder Stages:** 4 upsampling blocks with bilinear/transposed convolution, concatenation of corresponding encoder skip connections, and double $3 \times 3$ convolutions.
- **Final Projection:** $1 \times 1$ convolution projecting from 64 to 9 class channels.

---

## 3. Explicit 9-Class Segmentation Ontology & Overlap Hierarchy

When overlapping tissue annotations occur, masks are resolved strictly according to the hierarchical overlap priority:

| Class ID | Code | Canonical Tissue Name | Overlap Priority | Handling Rule |
|:---:|:---:|:---|:---:|:---|
| 0 | **BG** | Background / Slide Glass | 0 | Lowest default layer; overridden by all tissue |
| 1 | **ADI** | Adipose Tissue | 1 | Non-epithelial fat droplets |
| 2 | **MUS** | Smooth Muscle | 2 | Muscularis wall layers |
| 3 | **MUC** | Mucus | 3 | Extracellular secretory pools |
| 4 | **NOR** | Normal Gastric Mucosa | 4 | Intact glandular epithelium |
| 5 | **STR** | Cancer-Associated Stroma | 5 | Desmoplastic fibrotic matrix |
| 6 | **LYM** | Lymphocytes / Inflammatory | 6 | Dense immune infiltrates |
| 7 | **DEB** | Cellular Debris / Necrosis | 7 | Secondary necrotic breakdown |
| 8 | **TUM** | Gastric Adenocarcinoma | 8 | Highest priority; takes precedence in ambiguous borders |

---

## 4. QuPath GeoJSON Ingestion Pipeline
The module includes a verified polygon-to-raster mask rasterizer (`research/segmentation/annotations.py`) capable of parsing standard QuPath GeoJSON export files containing:
- FeatureCollections with `Polygon` and `MultiPolygon` geometries.
- Classification property matching against the canonical 9-class ontology table.
- Sub-pixel boundary rasterization using standard affine coordinate transformation.

*Prerequisite for Training:* Release of validated pixel-level segmentation masks from an external gastric carcinoma cohort (e.g., PAIP 2019 or DigestPath).

