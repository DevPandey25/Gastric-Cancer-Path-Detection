# GastricPath AI — Original Blueprint Phase Alignment

**Master Specification:** `GastricPath_AI_Research_and_Implementation_Blueprint.docx`  
**Date:** November 7, 2026  
**Document ID:** `GP-ORIGINAL-ALIGNMENT-2026-V1`

---

## 1. Overview & Clarification of Numbering Taxonomy

This document establishes the official alignment between the completed work in this repository and the **original Phase 0 through Phase 12 architecture** specified in the authoritative master blueprint (`GastricPath_AI_Research_and_Implementation_Blueprint.docx`).

### Structural Distinction
During recent foundation model research on the 10% stratified GCHTID benchmark, four focused empirical studies were executed using compact roadmap labels (Phases 6 through 9). These are officially categorized as **Supplementary Research Work Packages** under the Foundation Model & Post-Hoc Analysis stream. They do **not** replace the original heavy clinical engines (Tissue Segmentation, Nuclei Morphology, Spatial Cell GNNs, and WSI/MIL Pyramids).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 GASTRICPATH AI MASTER RESEARCH ARCHITECTURE                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ Core Original Pipeline (Phases 0–5):                                        │
│  - Phase 0: Repository Audit & Baseline Preservation        [Tested]        │
│  - Phase 1: Research Infrastructure & Reproducibility       [Tested]        │
│  - Phase 2: Dataset Audit, Manifests & Splitting            [Tested]        │
│  - Phase 3: Classification Benchmark Suite                 [Tested]        │
│  - Phase 4: Foundation Models (DINOv2 & Phikon Benchmarks)  [Tested]        │
│  - Phase 5: Explainability, Calibration & Abstention        [Complete]      │
├─────────────────────────────────────────────────────────────────────────────┤
│ Supplementary Research Work Packages (Foundation Model Post-Hoc Studies):   │
│  - Work Package A (Roadmap 6): Controlled Robustness Analysis (N=467)       │
│  - Work Package B (Roadmap 7): Feature-Space OOD Screening (Mahalanobis)    │
│  - Work Package C (Roadmap 8): Intra-Patch Spatial Token Analysis (N=96)   │
│  - Work Package D (Roadmap 9): Empirical Error Analysis & Gallery (N=467)   │
├─────────────────────────────────────────────────────────────────────────────┤
│ Advanced Downstream Clinical Engines (Original Phases 6–12):                │
│  - Phase 6: Tissue Segmentation                             [Blocked by data]
│  - Phase 7: Nuclei & Cellular Morphology                    [Blocked by data]
│  - Phase 8: Spatial Graph & Cell GNN Engine                 [Blocked by data]
│  - Phase 9: Whole-Slide Image (WSI) Engine & MIL            [Blocked by data]
│  - Phase 10: Multimodal Clinical Research                   [Partial/Blocked]
│  - Phase 11: Unified Research Workbench UI                  [In progress]   │
│  - Phase 12: Testing, Deployment & Research Release         [Active]        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Phase-by-Phase Alignment Matrix

| Original DOCX Phase | Original Description | Current Status | Completed Artifacts & Deliverables | Data / Hardware Blockers |
| :--- | :--- | :---: | :--- | :--- |
| **Phase 0** | Repository audit and baseline preservation | **Tested** | `docs/REPOSITORY_AUDIT.md`, baseline Git branch, SQLite verification | None |
| **Phase 1** | Research infrastructure and reproducibility | **Tested** | Central YAML config, experiment tracker, model registry, 21 unit tests | None |
| **Phase 2** | Dataset audit and validation | **Tested** | Image integrity audit (31,096 images), duplicate detection (SHA-256 & dHash), manifests | None |
| **Phase 3** | Classification research engine | **Tested** | `GastricCancerRNN` baseline, evaluation pipeline, per-class metrics | None |
| **Phase 4** | Pathology foundation models | **Tested** | DINOv2 ViT-S/14 full cache (31,096), Phikon ViT-B/16 pilot, 10% stratified benchmark (3,110), linear/nonlinear probes, FAISS retriever | GPU VRAM limits heavy 1B models (UNI2, GigaPath) to future cloud compute |
| **Phase 5** | Explainability and uncertainty | **Complete** | **Foundation Model:** Phikon LOO feature attribution, deletion faithfulness benchmark.<br>**CNN Baseline:** Grad-CAM, Integrated Gradients, Occlusion Sensitivity, MC Dropout.<br>**Calibration:** Validation temperature scaling ($T^*$), ECE, Brier score, reliability diagrams.<br>**Abstention:** Validation-calibrated selective classification ($\theta=0.40, 0.525$). | None |
| **Supplementary Package A** | Controlled Robustness Analysis (Roadmap 6) | **Tested** | Full test split ($N=467$) across 4 perturbation families (2,335 instances), paired bootstrap CIs, McNemar tests | None |
| **Supplementary Package B** | Lightweight OOD Screening (Roadmap 7) | **Tested** | Class-conditional Gaussian Mahalanobis distance, Ledoit-Wolf shrinkage, validation calibration, proxy-OOD ROCs | None |
| **Supplementary Package C** | Intra-Patch Spatial Analysis (Roadmap 8) | **Tested** | $14\times14$ token norm maps, $2\times2$ quadrant pooling, CLS vs Mean patch orthogonality discovery ($N=96$) | None |
| **Supplementary Package D** | Research Error Analysis (Roadmap 9) | **Tested** | Full error matrix, top confusion pairs, 48-exemplar deterministic gallery, cross-phase metadata synthesis | None |
| **Phase 6** | Tissue segmentation | **Blocked by data** | Architecture planned (U-Net, SegFormer) | **Blocked by data:** GCHTID contains only patch-level classification labels; zero pixel-level segmentation masks exist in the repository. |
| **Phase 7** | Nuclei and cellular morphology | **Blocked by data** | Architecture planned (HoVer-Net / StarDist) | **Blocked by data:** Requires nuclear instance segmentations or bounding boxes. |
| **Phase 8** | Spatial graph and GNN engine | **Blocked by data** | Architecture planned (Cellular Delaunay graphs, PyTorch Geometric GCN/GAT) | **Blocked by data:** Cellular graph construction requires validated nuclear detections from Phase 7. |
| **Phase 9** | Whole-slide image (WSI) engine | **Blocked by data** | OpenSlide WSI reader planned; CLAM / Attention MIL planned | **Blocked by data:** GCHTID consists strictly of cropped $224\times224$ tiles; zero gigapixel whole-slide images (`.svs`, `.ndpi`) exist in the repo. |
| **Phase 10** | Multimodal clinical research | **Implemented (Partial)** | Clinical cohort analyzer (300 patients) and miRNA target browser active on dashboard | **Deep fusion blocked by data:** GCHTID image tiles lack patient identifiers linking them to `HMU-GC-Clinical.csv`. |
| **Phase 11** | Research workbench UI | **In Progress** | FastAPI live inference interface active; multi-tab research workbench shell in development | None |
| **Phase 12** | Testing, deployment, and release | **Active** | 68 automated unit/regression tests passing, model cards, artifact provenance audits | None |

---

## 3. Original Phase 5 Scope vs. Completed Work

The table below audits the complete feature requirements of Original Phase 5 as defined in `GastricPath_AI_Research_and_Implementation_Blueprint.docx`:

| Requirement in Original DOCX | Implementation Details | Status |
| :--- | :--- | :---: |
| **Grad-CAM for supported CNNs** | Implemented on `GastricCancerRNN.patch_encoder.encoder[8]` (Conv2d 64$\to$128). Feature activations and gradients pooled across 16 sequential patches and assembled into a seamless $224\times224$ heatmap overlay. | **Tested** |
| **Integrated Gradients** | Implemented with $m=20$ Riemann interpolation steps along the straight-line path from neutral baseline to input patch sequence. Channel-magnitude aggregated to $224\times224$. | **Tested** |
| **Occlusion Sensitivity** | Implemented via sliding $28\times28$ neutral gray window (stride 28, 64 positions). Probability drop $\Delta P$ mapped and cubic-interpolated across the tile. | **Tested** |
| **Temperature Scaling** | Optimized scalar temperature $T^*$ **strictly on validation logits ($N=466$)** via L-BFGS-B minimizing cross-entropy NLL. Zero test data leakage. | **Tested** |
| **ECE & Brier Score** | Expected Calibration Error (10 bins) and multi-class Brier score computed before and after calibration on both validation and held-out test sets. | **Tested** |
| **Reliability Diagrams** | Dual-panel reliability diagrams (accuracy vs. confidence bars, diagonal reference, bin count annotations) generated for both Phikon and CNN-RNN models. | **Tested** |
| **Uncertainty / MC Dropout** | $T=20$ stochastic forward passes with active dropout on the CNN-RNN head to compute predictive distribution variance $\sigma^2_{\text{MC}}$ and Shannon entropy. | **Tested** |
| **Selective Classification / Abstention** | Validation-calibrated confidence thresholding. Evaluated on test set: Operating Point A ($90\%$ coverage $\to 71.81\%$ acc) and Point B ($80\%$ target acc $\to 76.81\%$ selective acc). | **Tested** |
| **No Fabricated Annotations Rule** | All attributions, heatmaps, and calibration curves are computed strictly through the neural networks. No artificial biological claims are made. | **Certified** |

---

## 4. Current State & Next Steps

1. **Original Phase 5 is Fully Complete:** Both foundation model (Phikon) and baseline CNN-RNN explainability, calibration, and uncertainty methods are operational, tested, and documented.
2. **Supplementary Analyses are Preserved:** All findings from the 4 supplementary work packages (Robustness, OOD, Spatial Tokens, Error Analysis) are retained with rigorous documentation.
3. **Advance to Next Feasible Phase:** Because Original Phases 6, 7, 8, and 9 are blocked by the absence of pixel segmentation masks, nuclear annotations, and WSI files in the GCHTID dataset, research advances logically to **Original Phase 10 / Phase 11 (Multimodal Exploration & Research Workbench UI)**, preserving data integrity without fabricating external datasets.

