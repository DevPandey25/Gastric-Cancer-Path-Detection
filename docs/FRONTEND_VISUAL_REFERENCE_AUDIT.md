# GastricPath AI — Frontend Visual Reference Audit & UI/UX Blueprint
**Document ID:** `GASTRICPATH-UI-AUDIT-2026-v2`  
**Audit Timestamp:** `2026-11-13T00:15:00+05:30`  
**Reference Assets:**
- **Reference 1 (`media_1794508930876_7c77d0ea.jpg`):** Deep navy computational pathology workstation featuring a 4-panel grid: histopathology image viewer with Grad-CAM heatmap overlay and thumbnail strip, neural network architecture flow diagram, feature extraction workspace (LBP texture histogram and GLCM co-occurrence heatmaps), and classification dashboard with radial confidence meter.
- **Reference 2 (`media_1794508930905_f4f6c3a9.jpg`):** Advanced spatial research console featuring a dark central tissue viewer, tissue composition analytics, 4-tile AI-derived insights grid, spatial interaction vs. distance line graph, molecular biomarker table, pathway analysis, and AI summary card.

---

## 1. Visual Design System & Palette Specification

To unify the aesthetic strengths of both references into a cohesive, dark computational pathology platform:

| Visual Element | Design Token | Hex / HSL Value | Purpose & Application |
| :--- | :--- | :--- | :--- |
| **Canvas Background** | `--bg-base` | `#080c14` / `hsl(220, 43%, 6%)` | Deepest dark navy/black background |
| **Panel / Card Background** | `--bg-surface` | `#0d1527` / `hsl(220, 38%, 10%)` | Slightly lighter navy for cards & stage |
| **Elevated Surface** | `--bg-card-hover` | `#131e36` / `hsl(220, 36%, 15%)` | Card hover and active tab background |
| **Border / Grid Line** | `--border-subtle` | `rgba(56, 189, 248, 0.12)` | Subtle blue-gray borders & separators |
| **Border Highlight / Focus** | `--border-focus` | `rgba(45, 212, 191, 0.40)` | Teal/cyan glow for active containers |
| **Primary Accent (Teal)** | `--cyan-400` / `--teal-400` | `#2dd4bf` / `#14b8a6` | Confidence rings, primary buttons, highlights |
| **Secondary Accent (Cyan)** | `--cyan-500` | `#06b6d4` | Interactive icons, metric values, telemetry |
| **Positive Indicator** | `--emerald-400` | `#34d399` | 96% radial meter, verified checkpoints |
| **Violet / Molecular** | `--violet-400` | `#a78bfa` | Biomarker targets, spatial graph edges |
| **Warning / Attention** | `--amber-400` | `#fbbf24` | Scientific disclaimer banners, caveats |
| **Primary Text** | `--text-primary` | `#f8fafc` | Main headings, metric values |
| **Secondary Text** | `--text-secondary` | `#94a3b8` | Subheadings, card titles, metadata |
| **Muted Text** | `--text-muted` | `#64748b` | Disclaimers, axis labels, tooltips |

---

## 2. 8-Class Gastric Tissue Palette (Consistent Across All Visualizations)

| Code | Tissue Class Name | Hex Code | Visual Identity |
| :---: | :--- | :---: | :--- |
| `TUM` | Gastric Adenocarcinoma | `#ef4444` | High-visibility crimson / coral red |
| `STR` | Cancer-Associated Stroma | `#f97316` | Amber orange |
| `LYM` | Lymphocytes | `#06b6d4` | Bright cyan |
| `NOR` | Normal Gastric Mucosa | `#10b981` | Emerald green |
| `MUC` | Extracellular Mucus | `#8b5cf6` | Violet |
| `DEB` | Necrotic Debris | `#64748b` | Slate blue-gray |
| `MUS` | Smooth Muscle | `#ec4899` | Magenta / rose |
| `ADI` | Adipose (Fat) Tissue | `#eab308` | Gold yellow |

---

## 3. Structural Layout Mapping

### 3.1 Collapsible Left Sidebar (Inspired by Reference 2 & 1)
- **Top:** GastricPath AI logo with glowing cellular glyph and title.
- **Section 1: OVERVIEW** — Live Dashboard (Ref 1), Spatial Research (Ref 2), Recent Analyses.
- **Section 2: HISTOPATHOLOGY** — Sample Explorer, Heatmaps, Feature Extraction, Classifications.
- **Section 3: AI MODELS** — GastricCancerRNN, Phikon, DINOv2, Architecture Explorer.
- **Section 4: RESEARCH ANALYTICS** — Metrics & Confusion Matrix, Robustness, Uncertainty & OOD, Error Galleries, Cell Morphology, WSI Pyramid.
- **Section 5: MOLECULAR RESEARCH** — miRNA Targets (`hsa-miR-497-5p`), Consensus Matrix, Pathways.
- **Section 6: RESEARCH TOOLS** — Research Copilot, Clinical Reporting & PDF Workbench, 32-Paper Registry.
- **Bottom:** Status pill (`127.0.0.1:8000`), Research disclaimer, settings icon.

### 3.2 Main Overview Layout (Reference 1 Inspiration)
- **Header:** Breadcrumbs, dataset counter (`31,096`), model telemetry (`GastricCancerRNN`), validation acc (`64.86%`), historical benchmark (`71.04%`), `AI Analysis` Copilot button.
- **Row 1 Panel A (Top Left):** Histopathology Image Explorer:
  - Patch canvas with zoom/pan and coordinate crosshair reticle.
  - Grad-CAM heatmap overlay toggle with interactive opacity slider ($0\% - 100\%$).
  - Continuous gradient legend: `Heat probability: TUM [Low ──────── High]`.
  - Right-side vertical thumbnail strip: Original H&E, Grad-CAM, Integrated Gradients, Reticle Grid.
- **Row 1 Panel B (Top Right):** Model Architecture Explorer:
  - Interactive SVG architecture diagram: Input $(224 \times 224 \times 3) \to \text{ConvBlocks} \to \text{BiLSTM} \to \text{Self-Attention} \to \text{8-Class Logits}$.
  - Interactive layer nodes with tooltips (parameter count, activations, dimensions).
  - Mathematical operator bar at bottom: $\int \text{Conv}$, $\sum \text{Pool}$, $f \text{Act}$, $\times \text{Dense}$, $= \text{Logits}$.
- **Row 2 Panel C (Bottom Left):** Feature Extraction Workspace:
  - Texture analysis: Local Binary Pattern (LBP) histogram and Gray-Level Co-occurrence Matrix (GLCM) $4 \times 4$ heatmaps.
  - Stain & color deconvolution: Optical density distributions (Hematoxylin vs. Eosin).
  - Dimensionality reduction: PCA / UMAP 2D feature scatter across classes.
- **Row 2 Panel D (Bottom Right):** Classification & Model Confidence Dashboard:
  - 8-class horizontal probability distribution bars.
  - Glowing circular radial progress indicator: **Model Prediction Confidence** (e.g. $83.5\%$).
  - Case metadata card: Sample ID, tissue label, model checkpoint, inference latency.

### 3.3 Advanced Spatial Research Layout (Reference 2 Inspiration)
- **Top:** Breadcrumbs (`Projects > Gastric Cancer Study > Sample GC-01 > Spatial Overview`), `AI Analysis` Copilot button.
- **Center Canvas:**
  - Large interactive 2D spatial candidate-object viewer with real tissue patch, centroid scatter, and Delaunay/kNN graph edges ($k=6$).
  - Canvas toolbar: Toggle 2D / Graph View, Pan, Zoom, Reset.
  - Disclaimer pill: *Experimental candidate-object spatial visualization — not validated cell segmentation*.
  - Bottom thumbnail strip with 6 reference sample patches across classes (`TUM`, `STR`, `LYM`, `NOR`, `MUC`, `MUS`).
- **Right Panel:**
  - Tissue Class Composition: 8-class percentage pills + vertical multi-color stacked bar chart.
  - AI-Derived Insights: 4-card grid (`Total Patches: 31,096`, `Benchmark Acc: 71.04%`, `Candidate Objects: 1,280`, `Spatial Graph Edges: 3,840`).
  - Spatial Interaction Score: SVG line chart showing proximity distribution vs. distance ($\mu m$).
- **Bottom Panel:**
  - Molecular biomarker table: `hsa-miR-497-5p` targets (`SEMA6D` 8/8 consensus, `CCNE1` 7/8 consensus, `PTEN`, `EGFR`, `CDH1`).
  - Top Enriched Pathways: in-silico predictions table with adjusted p-values.
  - AI Research Summary: glowing neural network brain icon with grounded research synthesis.

---

## 4. Scientific Rigor & Honest Data Contract

1. All tissue classes are explicitly presented as **tissue morphology classes**, never individual cell types.
2. Radial meter is labeled **Model Prediction Confidence**, never clinical risk, diagnostic certainty, or tumor purity.
3. Spatial graphs are designated as **candidate-object spatial graphs**, not verified cell-cell biological interactions.
4. All metrics are bound to verified backend endpoints in FastAPI.
