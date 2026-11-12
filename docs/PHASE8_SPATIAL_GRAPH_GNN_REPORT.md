# GastricPath AI — Original Phase 8: Spatial Graph and GNN Engine Research Report

**Document Version:** 1.0.0  
**Phase Identifier:** Original Phase 8 (Spatial Graph and GNN Engine)  
**Status:** Completed  
**Author:** GastricPath AI Research Team  
**Date:** November 2026  
**Cohort Provenance:** N=64 histological image patches (8 per canonical GCHTID class); 3,895 automated candidate segmented objects from Original Phase 7 (`research/cellular/`).  
**Baseline Model Invariant:** `models/checkpoints/gastric_rnn_classifier.pt` (SHA-256: `65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8`, verified intact).

---

## 1. Executive Summary and Candidate-Object Graph Provenance

Original Phase 8 establishes a graph-based spatial learning engine for gastric histopathology. In contrast to conventional tile-level convolutional or vision transformer encoders that operate on continuous raster grids, this phase models the spatial architecture of gastric tissue as a discrete topological graph.

> [!IMPORTANT]
> **Candidate-Object Graph Provenance Disclaimer:**
> The graph nodes constructed in this phase represent **automated candidate segmented objects** extracted via classical Ruifrok–Johnston HED color deconvolution and distance-transform watershed segmentation from Original Phase 7. They do **not** represent manually annotated, pathologist-verified ground-truth nuclei, nor do the edges imply confirmed biological or paracrine cell-to-cell signaling. Throughout this report and codebase, all graphs are strictly designated **"candidate-object graphs"**, and edges represent Euclidean spatial proximity.

This work is distinct from the previously completed supplementary Phase 8 work package (*Intra-Patch Spatial Representation Analysis*, which examined Phikon ViT-B/16 token grids). Original Phase 8 executes the master blueprint's core mandate for cellular spatial graph modeling.

### Core Deliverables Achieved
1. **Standardized 16-Dimensional Node Representation:** Extracted geometry, orientation harmonics ($\sin 2\theta, \cos 2\theta$), color intensity, and HED optical densities for all 3,895 candidate objects.
2. **Symmetric $k$-NN Spatial Graph Engine:** Built symmetric $k$-nearest neighbor ($k=6$) graphs with Euclidean distance attributes and self-loops for message-passing stability.
3. **Graph Neural Network Benchmark Suite:** Implemented lightweight 2-layer Graph Convolutional Networks (GCN; Kipf & Welling 2017), Graph Attention Networks (GAT; Veličković et al. 2018), and GraphSAGE (Hamilton et al. 2017), alongside an aggregated non-graph Morphology MLP baseline.
4. **Stratified 5-Fold Cross-Validation:** Conducted rigorous cross-validation across all 64 images, with feature scalers fitted strictly within training folds to eliminate data leakage.
5. **Topological Profiling:** Quantified distinct spatial density regimes across the 8 canonical GCHTID tissue types.
6. **Publication Visualizations & Test Suite:** Rendered 4 publication-grade figures and verified 13 unit/integration tests with 100% pass rate.

---

## 2. Theoretical Grounding and Literature Review

### 2.1 Graph Convolutional Networks (GCN)
As formulated by Kipf & Welling (ICLR 2017), spectral graph convolutions can be approximated via localized first-order Chebyshev polynomials. For an adjacency matrix $\mathbf{A} \in \mathbb{R}^{N \times N}$ with added self-loops $\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}_N$ and degree matrix $\tilde{\mathbf{D}}_{ii} = \sum_j \tilde{\mathbf{A}}_{ij}$, the layer-wise propagation rule is:

$$\mathbf{H}^{(l+1)} = \sigma\left(\tilde{\mathbf{D}}^{-\frac{1}{2}} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-\frac{1}{2}} \mathbf{H}^{(l)} \mathbf{W}^{(l)}\right)$$

In histopathology, symmetric normalization performs isotropic spatial smoothing over candidate objects, averaging local morphological signals across immediate neighbors.

### 2.2 Graph Attention Networks (GAT)
Veličković et al. (ICLR 2018) introduced masked self-attentional layers to address the isotropic limitation of GCNs. Rather than weighting all adjacent neighbors equally or by fixed degree, GAT dynamically assigns attention coefficients $\alpha_{ij}$:

$$\alpha_{ij} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^\top [\mathbf{W} \mathbf{h}_i \,\|\, \mathbf{W} \mathbf{h}_j]\right)\right)}{\sum_{k \in \mathcal{N}_i} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^\top [\mathbf{W} \mathbf{h}_i \,\|\, \mathbf{W} \mathbf{h}_k]\right)\right)}$$

Multi-head attention stabilizes learning by computing $K$ independent attention distributions and concatenating or averaging their outputs.

### 2.3 GraphSAGE (Sample and Aggregate)
Hamilton et al. (NeurIPS 2017) formulated an inductive framework that learns neighborhood aggregator functions rather than transductive node embeddings. For a target candidate object $u$, neighbor representations are aggregated:

$$\mathbf{h}_{\mathcal{N}(u)}^{(l)} = \text{AGGREGATE}_k\left(\{\mathbf{h}_v^{(l-1)}, \forall v \in \mathcal{N}(u)\}\right)$$
$$\mathbf{h}_u^{(l)} = \sigma\left(\mathbf{W}^{(l)} \cdot [\mathbf{h}_u^{(l-1)} \,\|\, \mathbf{h}_{\mathcal{N}(u)}^{(l)}]\right)$$

Using mean aggregation (`aggr='mean'`), GraphSAGE computes uniform localized feature averages concatenated with the ego-node representation, providing strong inductive generalization.

### 2.4 Graph Readout and Global Pooling
To perform graph-level classification under weak patch-level supervision (classifying the $224 \times 224$ tissue patch from candidate objects), node representations from the final layer $\mathbf{h}_i^{(L)}$ are pooled into a single graph vector $\mathbf{h}_G$:

$$\mathbf{h}_G = \text{global\_mean\_pool}(\{\mathbf{h}_i^{(L)}\}_{i=1}^N) = \frac{1}{N} \sum_{i=1}^N \mathbf{h}_i^{(L)}$$

A linear projection head then maps $\mathbf{h}_G \in \mathbb{R}^{d}$ to class logits $\mathbf{z} \in \mathbb{R}^8$.

---

## 3. Spatial Graph Construction and Feature Engineering

### 3.1 16-Dimensional Standardized Node Features
Each candidate segmented object in an image patch is parameterized by a continuous 16-dimensional feature vector $\mathbf{x}_i \in \mathbb{R}^{16}$:

| Index | Feature Name | Description | Normalization / Transform | Bounds |
|---|---|---|---|---|
| 0 | `centroid_x_norm` | Horizontal spatial centroid | $x / 224.0$ | $[0, 1]$ |
| 1 | `centroid_y_norm` | Vertical spatial centroid | $y / 224.0$ | $[0, 1]$ |
| 2 | `log_area` | Log-transformed object area | $\log(1 + \text{area}_{\text{px}^2})$ | $[0, \infty)$ |
| 3 | `perimeter_norm` | Object boundary perimeter | $\text{perimeter}_{\text{px}} / 100.0$ | $[0, \infty)$ |
| 4 | `equivalent_diameter_norm` | Diameter of circle with equivalent area | $d_{\text{eq}} / 50.0$ | $[0, \infty)$ |
| 5 | `aspect_ratio` | Major axis / minor axis ratio | $\text{clip}(a/b, 1.0, 10.0)$ | $[1, 10]$ |
| 6 | `eccentricity` | First eccentricity of fitted ellipse | $\sqrt{1 - (b/a)^2}$ | $[0, 1]$ |
| 7 | `solidity` | Area / convex hull area | $\text{area} / \text{convex\_area}$ | $[0, 1]$ |
| 8 | `circularity` | Isoperimetric shape factor | $4\pi \cdot \text{area} / \text{perimeter}^2$ | $[0, 1]$ |
| 9 | `sin_2orient` | Harmonic orientation component | $\sin(2\theta_{\text{rad}})$ | $[-1, 1]$ |
| 10 | `cos_2orient` | Harmonic orientation component | $\cos(2\theta_{\text{rad}})$ | $[-1, 1]$ |
| 11 | `intensity_r_norm` | Mean Red channel intensity | $I_R / 255.0$ | $[0, 1]$ |
| 12 | `intensity_g_norm` | Mean Green channel intensity | $I_G / 255.0$ | $[0, 1]$ |
| 13 | `intensity_b_norm` | Mean Blue channel intensity | $I_B / 255.0$ | $[0, 1]$ |
| 14 | `hematoxylin_od` | Ruifrok optical density (nuclear stain) | $\max(\text{OD}_H, 0)$ | $[0, \infty)$ |
| 15 | `eosin_od` | Ruifrok optical density (cytoplasmic stain) | $\max(\text{OD}_E, 0)$ | $[0, \infty)$ |

### 3.2 Graph Topology and Edge Generation
1. **Centroid Proximity Metric:** Pairwise Euclidean distances $d(u, v) = \|\mathbf{c}_u - \mathbf{c}_v\|_2$ are computed across all candidate object centroids within each patch.
2. **Symmetric $k$-NN:** For each node $u$, edges are formed to its $k=6$ nearest neighbors ($v \in \mathcal{N}_u$). To ensure an undirected graph structure, symmetric closure is applied: an edge exists between $u$ and $v$ if $v \in \mathcal{N}_u$ or $u \in \mathcal{N}_v$. In the directed PyG representation, both $(u, v)$ and $(v, u)$ are included.
3. **Normalized Edge Attributes:** Edge attribute $e_{uv} = d(u, v) / D_{\text{tile}}$, where $D_{\text{tile}} = \sqrt{224^2 + 224^2} \approx 316.78\text{ px}$.
4. **Self-Loops:** Self-edges $(i, i)$ with distance $0.0$ are added to preserve ego-node identity during message passing.
5. **Strict Patch Separation:** Graphs are constructed independently per image tile. No inter-tile edges exist.

---

## 4. Cohort Topological and Spatial Findings

The pilot cohort of 64 images (8 samples per class) contains 3,895 detected candidate instances. Topological metrics extracted from the candidate-object graphs reveal clear architectural differences across the 8 tissue categories:

### Cohort Topology Summary Table

| Class | Node Count ($N$, Mean ± SD) | Undirected Edges ($E$, Mean ± SD) | Mean Degree (Mean ± SD) | Graph Density | Mean Edge Dist (px, Mean ± SD) | Clustering Coeff |
|---|---|---|---|---|---|---|
| **ADI** (Adipose) | $16.75 \pm 12.58$ | $61.38 \pm 51.24$ | $6.61 \pm 1.64$ | $0.646 \pm 0.343$ | $61.16 \pm 22.47$ | $0.853 \pm 0.136$ |
| **DEB** (Debris) | $70.88 \pm 44.60$ | $254.88 \pm 155.60$ | $7.26 \pm 0.21$ | $0.177 \pm 0.165$ | $37.93 \pm 18.17$ | $0.645 \pm 0.091$ |
| **LYM** (Lymphocytes) | $\mathbf{132.88 \pm 24.15}$ | $\mathbf{470.50 \pm 84.88}$ | $7.09 \pm 0.11$ | $0.056 \pm 0.013$ | $\mathbf{21.51 \pm 1.32}$ | $0.565 \pm 0.039$ |
| **MUC** (Mucus) | $24.88 \pm 24.72$ | $88.38 \pm 88.54$ | $6.58 \pm 1.71$ | $0.574 \pm 0.362$ | $50.98 \pm 19.52$ | $0.803 \pm 0.158$ |
| **MUS** (Muscle) | $32.12 \pm 21.74$ | $118.25 \pm 79.14$ | $7.26 \pm 0.59$ | $0.415 \pm 0.344$ | $57.55 \pm 24.12$ | $0.730 \pm 0.150$ |
| **NOR** (Normal) | $80.00 \pm 28.46$ | $287.50 \pm 98.58$ | $7.23 \pm 0.16$ | $0.110 \pm 0.061$ | $28.23 \pm 5.88$ | $0.632 \pm 0.061$ |
| **STR** (Stroma) | $61.75 \pm 33.27$ | $223.00 \pm 116.09$ | $7.29 \pm 0.18$ | $0.167 \pm 0.124$ | $34.95 \pm 8.86$ | $0.641 \pm 0.066$ |
| **TUM** (Tumor) | $67.62 \pm 43.11$ | $242.12 \pm 157.51$ | $6.26 \pm 2.54$ | $0.101 \pm 0.067$ | $28.42 \pm 14.10$ | $0.531 \pm 0.217$ |

### Topological Characterization
- **LYM (Dense Cellularity):** Demonstrates the highest candidate node count ($132.88 \pm 24.15$) and the shortest edge proximity ($21.51 \pm 1.32\text{ px}$). Candidate nuclei form compact, tightly packed spatial configurations.
- **ADI & MUC (Acellular Hypocellularity):** Adipose tissue is dominated by lipid droplets with sparse peripheral nuclei, exhibiting the lowest candidate count ($16.75 \pm 12.58$) and the largest edge distances ($61.16 \pm 22.47\text{ px}$). Mucus pools display similarly sparse configurations ($24.88 \pm 24.72$ nodes).
- **TUM & STR (Organized Cellular Patterns):** Tumor patches exhibit moderate-to-high candidate counts ($67.62 \pm 43.11$) with an intermediate edge proximity ($28.42\text{ px}$), reflecting glandular disruption and cellular proliferation.

---

## 5. Experimental Protocol and Cross-Validation Results

### 5.1 Protocol and Leakage Prevention
- **Validation Protocol:** Stratified 5-Fold Cross-Validation ($N=64$ images, ~12–13 images per fold, balanced across 8 classes).
- **Feature Normalization Without Leakage:** To ensure zero test leakage, a `StandardScaler` was fitted strictly on the candidate objects from the training fold in each iteration, and applied to the validation fold.
- **Training Setup:** All models trained for 30 epochs with Adam optimizer ($\text{lr} = 0.01$, $\text{weight decay} = 10^{-4}$) using `CrossEntropyLoss`.
- **Random Baseline Reference:** In an 8-class balanced problem, random guessing yields $1 / 8 = 12.5\%$ accuracy.

### 5.2 Comparative Performance Benchmark

| Model | Architecture Type | Parameters | Accuracy (Mean ± SD) | Balanced Acc (Mean ± SD) | Macro F1 (Mean ± SD) |
|---|---|---|---|---|---|
| **Random Baseline** | Uniform Chance | 0 | 0.1250 | 0.1250 | 0.1250 |
| **GAT** | Graph Attention Network | 1,992 | $0.1410 \pm 0.0850$ | $0.1750 \pm 0.1118$ | $0.1017 \pm 0.0607$ |
| **GraphSAGE** | Inductive Mean Aggregation | 3,400 | $0.1897 \pm 0.0970$ | $0.1875 \pm 0.1169$ | $0.1471 \pm 0.0841$ |
| **GCN** | Graph Convolutional Network | 1,864 | $\mathbf{0.2051 \pm 0.0790}$ | $\mathbf{0.2125 \pm 0.1046}$ | $\mathbf{0.1550 \pm 0.0644}$ |
| **Morphology MLP** | Non-Graph Summary Baseline | 5,480 | $\mathbf{0.2205 \pm 0.0739}$ | $\mathbf{0.2375 \pm 0.0927}$ | $\mathbf{0.1875 \pm 0.0745}$ |

### 5.3 Per-Class Sensitivity Analysis (Best GNN vs Baseline)

| Class | GCN Recall (Sensitivity) | Morphology MLP Recall | Key Failure Mode / Confusion |
|---|---|---|---|
| **ADI** | 25.0% | 37.5% | Occasionally confused with MUC due to low nuclear density |
| **DEB** | 0.0% | 25.0% | Frequently misclassified as LYM or NOR due to high fragment count |
| **LYM** | 12.5% | 25.0% | Confused with DEB and NOR |
| **MUC** | 12.5% | 12.5% | Confused with MUS and ADI |
| **MUS** | 25.0% | 12.5% | Confused with DEB and MUC |
| **NOR** | 12.5% | 25.0% | Moderate confusion with DEB |
| **STR** | 37.5% | 25.0% | Distinct elongated morphology aids identification |
| **TUM** | 37.5% | 25.0% | Hyperchromatic, enlarged candidate nuclei correctly captured |

---

## 6. Scientific Discussion: Why Non-Graph Baseline Outperforms GNNs on Pilot Data

An essential scientific observation from this experiment is that the **non-graph Morphology MLP baseline** ($23.75\%$ balanced accuracy) achieved slightly higher performance than the best GNN architecture (**GCN**, $21.25\%$), while both GAT ($17.50\%$) and GraphSAGE ($18.75\%$) showed higher variance.

Several structural factors account for this finding in weakly supervised pilot cohorts:

1. **Weak Graph-Level Supervision with Small Sample Size ($N=64$):**
   GNNs are parameterized to learn relational representations across neighborhoods. With only 64 total graphs (51 training graphs per fold), the signal-to-noise ratio for message-passing weights is low. GNN layers easily overfit or suffer from oversmoothing when trained end-to-end with graph-level cross-entropy loss.
2. **First-Order Bulk Morphology Suffices for Coarse Discrimination:**
   Distinguishing adipose tissue from lymphocytic infiltration does not fundamentally require 2-hop message-passing; it requires knowing the *total object count* ($16.8$ vs $132.9$) and *mean optical density*. The 48-dimensional aggregated vector (mean, standard deviation, and median of all 16 morphological features) directly encodes this global distribution, providing the MLP with an uncorrupted summary of tissue density.
3. **Automated Candidate-Object Noise:**
   Because instances were segmented using classical watershed without manual nuclear boundary verification, edge relationships contain noise (e.g., fragmented debris or oversegmented cytoplasm). GNN message passing propagates this noise to neighboring nodes, whereas global median and mean pooling dampens isolated local segmentation artifacts.

---

## 7. Publication Visualizations

All publication figures were rendered at 300 DPI using headless Matplotlib and saved to `research/spatial_graph/visualizations/`:

### 7.1 Exemplar 8-Class Candidate-Object Graph Gallery
The figure displays one representative $224 \times 224$ H&E patch per canonical tissue class overlaid with its candidate-object spatial graph (cyan centroids, yellow Euclidean edges).

[![8-Class Candidate-Object Spatial Graph Gallery](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/spatial_graph/visualizations/spatial_graph_8class_gallery.png)](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/spatial_graph/visualizations/spatial_graph_8class_gallery.png)
*Figure 1: Candidate-object spatial graph overlays ($k=6$) across the 8 canonical GCHTID classes.*

### 7.2 Topological and Spatial Distributions
Panel A depicts the distribution of mean node degrees across the 64 pilot graphs; Panel B depicts the distribution of edge Euclidean distances (in pixels).

[![Graph Topology Distributions](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/spatial_graph/visualizations/graph_topology_distributions.png)](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/spatial_graph/visualizations/graph_topology_distributions.png)
*Figure 2: Distribution of node degrees and edge distances ($N=64$ pilot cohort).*

### 7.3 Cross-Validation Performance Benchmark
Grouped bar chart comparing Accuracy, Balanced Accuracy, and Macro F1 across GCN, GAT, GraphSAGE, and Morphology MLP baseline with fold standard deviation error bars.

[![GNN Cross-Validation Benchmark Comparison](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/spatial_graph/visualizations/gnn_cv_benchmark_comparison.png)](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/spatial_graph/visualizations/gnn_cv_benchmark_comparison.png)
*Figure 3: Stratified 5-fold cross-validation performance comparison across models.*

### 7.4 Confusion Matrix (Best GNN: GCN)
Normalized confusion matrix heatmap for the GCN model across the 8 tissue categories.

[![Confusion Matrix GCN](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/spatial_graph/visualizations/confusion_matrix_best_gnn.png)](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/spatial_graph/visualizations/confusion_matrix_best_gnn.png)
*Figure 4: Normalized pooled confusion matrix for the GCN model (5-fold CV).*

---

## 8. Artifact and File Manifest

The implementation generated the following validated files:

1. **`research/spatial_graph/graph_features.py`**: 16-D node feature extractor and 48-D image summary aggregator.
2. **`research/spatial_graph/graph_builder.py`**: Symmetric $k$-NN spatial graph builder with topological stats computation.
3. **`research/spatial_graph/graph_serialization.py`**: `.pt` graph file persistence, manifest creation, and data loader.
4. **`research/spatial_graph/gnn_models.py`**: GCN, GAT, GraphSAGE, and Morphology MLP architectures.
5. **`research/spatial_graph/graph_visualization.py`**: Publication visualization generators.
6. **`research/spatial_graph/config.json`**: Declarative graph schema, architecture, and CV hyperparameter configuration.
7. **`research/spatial_graph/evaluation.py`**: Master pipeline and 5-fold CV runner.
8. **`research/spatial_graph/graphs/*.pt`**: 64 serialized PyTorch Geometric graph objects.
9. **`research/spatial_graph/spatial_graph_manifest.csv`**: Manifest with image IDs, paths, node counts, edge counts, and graph metrics.
10. **`research/spatial_graph/graph_summary_statistics.csv`**: Aggregated topological statistics by class.
11. **`research/spatial_graph/cross_validation_results.csv`**: Per-fold metrics across all 4 architectures.
12. **`research/spatial_graph/evaluation_summary.json`**: Consolidated benchmark results and metadata.
13. **`tests/test_original_phase8.py`**: Test suite with 13 automated tests.

---

## 9. Limitations and Future Research Directions

1. **Candidate-Object Nature of Segmentations:**
   The nodes represent candidate objects identified by classical watershed rather than manual expert nuclear annotations. When StarDist or HoVer-Net weights become accessible or supervised annotations are supplied, nodes should be replaced with confirmed nuclear instances.
2. **Cohort Scale ($N=64$):**
   The current pilot evaluated 64 images (8 per class). While sufficient to prove pipeline operability, detect topological differences, and execute 5-fold CV without leakage, expanding to larger cohorts ($N \ge 1,000$) is recommended to provide GNNs with adequate training supervision.
3. **Multi-Scale Spatial Modeling:**
   Future iterations should evaluate hierarchical graphs that connect candidate nuclear nodes to tissue-level superpixels or patch-level embeddings, bridging cellular morphology with WSI-level architectural context.

