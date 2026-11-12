# GastricPath AI — Phase 5–12 Research & Implementation Blueprint

**Document type:** Research and engineering execution specification  
**Status:** Planning baseline for Antigravity  
**Applies to:** Work beginning after Phase 4.4 has completed and its final evaluation artifacts have been verified  
**Primary objective:** Complete Phases 5–12 with a credible, reproducible, CPU-conscious research contribution—without expanding the project into an open-ended platform.

---

## 0. Executive direction

GastricPath AI should now move from repeated model experimentation toward a compact research and product-completion sequence. The remaining phases should reuse the existing GCHTID dataset, trained classifiers, Phikon and DINOv2 representations, FastAPI application, and existing research artifacts.

The recommended research contribution is:

> **An image-patch-level study of gastric histopathology classification using generic and pathology-pretrained representations, accompanied by lightweight explanation, controlled robustness, and novelty-screening analyses.**

This is deliberately narrower than a whole-slide computational pathology system. The available GCHTID data are independent 224×224 image patches, and the known files do not provide a verified patient/slide crosswalk. Consequently, this project must not claim patient-level generalization, whole-slide tissue composition, tumor-area estimation, clinical validity, or biomarker association.

### Phase completion map

| Phase | Theme | Minimum useful output | Effort target |
|---|---|---|---|
| 5 | Explainable AI | A small, class-balanced explanation set plus faithfulness check | Low–medium |
| 6 | Robustness | Four controlled perturbation tests on a fixed evaluation subset | Low |
| 7 | OOD / uncertainty | A simple distance score and validation-calibrated rejection threshold | Low–medium |
| 8 | Intra-patch spatial analysis | Token-grid / regional analysis on a small sample; no new large model | Low |
| 9 | Research error analysis | Class-wise error audit and curated correct/error examples | Low |
| 10 | Dashboard integration | Read-only research results and explanation views in the existing app | Medium |
| 11 | Final verification | Essential regression, artifact and reproducibility checks | Low–medium |
| 12 | Release | Final report, README, demo evidence and final commit | Low |

**One phase means one deliverable, not a chain of micro-phases.** Each phase should normally have one implementation pass, one focused test pass, and one concise report/update. Do not create separate numbered subphases unless a concrete technical blocker requires one.

---

## 1. Project facts and constraints that every phase must respect

### 1.1 Dataset and labels

- Dataset: GCHTID, 31,096 image patches, 224×224 pixels, eight balanced classes (ADI, DEB, LYM, MUC, MUS, NOR, STR, TUM).
- Existing research split: train 21,767; validation 4,664; test 4,665, based on the existing manifest and duplicate-group-aware splitting work.
- Existing exact/perceptual duplicate analysis found duplicate groups. Reuse the established manifest and split assignments; never create a new random split for convenience.
- These are **image patches**, not verified independent patients or slides.
- The project has a 300-patient clinical CSV and a separate biomarker workbook, but no verified image-to-patient or image-to-biomarker mapping. Do not join by row order, filenames, class counts, or guesswork.

### 1.2 Existing models and research assets

- Historical gastric CNN-RNN baseline checkpoint: `models/checkpoints/gastric_rnn_classifier.pt`.
- DINOv2 ViT-S/14 frozen CLS and mean-patch embedding caches from Phase 4.
- Phikon (`owkin/phikon`) ViT-B/16 frozen embeddings from Phase 4.4, subject to the actual completion and integrity report.
- Existing Phase 4 linear-probe code, evaluation utilities, manifests, tests and reports should be reused rather than rewritten.
- Historical CNN-RNN results have a different test population from the newer Phase 4 representation benchmark. Keep historical numbers as a separately labelled reference unless identical sample-level predictions are available.

### 1.3 Hardware / execution constraints

The development machine is Windows 11, Python 3.13, CPU-only PyTorch, with limited immediately available RAM (roughly 1.3 GB was reported during the Phikon extraction) and a finite disk budget. Therefore:

- Do not run a second heavy extraction, full fine-tuning, or large-scale pixel attribution sweep.
- Do not cache all Phikon patch tokens for all 31,096 images. At 196 tokens × 768 dimensions × float32, this would require about 18.7 GB before metadata and overhead.
- Use existing embedding caches wherever possible.
- Run explanation and spatial-token extraction on a small, deterministic subset.
- Use batch size 1–4 for model-forward explanation work unless a measured pilot demonstrates safe headroom.
- Record actual runtime and peak RSS for any new CPU-heavy operation; do not promise estimated timings without measuring.

### 1.4 Scientific language

Use these terms consistently:

- “Patch-level classification” rather than “patient diagnosis”.
- “Image-level split” rather than “patient-independent split”.
- “Synthetic perturbation robustness” rather than “scanner robustness” unless real multi-scanner data exist.
- “OOD proxy / novelty screening” rather than “validated OOD detection” when the negatives are synthetic corruptions.
- “Attention/attribution visualization” rather than “proof of model reasoning”.
- “Research prototype” rather than “clinical decision support system”.

---

## 2. Shared experimental protocol

### 2.1 Immutable evaluation contract

1. Read and preserve the authoritative manifest, class mapping, split files and duplicate-group logic established in earlier phases.
2. Use the final Phase 4.4 evaluation configuration and predictions as the source of truth for the selected primary classifier.
3. The test labels must not be used to select thresholds, perturbation severities, explanation methods, OOD cutoffs, UI design, or model checkpoints.
4. Use train data to fit statistical reference quantities (e.g. class means/covariance) and validation data to calibrate thresholds and select design choices.
5. Evaluate the held-out test set only for final reporting. For exploratory visualization, use a fixed subset selected before inspecting outcomes and report its sampling rule.
6. Where a phase uses a reduced subset for practical reasons, select it deterministically and stratify by class and split. Save the exact sample IDs in a manifest.
7. Never compare predictions from different sample sets with paired statistical tests.

### 2.2 Reusable output conventions

Suggested directory pattern (adapt to repository conventions after inspecting existing directories):

```text
research/
  explainability/
  robustness/
  ood/
  spatial_analysis/
  error_analysis/
  outputs/
    phase5/
    phase6/
    phase7/
    phase8/
    phase9/
```

Each phase should store:

- `config.json` or YAML with seed, model/checkpoint hash, sample IDs, preprocessing and method parameters.
- A compact CSV/JSON with per-image and aggregate measurements.
- PNG plots or example overlays only where useful.
- A concise `docs/PHASE<N>_..._REPORT.md` (prefer one report per phase).

Do not duplicate full source datasets, checkpoint weights or large feature arrays into report folders.

---

# 3. Phase specifications

## Phase 5 — Explainable AI for patch-level predictions

### Objective
Create a small, reproducible explanation gallery for the final selected Phikon classifier (or the best Phase 4.4 model, if Phikon does not win), and test whether the visualized regions materially affect the predicted score.

### Research basis
- Attention rollout/flow estimates information flow across transformer layers; raw attention alone is not a complete explanation. Abnar and Zuidema describe rollout and attention flow as approximations and compare them with ablation/gradient importance.
- Chefer et al. propose relevance propagation through transformer attention and skip connections, rather than treating attention weights as sufficient explanations.
- Jain and Wallace caution that attention weights should not automatically be treated as explanations.
- Wu et al. (CVPR 2024) specifically study faithfulness of Vision Transformer explanations, reinforcing the need for evaluation beyond attractive heatmaps.

### Recommended implementation
1. Inspect the actual cached Phikon model, Transformers version and output API. Verify whether per-layer attention tensors can be returned with `output_attentions=True`. Do not assume the installed implementation exposes them.
2. For a small fixed subset (recommended: 4–5 images per class, 32–40 total, selected from the held-out test split using a recorded seed), generate one of:
   - attention rollout from the CLS token, if attention tensors are available and verified; or
   - gradient-weighted/transformer relevance method only if supported by the actual architecture and implementation; otherwise
   - a simpler input/patch occlusion attribution tied to the final classifier score.
3. Preserve the original 224×224 image and render a clearly labelled overlay. Use a fixed color scale and document interpolation from 14×14 token grid to image resolution.
4. Run a lightweight deletion test: progressively mask the highest-ranked regions and compare the target-class score with (a) random-region deletion and (b) lowest-ranked-region deletion. Use a small number of steps (e.g. 0%, 20%, 40%, 60%, 80%) on the same 32–40 samples.
5. Report average score drop / deletion curve and example-level variance. Do not call the map “faithful” solely because it looks plausible.
6. Include examples across all eight classes, not only TUM or correctly classified images. Mark misclassified cases explicitly.

### Required outputs
- `research/explainability/` code and deterministic sample manifest.
- A compact set of overlays and a contact sheet.
- Deletion-vs-random comparison summary (CSV/JSON and one plot).
- `docs/PHASE5_EXPLAINABILITY_REPORT.md`.

### Acceptance criteria
- The explanation method matches the actual loaded model and its returned tensors.
- Every image ID and target class is traceable.
- The deletion experiment includes a random baseline and uses fixed perturbation settings.
- No clinical/pathologist interpretation is asserted without expert annotations.
- Explanation maps are clearly labelled as model attribution, not tissue segmentation.

### If blocked
If Phikon attention extraction requires substantial model refactoring or unavailable attention tensors, do not spend days implementing custom transformer LRP. Use a small perturbation-based occlusion analysis, or use the CNN-RNN Grad-CAM where the convolutional feature layer is available. Record the limitation and proceed.

---

## Phase 6 — Controlled robustness analysis

### Objective
Measure how the selected Phase 4.4 classifier changes under a small number of realistic-but-synthetic image perturbations. This is a stress test, not external-domain validation.

### Research basis
- Histopathology images exhibit stain variation from preparation and laboratory processes. Macenko et al. introduced stain normalization in optical-density space; later work evaluates stain normalization methods and their effects on downstream pathology tasks.
- The robustness literature distinguishes synthetic corruption tests from external validation on genuinely independent scanners, laboratories and cohorts.

### Recommended implementation
Use the same fixed test sample IDs for clean and transformed predictions. For runtime efficiency, a deterministic stratified subset of approximately 400–800 test patches is acceptable, but the subset must be chosen before running the experiment and saved to disk.

Run four perturbation families with one predeclared severity each:

1. **Brightness/contrast shift:** moderate, fixed transformation parameters.
2. **Gaussian blur:** one mild blur level (document sigma/kernel).
3. **JPEG compression:** one quality level representing visible compression but not destroying the image.
4. **Stain-like color shift:** either implement a documented optical-density/H&E-vector perturbation using a tested method, or use a clearly labelled simple color-channel perturbation. Do not call generic HSV jitter “Macenko stain augmentation”.

Apply the exact preprocessing expected by the selected model after image transformation. Do not refit the model or tune parameters on transformed test images.

### Metrics
- Accuracy and macro-F1 on clean and each transformed subset.
- Per-class recall change.
- Prediction consistency / label-flip rate between clean and transformed versions.
- Mean change in maximum predicted probability (descriptive only; not proof of calibration).
- Optional paired confidence interval for clean-minus-corrupted accuracy if the implementation is already available.

### Required outputs
- `research/robustness/` transformation functions, fixed config and subset manifest.
- One table comparing clean and perturbed results.
- One plot of performance degradation and one class-level summary.
- `docs/PHASE6_ROBUSTNESS_REPORT.md`.

### Acceptance criteria
- Same image IDs in clean and perturbation arms.
- Perturbation operations and parameters are deterministic and saved.
- Clean performance is reproduced on the chosen subset before interpreting corruption effects.
- Conclusions use “under simulated perturbation” wording.

### Scope guard
Do not implement a full Macenko/Vahadane normalization benchmark, multiple severities for every transform, adversarial attacks, or multi-domain adaptation in this project-completion cycle. Those are future research.

---

## Phase 7 — Lightweight OOD / uncertainty screening

### Objective
Add a research-only novelty score that can flag feature vectors unlike the training distribution. This is an exploratory rejection mechanism, not a validated clinical safety function.

### Research basis
Lee et al. (NeurIPS 2018) describe class-conditional Gaussian feature distributions and Mahalanobis distance for detecting abnormal samples. Their method motivates a simple feature-space distance baseline, but performance is dataset- and representation-dependent. OOD evaluation should report detection metrics and clearly define what counts as in-distribution and out-of-distribution.

### Recommended implementation
1. Use the winning frozen embedding representation from Phase 4.4 (not a newly trained encoder).
2. Compute each class mean from train embeddings only.
3. Estimate a numerically stable covariance using a shrinkage or diagonal estimator; inspect conditioning and avoid unstable full covariance inversion without regularization.
4. For each vector, calculate the minimum class-conditional Mahalanobis distance:

   `score(x) = min_c (x - mu_c)^T Sigma^{-1} (x - mu_c)`

   Higher values indicate greater distance from the fitted class distributions.
5. Use validation in-distribution scores to set a predeclared rejection threshold (for example, a selected upper validation percentile). Freeze it before test evaluation.
6. Compare against maximum softmax probability as a simple baseline if probabilities are available.
7. For a limited proxy-OOD assessment, use only the controlled corruptions created in Phase 6 and label them **synthetic OOD proxies**. Do not present them as unseen clinical tissue classes.
8. If resources permit, compute AUROC and FPR@95% TPR for ID-vs-proxy separation, with exact definition of score direction. Also report in-distribution false-rejection rate at the selected threshold.

### Important distinctions
- A low maximum softmax probability is not equivalent to OOD detection.
- Mahalanobis distance can respond to class overlap, noise, stain changes and feature geometry; it does not establish biological novelty.
- Do not train an “unknown” class from the existing eight labels.
- Do not claim the model can reject all unseen tissue types.

### Required outputs
- `research/ood/` fitting and scoring code plus saved train/validation parameters.
- Threshold and per-sample scores in CSV/JSON.
- ID-vs-synthetic-proxy OOD plot (only if proxy data are available).
- `docs/PHASE7_OOD_UNCERTAINTY_REPORT.md`.

### Acceptance criteria
- Means/covariance are fitted on train only.
- Threshold is calibrated on validation only.
- Test is scored once with frozen parameters.
- False rejection and proxy-OOD performance are reported separately.
- App labels use “novelty score” or “research OOD flag”, not “safe/unsafe diagnosis”.

---

## Phase 8 — Intra-patch spatial representation analysis

### Objective
Investigate whether the selected pathology representation preserves useful within-patch spatial variation beyond a global CLS or mean-patch vector. This analysis is limited to a 224×224 patch and its 14×14 token grid.

### Research basis
Vision Transformers tokenize an image into spatial patches. In Phikon ViT-B/16 at 224×224, a 16-pixel patch stride corresponds to a 14×14 grid (196 spatial tokens), subject to verification from the loaded configuration. Spatial token patterns are not whole-slide coordinates and do not encode relationships between separate GCHTID files.

### Recommended implementation
1. Reuse the Phikon model and only request the last-layer spatial tokens for a small deterministic subset (recommended 8–16 images per class; 64–128 total). Do not store all token tensors for the full dataset.
2. Compare three compact summaries from the same forward pass:
   - CLS embedding;
   - global mean of the 196 patch tokens;
   - 2×2 regional pooling (four coarse quadrants, with a fixed and documented token-grid partition).
3. For spatial visualization, render per-token feature norm or the selected Phase 5 attribution score as a 14×14 heatmap. Do not label feature magnitude as “tumor probability”.
4. Summarize within-image spatial variation using simple descriptors (e.g. quadrant embedding distances, token-norm dispersion, or similarity between regional means). Avoid fitting a new GNN or large attention head.
5. If classification probes are already easy to train on the cached subset, run a small train/validation-only comparison and do not retune on test. Otherwise, keep Phase 8 descriptive and do not add a second benchmark.

### Required outputs
- `research/spatial_analysis/` small-sample extraction/analysis script and manifest.
- A 14×14 token visualization and regional pooling illustration.
- A compact table of spatial descriptors / representation comparison.
- `docs/PHASE8_SPATIAL_ANALYSIS_REPORT.md`.

### Acceptance criteria
- The 14×14 grid is verified from model config and forward tensor shape.
- The script does not create a full-dataset 196×768 float32 cache.
- No slide-level or inter-patch topology claims.
- Feature norms and attention scores are not called tissue segmentation.

### Explicitly deferred
Whole-slide multiple-instance learning, CLAM/TransMIL, patient-level graphs, and cellular GNNs require slide IDs, WSI coordinates, cellular annotations or other data not currently available. CLAM is designed around bags of WSI instances and slide-level labels; those inputs cannot be fabricated from independent patch filenames.

---

## Phase 9 — Research error analysis and class-level findings

### Objective
Turn Phase 4.4 metrics and Phases 5–8 outputs into a small, auditable analysis of where the selected model performs well and where it fails.

### Recommended implementation
1. Load saved test predictions and labels from the final Phase 4.4 evaluation; do not rerun evaluation unless artifacts are missing or corrupted.
2. Produce the normalized confusion matrix and per-class precision, recall and F1.
3. List the most frequent off-diagonal confusions.
4. Select a fixed small gallery (e.g. up to 3 correct and 3 incorrect images per class) using deterministic rules. Include predicted class, true class, probability, sample ID and whether it is part of the explanation/robustness subset.
5. Summarize error patterns using neutral visual descriptions only. Without expert annotations, do not attribute mistakes to specific histological structures or biological mechanisms.
6. Identify low-confidence correct predictions and high-confidence errors as illustrative examples, not as a new model score.
7. Cross-reference Phase 5 attribution examples and Phase 6 perturbation-induced label flips where applicable.

### Required outputs
- `research/error_analysis/` sample selection CSV and plots.
- Compact confusion/error gallery.
- `docs/PHASE9_ERROR_ANALYSIS_REPORT.md`.

### Acceptance criteria
- Every shown example links to a real sample ID and recorded ground-truth/predicted label.
- Images are drawn from the documented held-out test split.
- No manually invented pathology explanations.
- The report discusses class imbalance only if observed; GCHTID's supplied classes are balanced at dataset level.

---

## Phase 10 — Integrate research outputs into the existing dashboard

### Objective
Expose the completed research results in the current FastAPI dashboard without replacing the working prediction workflow or creating a second application.

### Recommended implementation
Add a small, read-only “Research Insights” section to the existing UI. Reuse the current layout, API patterns, authentication/session behavior (if any), and static asset approach.

Minimum views:
1. **Model comparison:** Phase 4.4 metrics, with sample count and split caveats always visible.
2. **Explainability gallery:** selected sample image, overlay, target/predicted label, attribution-method caption, deletion-test summary.
3. **Robustness:** clean vs perturbation metric table and label-flip summary.
4. **Novelty screening:** distance score and research OOD flag with explicit note that it is not clinical confidence.
5. **Error analysis:** confusion matrix and representative error cards.

### Engineering boundaries
- Prefer static JSON/CSV generated by research scripts and served read-only; do not recompute model inference on every page request.
- Do not add schema migrations or mutate prediction history merely to show research results.
- Keep user-upload prediction behavior unchanged.
- Add graceful “research results not generated yet” states.
- Validate paths and avoid exposing arbitrary local file paths through API responses.
- Avoid displaying patient identifiers; no patient-linked dataset exists.
- Preserve app startup and lazy-loading behavior established in earlier phases.

### Required outputs
- One route/view or page section integrated into the existing dashboard.
- Minimal API tests for research result loading and missing-artifact behavior.
- `docs/PHASE10_DASHBOARD_INTEGRATION_REPORT.md`.

### Acceptance criteria
- Existing dashboard routes and prediction workflow still work.
- Research pages do not load heavyweight models at startup.
- Static research artifacts are served safely and predictably.
- Every chart shows its evaluation sample count and key limitation.

---

## Phase 11 — Final validation and reproducibility

### Objective
Run one focused final verification pass across the project. This is not another research phase and must not repeat already-passing audits without reason.

### Required checks
1. Run the Phase 5–10 tests added or changed by this work.
2. Run the existing project regression suite.
3. Start the FastAPI app and verify health/root and core prediction route with a small smoke test.
4. Verify reports and metrics agree with the machine-readable results.
5. Verify model/checkpoint hashes, manifest identities and cache shapes for artifacts used in final conclusions.
6. Confirm the original CNN-RNN checkpoint, DINOv2 cache, Phikon cache, source dataset and SQLite history were not unintentionally altered.
7. Check Git diff for accidental large files, local paths, secrets, API keys, temporary logs, duplicated data or generated full-size caches.
8. Re-run a documented minimal reproduction command from a clean process/environment.

### Required outputs
- `docs/PHASE11_FINAL_VALIDATION_REPORT.md`.
- Reproducibility commands in the README or `docs/REPRODUCIBILITY.md`.

### Acceptance criteria
- All relevant tests pass, or failures are clearly listed with root cause and impact.
- No unexplained modifications to protected artifacts.
- The application starts and core routes work.
- Evaluation results can be regenerated from saved configurations and existing artifacts.

---

## Phase 12 — Final research release and handover

### Objective
Package GastricPath AI as a stable, documented research prototype suitable for GitHub demonstration, academic discussion and technical interviews.

### Final report structure
`docs/GASTRICPATH_AI_FINAL_RESEARCH_REPORT.md`

1. Abstract and research question.
2. Dataset and class taxonomy.
3. Data integrity, duplicate analysis and split design.
4. Original CNN-RNN baseline and historical caveats.
5. DINOv2 and Phikon frozen-representation comparison.
6. Explainability method and faithfulness experiment.
7. Synthetic robustness analysis.
8. OOD/novelty screening experiment.
9. Intra-patch spatial analysis.
10. Error analysis.
11. Dashboard architecture and feature summary.
12. Limitations, ethics and reproducibility.
13. Conclusion and concise future work.

### Release checklist
- Update README with a clear project description, architecture diagram, setup instructions, screenshots and reproducibility commands.
- Add a concise research summary with the final measured numbers; no placeholder metrics.
- Include dataset provenance and licensing notes. Do not redistribute source images or pretrained weights unless their licences explicitly permit it.
- Include exact model identifiers, hashes, preprocessing details, seeds, split manifest references and software versions.
- Confirm all citations and references are real and accessible.
- Remove temporary files only after confirming they are not evidence or required checkpoints.
- Do not include secrets, user data, raw patient records or local machine paths.
- Create a final Git commit only after Phase 11 succeeds. Do not push remotely unless the owner explicitly asks.

### Acceptance criteria
- One final research report and one usable README.
- All charts trace to machine-readable metrics.
- No unsupported patient/slide/clinical claims.
- Working tree is clean after the final commit.
- Final response states commit hash, tests, major findings and limitations.

---

# 4. Research reading list for Antigravity

**Instruction:** Read the items below before implementing the related phase. Prefer the paper/model card/official documentation over blogs or copied summaries. Do not attempt to read every paper in full when the specified sections are sufficient. Record a short note of which design decision each source supports.

## Essential readings

### R1. Phikon model card — model architecture and intended use
- Link: https://huggingface.co/owkin/phikon
- Read: Model description, technical specifications, preprocessing, usage notes, training description and licence.
- Use in: Phases 5 and 8.
- Reason: Confirm the actual model is original `owkin/phikon`, not Phikon-v2; verify image size, architecture and feature outputs against the loaded checkpoint.

### R2. Abnar & Zuidema (2020), “Quantifying Attention Flow in Transformers”
- Link: https://aclanthology.org/2020.acl-main.385/
- DOI: https://doi.org/10.18653/v1/2020.acl-main.385
- Read: Attention rollout and attention flow methods, assumptions, and comparison to ablation/gradient importance.
- Use in: Phase 5.
- Reason: Background for transformer attention propagation. Treat the methods as approximations, not definitive explanations.

### R3. Chefer, Gur & Wolf (CVPR 2021), “Transformer Interpretability Beyond Attention Visualization”
- Link: https://openaccess.thecvf.com/content/CVPR2021/html/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.html
- Read: Method overview, relevance propagation through attention and residual connections, and evaluation.
- Use in: Phase 5.
- Reason: Explains why raw attention alone may be insufficient. Implement only if the actual Phikon architecture and installed dependencies support it within the time budget.

### R4. Wu et al. (CVPR 2024), “On the Faithfulness of Vision Transformer Explanations”
- Link: https://openaccess.thecvf.com/content/CVPR2024/html/Wu_On_the_Faithfulness_of_Vision_Transformer_Explanations_CVPR_2024_paper.html
- Read: Definition of faithfulness, proposed evaluation and limitations of existing metrics.
- Use in: Phase 5.
- Reason: Motivates testing attribution maps rather than selecting them based on appearance.

### R5. Selvaraju et al. (ICCV 2017), “Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization”
- Link: https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html
- Read: Method and architectural applicability.
- Use in: Phase 5, only for CNN feature maps such as the historical CNN-RNN if suitable.
- Reason: Grad-CAM is designed around convolutional feature maps; do not apply it to a ViT as if it were a CNN without a justified adaptation.

### R6. Macenko et al. (2009), “A Method for Normalizing Histology Slides for Quantitative Analysis”
- Link: https://www.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf
- Read: Optical-density representation, stain-vector estimation and normalization pipeline.
- Use in: Phase 6.
- Reason: Provides the basis for describing stain-related variation correctly. If only simple color jitter is implemented, do not call it Macenko normalization or a validated stain simulation.

### R7. Lee et al. (NeurIPS 2018), “A Unified Framework for Detecting Out-of-Distribution and Adversarial Samples”
- Link: https://research.google/pubs/a-unified-framework-for-detecting-out-of-distribution-and-adversarial-samples/
- Read: Class-conditional Gaussian feature modelling and Mahalanobis score.
- Use in: Phase 7.
- Reason: Provides the baseline method for distance-based novelty screening. Re-estimate mean/covariance from this project's training features only.

### R8. Jain & Wallace (NAACL 2019), “Attention is not Explanation”
- Link: https://aclanthology.org/N19-1357/
- DOI: https://doi.org/10.18653/v1/N19-1357
- Read: Abstract, core experimental critique and discussion.
- Use in: Phase 5.
- Reason: A caution against equating attention weights with faithful attribution. The paper is from NLP, so use it as a conceptual warning rather than direct pathology evidence.

### R9. Captum documentation — feature ablation and attribution metrics
- Feature ablation: https://captum.ai/api/feature_ablation.html
- Metrics: https://captum.ai/api/metrics.html
- Read: Grouped feature ablation, infidelity and sensitivity definitions.
- Use in: Phases 5 and 9, if using PyTorch attribution tooling.
- Reason: Provides documented implementation patterns and evaluation concepts. A small custom occlusion implementation is also acceptable if tested.

### R10. scikit-learn calibration documentation
- Link: https://scikit-learn.org/stable/modules/calibration.html
- Read: Reliability diagrams and the distinction between calibration and proper scoring rules.
- Use in: Phase 7 and final reporting.
- Reason: Avoid interpreting one Brier score as a pure calibration measure. Keep probability confidence distinct from feature-space novelty.

## Useful contextual reading

### R11. CLAM — data-efficient weakly supervised computational pathology on whole-slide images
- Link: https://www.nature.com/articles/s41551-020-00682-w
- Read: The required WSI/slide-level bag setting and attention-based MIL formulation.
- Use in: Scope decisions for Phase 8.
- Reason: Supports the decision not to simulate WSI bags from independent GCHTID patch files. Do not implement CLAM without slide membership and slide-level labels.

### R12. Stain-normalization robustness / evaluation context
- Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC6778842/
- Read: Stain variability and Macenko normalization overview.
- Use in: Phase 6.
- Reason: Background on stain-normalization methods and sources of variation.

---

# 5. Antigravity working instructions

Before implementation, read these repository documents and artifacts in this order (use the actual current filenames if one has been renamed):

1. `docs/IMPLEMENTATION_STATUS.md`.
2. `docs/PHASE4_4_FINAL_COMPARATIVE_RESEARCH_REPORT.md` or the actual final Phase 4.4 report that exists.
3. Phase 4.4 model card, metric JSON/CSV and final predictions.
4. Existing dataset manifest, split files and class mapping.
5. Existing FastAPI routes, dashboard templates/components and test conventions.
6. The relevant research readings in Section 4 before each phase.

### How to execute
- Complete Phases 5–12 sequentially. Do not run Phase 5–12 model-dependent tasks until Phase 4.4 evaluation is complete and verified.
- At the start of each phase, inspect only the files needed for that phase; do not repeat a full repository audit.
- Reuse the current environment and dependencies. Add a dependency only when essential and document why.
- Make the smallest implementation that meets the acceptance criteria.
- Run focused tests for changed functionality, not every test suite after every file edit.
- Write one concise phase report and update `docs/IMPLEMENTATION_STATUS.md` once at the end of each completed phase.
- Preserve all prior research outputs and raw evidence.
- If a method is blocked by missing metadata or unavailable model internals, document the blocker, use the specified fallback, and continue. Do not create new research scope to work around a limitation.
- Do not fabricate clinical metadata, pathology annotations, scanner identities, slide coordinates, patient links or external validation results.
- Do not commit after each phase. Use one final commit after Phase 11/12 validation, unless the repository's existing workflow requires otherwise.

### Resource budget and stop rules
- No full-dataset raw-token cache.
- No additional pretrained model downloads.
- No end-to-end fine-tuning.
- No whole-slide MIL or GNN on fabricated slide bags.
- No large-scale parameter sweeps.
- If a CPU task exceeds the predeclared sample/runtime budget, stop it safely, retain logs and use the fallback or smaller subset.
- If free disk is insufficient, do not start the operation.
- Never terminate Phase 4.4 or overwrite its outputs while it is running.

### Definition of done
The project is complete when:
1. Phases 5–9 have reproducible, appropriately caveated research outputs.
2. Phase 10 exposes those existing outputs in the working dashboard without breaking prediction functionality.
3. Phase 11 essential tests and integrity checks pass or are transparently documented.
4. Phase 12 contains the final report, README, reproducibility instructions and one final Git commit.

---

# 6. Final limitations statement to include in the report

GastricPath AI is a research prototype evaluating image-patch-level gastric histopathology classification on GCHTID. The available data and metadata do not establish patient- or slide-independent evaluation, and the clinical and biomarker tables are not linked to individual patches. The robustness experiments use synthetic image perturbations, and any OOD evaluation based on these perturbations is a proxy experiment rather than validation against a representative external tissue cohort. Explanation maps and spatial token analyses are model-derived visualizations, not expert-annotated tissue segmentation. The system has not been clinically validated and must not be presented as a diagnostic device.

---

# 7. Bibliography / source links

1. Owkin. *Phikon model card*. https://huggingface.co/owkin/phikon
2. Abnar, S. & Zuidema, W. (2020). *Quantifying Attention Flow in Transformers*. ACL. https://doi.org/10.18653/v1/2020.acl-main.385
3. Chefer, H., Gur, S. & Wolf, L. (2021). *Transformer Interpretability Beyond Attention Visualization*. CVPR. https://openaccess.thecvf.com/content/CVPR2021/html/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.html
4. Wu, J. et al. (2024). *On the Faithfulness of Vision Transformer Explanations*. CVPR. https://openaccess.thecvf.com/content/CVPR2024/html/Wu_On_the_Faithfulness_of_Vision_Transformer_Explanations_CVPR_2024_paper.html
5. Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*. ICCV. https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html
6. Macenko, M. et al. (2009). *A Method for Normalizing Histology Slides for Quantitative Analysis*. https://www.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf
7. Lee, K. et al. (2018). *A Unified Framework for Detecting Out-of-Distribution and Adversarial Samples*. NeurIPS. https://research.google/pubs/a-unified-framework-for-detecting-out-of-distribution-and-adversarial-samples/
8. Jain, S. & Wallace, B. C. (2019). *Attention is not Explanation*. NAACL. https://doi.org/10.18653/v1/N19-1357
9. Captum. *Feature Ablation and Attribution Metrics documentation*. https://captum.ai/api/feature_ablation.html and https://captum.ai/api/metrics.html
10. scikit-learn. *Probability Calibration*. https://scikit-learn.org/stable/modules/calibration.html
11. Lu, M. Y. et al. (2021). *Data-efficient and weakly supervised computational pathology on whole-slide images*. Nature Biomedical Engineering. https://www.nature.com/articles/s41551-020-00682-w
12. *A High-Performance System for Robust Stain Normalization of Whole-Slide Images in Histopathology*. https://pmc.ncbi.nlm.nih.gov/articles/PMC6778842/
