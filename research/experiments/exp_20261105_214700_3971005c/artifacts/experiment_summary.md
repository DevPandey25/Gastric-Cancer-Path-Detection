# Experiment Summary: exp_20261105_214700_3971005c

- **Model Architecture:** ResNet50
- **Parameter Count:** 23524424
- **Input Representation:** 2d
- **Dataset:** GCHTID (Smoke Test Subset)
- **Timestamp (UTC):** 2026-11-05T21:47:39.695565+00:00
- **Execution Runtime:** 32.66 seconds
- **Git Commit Hash:** `unknown`

## Overall Test Evaluation Metrics

| Metric | Value |
|:---|:---:|
| **Test Accuracy** | **0.125** |
| **Balanced Accuracy** | 0.125 |
| **Macro F1-Score** | **0.0278** |
| **Weighted F1-Score** | 0.0278 |
| **Macro Precision** | 0.0156 |
| **Macro Recall** | 0.125 |
| **Cohen's Kappa** | 0.0 |
| **Macro AUROC (OvR)** | 0.4464 |
| **Expected Calibration Error (ECE)** | 0.0263 |
| **Brier Score** | 0.8806 |

## Scientific Integrity Limitation

> [!IMPORTANT]
> Evaluated on image-level group-stratified splits of the GCHTID dataset. True patient or slide identifiers
> are not encoded in image filenames. Results represent patch classification performance and must not be
> interpreted as patient-level clinical diagnostic accuracy or anatomical tumor area burden.
