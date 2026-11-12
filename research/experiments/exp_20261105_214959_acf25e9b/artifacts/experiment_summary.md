# Experiment Summary: exp_20261105_214959_acf25e9b

- **Model Architecture:** GastricCancerRNN
- **Parameter Count:** 869640
- **Input Representation:** sequence
- **Dataset:** GCHTID (GCHTID Group-Stratified)
- **Timestamp (UTC):** 2026-11-05T21:50:22.125607+00:00
- **Execution Runtime:** 21.87 seconds
- **Git Commit Hash:** `unknown`

## Overall Test Evaluation Metrics

| Metric | Value |
|:---|:---:|
| **Test Accuracy** | **0.715** |
| **Balanced Accuracy** | 0.715 |
| **Macro F1-Score** | **0.7115** |
| **Weighted F1-Score** | 0.7115 |
| **Macro Precision** | 0.7306 |
| **Macro Recall** | 0.715 |
| **Cohen's Kappa** | 0.6743 |
| **Macro AUROC (OvR)** | 0.9612 |
| **Expected Calibration Error (ECE)** | 0.0351 |
| **Brier Score** | 0.3806 |

## Scientific Integrity Limitation

> [!IMPORTANT]
> Evaluated on image-level group-stratified splits of the GCHTID dataset. True patient or slide identifiers
> are not encoded in image filenames. Results represent patch classification performance and must not be
> interpreted as patient-level clinical diagnostic accuracy or anatomical tumor area burden.
