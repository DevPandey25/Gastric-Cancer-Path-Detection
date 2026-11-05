# Experiment Summary: exp_20261105_221027_2463a2de

- **Model Architecture:** GastricCancerRNN
- **Parameter Count:** 869640
- **Input Representation:** sequence
- **Dataset:** GCHTID (GCHTID Group-Stratified)
- **Timestamp (UTC):** 2026-11-05T22:13:33.731281+00:00
- **Execution Runtime:** 185.17 seconds
- **Git Commit Hash:** `unknown`

## Overall Test Evaluation Metrics

| Metric | Value | 95% Confidence Interval |
|:---|:---:|:---:|
| **Test Accuracy** | **0.7104** | [0.6972, 0.7232] (Wilson) |
| **Balanced Accuracy** | 0.7104 | [0.698, 0.7225] (Bootstrap) |
| **Macro F1-Score** | **0.7114** | [0.6984, 0.7235] (Bootstrap) |
| **Weighted F1-Score** | 0.7114 | — |
| **Macro Precision** | 0.7228 | — |
| **Macro Recall** | 0.7104 | — |
| **Cohen's Kappa** | 0.669 | — |
| **Macro AUROC (OvR)** | 0.9593 | — |
| **Expected Calibration Error (ECE)** | 0.0167 | — |
| **Brier Score** | 0.388 | — |

## Scientific Integrity Limitation

> [!IMPORTANT]
> Evaluated on image-level group-stratified splits of the GCHTID dataset. True patient or slide identifiers
> are not encoded in image filenames. Results represent patch classification performance and must not be
> interpreted as patient-level clinical diagnostic accuracy or anatomical tumor area burden.
