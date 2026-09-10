# MULTIPL Pipeline

Standardized training and validation framework for the **MULTIPL/PASS-01** multimodal machine learning architecture. This repository accompanies the manuscript and provides the core pipeline used to train and validate unimodal and multimodal estimators for predicting **Differential Treatment Effects (DTE)** and clinical endpoints in **Pancreatic Ductal Adenocarcinoma (PDAC)** cohorts.

---

## Overview

The framework supports training and validation across four data modalities:

| Modality | Description |
|---|---|
| Clinical | Patient demographics, treatment, and clinical metadata |
| Genomic (DNA) | Mutational and copy-number DNA features |
| Transcriptomic (RNA) | RNAseq expression features |
| Histopathology | Whole-slide image embeddings (GigaPath) |

Two cohorts are used:

| Cohort | Role | N |
|---|---|---|
| COMPASS | Training | 266 |
| PASS-01 | External validation | 160 |

Two DTE tasks, each with two prediction targets:

| Task | Targets |
|---|---|
| DTE-FFX | `orr`, `1yOS` |
| DTE-GNP | `orr`, `1yOS` |

---

## Repository Structure

```text
multipl-pass01/
├── environment.yml
├── scripts/
│   ├── generate_mock_data.py       # Generate synthetic data for testing
│   └── run_dte_pipeline_test.sh    # End-to-end test runner
├── src/
│   ├── training/
│   │   ├── config.py
│   │   ├── unimodal.py             # Step 1
│   │   ├── earlyfusion.py          # Step 2
│   │   ├── latefusion.py           # Step 3
│   │   ├── aggregate_predictions.py # Step 4
│   │   ├── build_final_preds.py    # Step 5
│   │   ├── auc_analysis.py         # Step 6
│   │   ├── model_selection.py      # Step 7
│   │   └── build_final_models.py   # Step 8
│   └── validation/
│       ├── validate_pipelines.py   # Step 9
│       └── build_val_preds.py      # Step 10
└── data/                           # Not included — see Data Availability
    └── processed/
        ├── COMPASS/
        └── PASS-01/
```

---

## Data Availability & Privacy

> [!IMPORTANT]
> Raw and processed genomic/clinical data from the COMPASS/PASS-01 trials are **not included** in this repository. The underlying patient cohorts contain protected clinical and molecular data and cannot be publicly distributed.

This repository provides the full pipeline code but excludes patient-level datasets, processed feature matrices, and derived clinical annotations.

**Testing with mock data:** A synthetic data generator is included for pipeline verification (see [Quick Start](#quick-start-with-mock-data)).

---

## Environment Setup

```bash
conda env create -f environment.yml
conda activate MULTIPL-TabPFN
```

All pipeline commands below assume the `MULTIPL-TabPFN` environment is active.

---

## Quick Start with Mock Data

To verify the full pipeline without real patient data, generate synthetic data first:

```bash
python scripts/generate_mock_data.py
```

This creates mock feature matrices and labels in `data/processed/COMPASS/` and `data/processed/PASS-01/` with the correct shapes and column names. All downstream steps can then be run as-is.

---

## Full Pipeline

> [!NOTE]
> Training scripts (Steps 1–8) must be run from the `src/training/` directory. Validation scripts (Steps 9–10) must be run from `src/validation/`. All file paths in the scripts are relative to those directories.

### Step 1 — Unimodal Training

Train unimodal models (LR, XGBoost, TabPFN) for each modality, task, and target.

```bash
cd src/training
python unimodal.py
```

### Step 2 — Early Fusion Training

Train concatenated early-fusion models. Requires Step 1 to be complete.

```bash
python earlyfusion.py
```

### Step 3 — Late Fusion Training

Train late-fusion stacking models (avg, LR, XGBoost, TabPFN stackers). Requires Step 1.

```bash
python latefusion.py
```

---

### Step 4 — Aggregate OOF Predictions

Aggregate out-of-fold predictions across seeds for each model.

```bash
python aggregate_predictions.py
# If training ran on a different day:
python aggregate_predictions.py --date 20260527
```

### Step 5 — Build Wide Prediction Table

Merge all model predictions into a single wide CSV per task/target for AUC analysis.

```bash
python build_final_preds.py
# If training ran on a different day:
python build_final_preds.py --date 20260527
```

### Step 6 — AUC + Confidence Interval Analysis (Training)

Compute AUC with 95% bootstrap CIs and pairwise DeLong tests across all models.
Run once per task/target combination:

```bash
python auc_analysis.py -f ../../results/DTE-FFX/COMPASS_DTE-FFX_orr_preds.csv
python auc_analysis.py -f ../../results/DTE-FFX/COMPASS_DTE-FFX_1yOS_preds.csv
python auc_analysis.py -f ../../results/DTE-GNP/COMPASS_DTE-GNP_orr_preds.csv
python auc_analysis.py -f ../../results/DTE-GNP/COMPASS_DTE-GNP_1yOS_preds.csv
```

Outputs per run:
- `*_auc_ci_summary.csv` — AUC + 95% CI for every predictor
- `*_pairwise_auc_tests_complete_case.csv` — DeLong pairwise significance tests

---

### Step 7 — Model Selection

Identify the best late-fusion, early-fusion, and unimodal models per task/target using the AUC CI summaries from Step 6.

```bash
python model_selection.py -f ../../results/DTE-FFX/COMPASS_DTE-FFX_orr_auc_ci_summary.csv
python model_selection.py -f ../../results/DTE-FFX/COMPASS_DTE-FFX_1yOS_auc_ci_summary.csv
python model_selection.py -f ../../results/DTE-GNP/COMPASS_DTE-GNP_orr_auc_ci_summary.csv
python model_selection.py -f ../../results/DTE-GNP/COMPASS_DTE-GNP_1yOS_auc_ci_summary.csv
```

### Step 8 — Build Final Models

Refit final models on the full training set using the best hyperparameters identified across cross-validation folds.

```bash
python build_final_models.py
```

Final model files are saved to `results/{task}/{Modality}/{target}/{model_type}/{date}/`.

---

### Step 9 — External Validation

Run all final models on the PASS-01 external validation cohort. Requires Steps 1–8 and PASS-01 data in `data/processed/PASS-01/`.

```bash
cd ../validation
python -u validate_pipelines.py
# If build_final_models.py ran on a different day:
python -u validate_pipelines.py --date 20260527
```

Per-model prediction CSVs are saved to `results/PASS-01/{task}/`.

### Step 10 — Build Validation Prediction Table

Assemble the per-model CSVs into a single wide prediction table per task/target (same format as Step 5, for use with `auc_analysis.py`).

```bash
python build_val_preds.py
```

### Step 11 — AUC + CI Analysis (Validation)

Compute AUC with 95% bootstrap CIs and DeLong tests on the external validation predictions.

```bash
cd ../training
python auc_analysis.py -f ../../results/PASS-01/DTE-FFX/PASS01_DTE-FFX_orr_preds.csv
python auc_analysis.py -f ../../results/PASS-01/DTE-FFX/PASS01_DTE-FFX_1yOS_preds.csv
python auc_analysis.py -f ../../results/PASS-01/DTE-GNP/PASS01_DTE-GNP_orr_preds.csv
python auc_analysis.py -f ../../results/PASS-01/DTE-GNP/PASS01_DTE-GNP_1yOS_preds.csv
```

Final AUC summaries are saved alongside the prediction CSVs in `results/PASS-01/{task}/`.

---

## Summary of Execution Order

| Step | Script | Run from |
|------|--------|----------|
| 1 | `unimodal.py` | `src/training/` |
| 2 | `earlyfusion.py` | `src/training/` |
| 3 | `latefusion.py` | `src/training/` |
| 4 | `aggregate_predictions.py` | `src/training/` |
| 5 | `build_final_preds.py` | `src/training/` |
| 6 | `auc_analysis.py` (×4) | `src/training/` |
| 7 | `model_selection.py` (×4) | `src/training/` |
| 8 | `build_final_models.py` | `src/training/` |
| 9 | `validate_pipelines.py` | `src/validation/` |
| 10 | `build_val_preds.py` | `src/validation/` |
| 11 | `auc_analysis.py` (×4) | `src/training/` |

---

## Key Results

Best-performing models on the COMPASS training cohort:

| Task | Target | Model | AUC |
|------|--------|-------|-----|
| DTE-FFX | ORR | latefusion_tabpfn_lr | 0.705 |
| DTE-FFX | 1yOS | latefusion_tabpfn_lr | 0.730 |
| DTE-GNP | ORR | latefusion_tabpfn_lr | 0.633 |
| DTE-GNP | 1yOS | latefusion_tabpfn_lr | 0.730 |

---

## Citation

If you use this repository in academic work, please cite the accompanying manuscript.

```bibtex
@article{pass01_multipl,
  title   = {Multimodal Machine Learning for Predicting Outcomes in the PASS-01 Trial of Systemic Therapy for Metastatic Pancreatic Cancer},
  author  = {Quan, Wei and Henault, David and Zhang, Amy and Jang, Gun Ho and Hasnain, Syeda Mariam and Bevacqua, Daniela and Deng, Yangqing and Flores-Figueroa, Eugenia and Ni, Kewei and Light, Nicholas and Wilson, Julie M. and Dodd, Anna and Tsang, Erica S. and King, Daniel A. and Habowski, Amber N. and Yu, Kenneth and Perez, Kimberly and Aguirre, Andrew J. and O'Reilly, Eileen M. and Wolpin, Brian M. and Pugh, Trevor J. and Tuveson, David A. and Jaffee, Elizabeth M. and Gallinger, Steven and O'Kane, Grainne and Notta, Faiyaz and Knox, Jennifer J. and Grant, Robert C.},
  journal = {medRxiv},
  year    = {2026},
  doi     = {10.64898/2026.08.24.26360900},
  url     = {https://www.medrxiv.org/content/10.64898/2026.08.24.26360900v1}
}
```

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.
