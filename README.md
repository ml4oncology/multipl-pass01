# MULTIPL Pipeline

Standardized training and validation framework for the **MULTIPL/PASS-01** multimodal machine learning architecture. This repository accompanies the manuscript and provides the core pipeline used to train and validate unimodal and multimodal estimators for predicting **Differential Treatment Effects (DTE)** and clinical endpoints in **Pancreatic Ductal Adenocarcinoma (PDAC)** cohorts.

---

## Overview

The framework supports training and validation across four data modalities:

| Modality | Description |
|---|---|
| Clinical | Patient demographics, treatment, and clinical metadata |
| Genomic (DNA) | DNA features |
| Transcriptomic (RNA) | RNAseq expression |
| Histopathology (WSI) | Whole-slide histopathology imaging features |

The repository includes:
- Unimodal model training
- Early-fusion multimodal training
- Late-fusion multimodal training
- Independent validation workflows

---

## Repository Structure

```text
multipl-pass01/
├── .gitignore
├── README.md
├── environment.yml
│
└── src/
    ├── training/
    │   ├── preprocess.py
    │   ├── train_unimodal.py
    │   ├── train_early_fusion.py
    │   └── train_late_fusion.py
    │
    └── validation/
        └── validate_pipelines.py
```

---

## Data Availability & Privacy

> [!IMPORTANT]
> Raw and processed genomic/clinical data from the COMPASS/PASS-01 trials are **not included** in this repository.

The underlying patient cohorts contain protected clinical and molecular data and cannot be publicly distributed.

This repository therefore provides:
- The full training/validation pipeline
- Model orchestration code
- Experimental framework and reproducibility utilities

but excludes:
- Patient-level datasets
- Processed feature matrices
- Derived clinical annotations

---

## Environment Setup

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate TabPFN
```

---

## Pipeline Execution Order

The framework is designed to be executed in the following order.

### 1. Model Training

Run the training scripts in `src/training/` to generate model checkpoints.

Example:

```bash
python src/training/unimodal.py
```

Additional training scripts:

```bash
python src/training/earlyfusion.py
python src/training/latefusion.py
```

---

### 2. Validation

After checkpoints have been generated, run the validation pipeline:

```bash
python src/validation/validate_pipelines.py
```

---

## Modality Dependency Rules

The following execution dependencies apply to both training and validation:

1. **Unimodal models must be trained first**
   - Required to establish baseline predictions

2. **Fusion models depend on unimodal outputs**
   - Early-fusion and late-fusion pipelines can be executed in any order after unimodal completion

---

## Methodological Notes

The repository is intended to support:
- Reproducible multimodal benchmarking
- Comparative fusion strategy evaluation
- Translational oncology machine learning research

The implementation focuses on standardized evaluation workflows for DTE modeling within the study framework.

---

## Citation

If you use this repository in academic work, please cite the accompanying manuscript.

```bibtex
@article{pass01_multipl,
  title   = {TODO},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO}
}
```

---

## License

Specify license information here.

```text
TODO: Add license
```
