#!/usr/bin/env python3
"""
Generate mock COMPASS data for end-to-end pipeline testing.

Run from repo root (conda activate work first):
  python scripts/generate_mock_data.py

Generates:
  data/processed/COMPASS/{task}/{Modality}_{task}_{X,y,ids}.pkl
  data/processed/COMPASS/{task}/split_registry.csv
  data/processed/COMPASS/modality_map.json
  data/processed/COMPASS/COMPASS_PurIST_scores.csv
  configs/artifacts_templates_{task}.json      (for earlyfusion.py)
  configs/latefusion_paths_sets_{task}.json    (for latefusion.py)
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import date
from sklearn.model_selection import StratifiedGroupKFold

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).parent.parent
DATA_DIR    = REPO_ROOT / "data" / "processed" / "COMPASS"
CONFIGS_DIR = REPO_ROOT / "configs"
DATE_STR    = date.today().strftime("%Y%m%d")

# ── Cohort sizes ──────────────────────────────────────────────────────────────
N_PROGNOSIS = 80
N_DTE_FFX   = 40
N_DTE_GNP   = 40

# ── Seeds (must match config.py) ──────────────────────────────────────────────
RANDOM_STATES = [7270, 860, 5390, 5191, 5734, 6265, 466, 4426, 5578, 8322]

# ── Clinical columns (mirrors unimodal.py schema) ─────────────────────────────
CLIN_CONT          = ["Age", "Ca_19_9"]
CLIN_BIN_TREATMENT = ["Treatment_GA", "Treatment_Other"]
CLIN_BIN_BASE      = [
    "Gender_Male",
    "Race_American Indian Or Alaska Native",
    "Race_Asian",
    "Race_Black Or African American",
    "Race_Unknown",
    "ECOG_1",
    "ECOG_2",
]
CLIN_COLS = CLIN_CONT + CLIN_BIN_TREATMENT + CLIN_BIN_BASE

# ── DNA columns (mutational signatures + burden + repair + panel genes) ────────
DNA_SIG_COLS    = [f"csnnls_sig{i}" for i in [1, 2, 3, 5, 6, 8, 13, 17, 18, 20, 26]]
DNA_BURDEN_COLS = ["snv_count", "del_count", "ins_count", "sv_count"]
DNA_REPAIR_COLS = ["dsbr_score", "mmr_score"]
DNA_PANEL_GENES = [
    "KRAS", "TP53", "SMAD4", "CDKN2A", "ARID1A", "RNF43", "BRCA1", "BRCA2",
    "ATM", "PALB2", "MLH1", "MSH2", "MSH6", "PMS2", "ERBB2", "FGFR3",
    "PIK3CA", "PTEN", "RB1", "STK11", "GATA6", "KDM6A", "PTPRD", "TGFBR2",
    "ACVR1B", "BMPR1A", "MAP2K4", "MLL2", "MLL3", "ROBO1", "ROBO2",
    "SLIT2", "EPHB6", "RASA1", "RASA2", "DOT1L", "KDM5C", "KEAP1",
    "NFE2L2", "ZNRF3", "AXIN1", "APC", "CTNNB1", "FBXW7", "NOTCH1",
    "NOTCH2", "JAK1", "JAK2", "CREBBP", "EP300",
]
DNA_COLS = DNA_SIG_COLS + DNA_BURDEN_COLS + DNA_REPAIR_COLS + DNA_PANEL_GENES

# ── RNA (300-gene subset of 19k, sufficient for SelectKBest with k up to 75) ──
N_RNA    = 300
RNA_COLS = [f"ENSG{str(i).zfill(11)}" for i in range(1, N_RNA + 1)]

# ── Histopathology (GigaPath 768-dim embeddings) ──────────────────────────────
N_HISTO_DIMS = 768
HISTO_COLS   = [f"dim_{i}" for i in range(N_HISTO_DIMS)]


# ── Generators ────────────────────────────────────────────────────────────────

def make_donor_ids(prefix: str, n: int) -> list:
    return [f"{prefix}_{str(i).zfill(4)}" for i in range(1, n + 1)]


def make_y(n: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({
        "orr":  rng.integers(0, 2, size=n).astype(int),
        "1yOS": rng.integers(0, 2, size=n).astype(int),
    })


def make_ids(donors: list) -> pd.DataFrame:
    return pd.DataFrame({"donor": donors})


def make_clinical_X(n: int, rng: np.random.Generator) -> pd.DataFrame:
    data: dict = {
        "Age":     rng.uniform(40.0, 80.0, size=n),
        "Ca_19_9": rng.exponential(scale=500.0, size=n),
    }
    for col in CLIN_BIN_TREATMENT + CLIN_BIN_BASE:
        data[col] = rng.integers(0, 2, size=n).astype(float)
    return pd.DataFrame(data, columns=CLIN_COLS)


def make_dna_X(n: int, rng: np.random.Generator) -> pd.DataFrame:
    data: dict = {}
    raw  = rng.exponential(scale=0.1, size=(n, len(DNA_SIG_COLS)))
    sums = raw.sum(axis=1, keepdims=True) + 1e-9
    for i, col in enumerate(DNA_SIG_COLS):
        data[col] = (raw / sums)[:, i]
    data["snv_count"]  = rng.integers(0, 500, size=n).astype(float)
    data["del_count"]  = rng.integers(0, 50,  size=n).astype(float)
    data["ins_count"]  = rng.integers(0, 30,  size=n).astype(float)
    data["sv_count"]   = rng.integers(0, 20,  size=n).astype(float)
    data["dsbr_score"] = rng.uniform(0.0, 1.0, size=n)
    data["mmr_score"]  = rng.uniform(0.0, 1.0, size=n)
    for gene in DNA_PANEL_GENES:
        prob = 0.9 if gene == "KRAS" else 0.7 if gene == "TP53" else 0.2
        data[gene] = rng.binomial(1, prob, size=n).astype(float)
    return pd.DataFrame(data, columns=DNA_COLS)


def make_rna_X(n: int, rng: np.random.Generator) -> pd.DataFrame:
    vals = rng.normal(loc=4.0, scale=2.0, size=(n, N_RNA)).clip(0.0)
    return pd.DataFrame(vals, columns=RNA_COLS)


def make_histo_X(n: int, rng: np.random.Generator) -> pd.DataFrame:
    vals = rng.normal(loc=0.0, scale=0.1, size=(n, N_HISTO_DIMS))
    return pd.DataFrame(vals, columns=HISTO_COLS)


def make_split_registry(donors: list, y: np.ndarray, seeds, n_splits: int = 5) -> pd.DataFrame:
    donors_arr = np.array(donors, dtype=str)
    labels     = np.array(y, dtype=int)
    rows: list = []
    for seed in seeds:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
        for fold_idx, (_, te_idx) in enumerate(
            sgkf.split(np.zeros(len(labels)), labels, groups=donors_arr), start=1
        ):
            for d in np.unique(donors_arr[te_idx]):
                rows.append((int(seed), str(d), int(fold_idx)))
    return pd.DataFrame(rows, columns=["seed", "donor", "outer_fold"])


def save_pkl(task_dir: Path, tag: str,
             X: pd.DataFrame, y: pd.DataFrame, ids: pd.DataFrame) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(X,   task_dir / f"{tag}_X.pkl",   compress=3)
    joblib.dump(y,   task_dir / f"{tag}_y.pkl",   compress=3)
    joblib.dump(ids, task_dir / f"{tag}_ids.pkl", compress=3)
    print(f"  {tag:<38}  X={str(X.shape):<14}  y={str(y.shape)}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"DATE_STR : {DATE_STR}")
    print(f"Repo root: {REPO_ROOT}\n")

    rng = np.random.default_rng(42)

    task_config = {
        "prognosis": {"n": N_PROGNOSIS, "prefix": "COMP"},
        "DTE-FFX":   {"n": N_DTE_FFX,   "prefix": "COMP_FFX"},
        "DTE-GNP":   {"n": N_DTE_GNP,   "prefix": "COMP_GNP"},
    }

    for task, cfg in task_config.items():
        n      = cfg["n"]
        donors = make_donor_ids(cfg["prefix"], n)
        print(f"=== {task}  (n={n}) ===")
        task_dir = DATA_DIR / task

        y   = make_y(n, rng)
        ids = make_ids(donors)

        X_clin  = make_clinical_X(n, rng)
        X_dna   = make_dna_X(n, rng)
        X_rna   = make_rna_X(n, rng)
        X_histo = make_histo_X(n, rng)

        save_pkl(task_dir, f"Clinical_{task}",      X_clin,  y, ids)
        save_pkl(task_dir, f"DNA_{task}",            X_dna,   y, ids)
        save_pkl(task_dir, f"RNA_{task}",            X_rna,   y, ids)
        save_pkl(task_dir, f"Histopathology_{task}", X_histo, y, ids)

        # EarlyFusion: concatenation of all modality features (same row order)
        X_ef = pd.concat([X_clin, X_dna, X_rna, X_histo], axis=1)
        X_ef.index = pd.RangeIndex(len(X_ef))
        save_pkl(task_dir, f"EarlyFusion_{task}", X_ef, y, ids)

        # Split registry (stratify by orr label, group by donor)
        reg = make_split_registry(donors, y["orr"].values, RANDOM_STATES)
        reg_path = task_dir / "split_registry.csv"
        reg.to_csv(reg_path, index=False)
        print(f"  split_registry.csv                      shape={reg.shape}\n")

    # ── PurIST scores (prognosis only, needed by build_final_preds.py) ────────
    donors_prog = make_donor_ids("COMP", N_PROGNOSIS)
    purist_df   = pd.DataFrame({
        "donor":          donors_prog,
        "predicted_prob": np.random.default_rng(99).uniform(0.0, 1.0, N_PROGNOSIS),
    })
    purist_path = DATA_DIR / "COMPASS_PurIST_scores.csv"
    purist_df.to_csv(purist_path, index=False)
    print(f"Saved: COMPASS_PurIST_scores.csv")

    # ── modality_map.json (column slices for the EarlyFusion X matrix) ────────
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    modality_map = {
        "Clinical":       CLIN_COLS,
        "DNA":            DNA_COLS,
        "RNA":            RNA_COLS,
        "Histopathology": HISTO_COLS,
    }
    mm_path = DATA_DIR / "modality_map.json"
    with open(mm_path, "w") as f:
        json.dump(modality_map, f, indent=2)
    print(f"Saved: data/processed/COMPASS/modality_map.json")

    # ── Per-task config JSONs ─────────────────────────────────────────────────
    # Paths are relative to src/training/ (where the scripts run from).
    for task in ["prognosis", "DTE-FFX", "DTE-GNP"]:
        # artifacts_templates_{task}.json — unimodal model artifact paths for earlyfusion.py
        # Template vars consumed by earlyfusion.py: {seed}, {target}, {model}
        artifacts_templates = {
            mod: str(
                Path("../../results") / task / mod
                / "{target}" / "{model}" / DATE_STR
                / f"{mod}_{{seed}}_models.joblib"
            )
            for mod in ["Clinical", "DNA", "RNA", "Histopathology"]
        }
        tmpl_path = CONFIGS_DIR / f"artifacts_templates_{task}.json"
        with open(tmpl_path, "w") as f:
            json.dump(artifacts_templates, f, indent=2)
        print(f"Saved: configs/artifacts_templates_{task}.json")

        # latefusion_paths_sets_{task}.json — unimodal OOF prediction dirs for latefusion.py
        # Template var consumed by latefusion.py: ${target}
        paths_sets = {
            base_model: {
                mod: str(
                    Path("../../results") / task / mod
                    / "${target}" / base_model / DATE_STR
                )
                for mod in ["Clinical", "DNA", "RNA", "Histopathology"]
            }
            for base_model in ["lr", "xgb", "tabpfn"]
        }
        lf_path = CONFIGS_DIR / f"latefusion_paths_sets_{task}.json"
        with open(lf_path, "w") as f:
            json.dump(paths_sets, f, indent=2)
        print(f"Saved: configs/latefusion_paths_sets_{task}.json")

    # ── PASS-01 External Validation Data ─────────────────────────────────────
    # n=160 patients; same modality columns as COMPASS training data.
    # Each CSV has: Subject, orr, 1yOS + feature columns.
    # ~50% of patients have outcome labels (rest are NaN = unlabeled).
    N_PASS01   = 160
    PASS01_DIR = REPO_ROOT / "data" / "processed" / "PASS-01"
    PASS01_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n=== PASS-01 validation data (n={N_PASS01}) ===")
    rng_val = np.random.default_rng(123)
    subjects = [f"PASS01_{str(i).zfill(4)}" for i in range(1, N_PASS01 + 1)]

    X_val_clin  = make_clinical_X(N_PASS01, rng_val)
    X_val_dna   = make_dna_X(N_PASS01, rng_val)
    X_val_rna   = make_rna_X(N_PASS01, rng_val)
    X_val_histo = make_histo_X(N_PASS01, rng_val)

    # ~50% labeled outcomes; rest NaN (simulates partial follow-up)
    n_labeled = N_PASS01 // 2
    y_orr  = np.where(np.arange(N_PASS01) < n_labeled,
                      rng_val.integers(0, 2, N_PASS01).astype(float), np.nan)
    y_1yos = np.where(np.arange(N_PASS01) < n_labeled,
                      rng_val.integers(0, 2, N_PASS01).astype(float), np.nan)

    for modality, X_val in [
        ("Clinical",       X_val_clin),
        ("DNA",            X_val_dna),
        ("RNA",            X_val_rna),
        ("Histopathology", X_val_histo),
    ]:
        df = pd.DataFrame({"Subject": subjects, "orr": y_orr, "1yOS": y_1yos})
        df = pd.concat([df, X_val.reset_index(drop=True)], axis=1)
        out = PASS01_DIR / f"PASS01_{modality}_Validation.csv"
        df.to_csv(out, index=False)
        print(f"  PASS01_{modality}_Validation.csv      shape={df.shape}")

    # EarlyFusion validation: concatenation of all modality features
    X_val_ef = pd.concat(
        [X_val_clin, X_val_dna, X_val_rna, X_val_histo], axis=1
    ).reset_index(drop=True)
    df_ef = pd.DataFrame({"Subject": subjects, "orr": y_orr, "1yOS": y_1yos})
    df_ef = pd.concat([df_ef, X_val_ef], axis=1)
    out_ef = PASS01_DIR / "PASS01_EarlyFusion_Validation.csv"
    df_ef.to_csv(out_ef, index=False)
    print(f"  PASS01_EarlyFusion_Validation.csv  shape={df_ef.shape}")

    # ── Usage hint ────────────────────────────────────────────────────────────
    print(f"\nAll mock data written. DATE_STR={DATE_STR}")
    print("\nRun training from src/training/ with conda activate work:")
    print(f"  # 1. Unimodal (example — 1 seed, 1 modality, 1 target):")
    print(f"  python unimodal.py --modality Clinical --targets orr --models lr \\")
    print(f"    --seeds 7270 --task prognosis")
    print(f"")
    print(f"  # 2. Early Fusion (after unimodal for all seeds):")
    print(f"  python earlyfusion.py --targets orr --models lr --seeds 7270 \\")
    print(f"    --task prognosis \\")
    print(f"    --modality-map ../../data/processed/COMPASS/modality_map.json \\")
    print(f"    --registry-csv ../../data/processed/COMPASS/prognosis/split_registry.csv \\")
    print(f"    --artifacts-paths-json ../../configs/artifacts_templates_prognosis.json")
    print(f"")
    print(f"  # 3. Late Fusion (after unimodal for all seeds):")
    print(f"  python latefusion.py --targets orr --base-set lr --stacker lr \\")
    print(f"    --seeds 7270 --task prognosis \\")
    print(f"    --paths-json ../../configs/latefusion_paths_sets_prognosis.json")


if __name__ == "__main__":
    main()
