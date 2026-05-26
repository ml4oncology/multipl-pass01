#!/usr/bin/env python3
"""
PASS-01 Multimodal Validation Pipeline
Consolidated evaluation framework for Unimodal, Early Fusion, and Late Fusion models.
"""

import sys
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score

# Make src/training importable so joblib can unpickle models that reference
# unimodal.py custom classes (DNAPreprocessor, _dna_summary_selector, etc.)
_TRAINING_DIR = Path(__file__).resolve().parent.parent / "training"
if str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))

# Silence unnecessary user warnings from unpickled pipelines if required
warnings.filterwarnings("ignore", category=UserWarning)

# =====================================================================
# 1. Global Configurations
# =====================================================================
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from datetime import date

BASE_RESULTS = Path("../../results")
PRED_DIR = BASE_RESULTS / "PASS-01"
EF_VAL_PATH = Path("../../data/processed/PASS-01/PASS01_EarlyFusion_Validation.csv")

SUBJECT_COL = "Subject"
DATASTR = date.today().strftime("%Y%m%d")

TASKS = ["DTE-FFX", "DTE-GNP"]
TARGETS = ["orr", "1yOS"]
MODALITIES = ["Clinical", "DNA", "Histopathology", "RNA"]
MODEL_TYPES = ["lr", "xgb", "tabpfn"]
STACKERS = ["lr", "xgb", "tabpfn", "avg"]


# =====================================================================
# 2. Custom Transformers / Classes (Required for unpickling)
# =====================================================================
class ColumnSubsetter(BaseEstimator, TransformerMixin):
    """Subset a DataFrame to a fixed list of columns.
    Matches the implementation in build_final_models.py for pickle compatibility.
    """
    def __init__(self, columns: Optional[List[str]]):
        self.columns = list(columns) if columns is not None else None

    def fit(self, X, y=None):
        if self.columns is not None:
            missing = [c for c in self.columns if c not in X.columns]
            if missing:
                raise ValueError(
                    f"[ColumnSubsetter] Missing columns: {missing[:8]}{'...' if len(missing)>8 else ''}"
                )
        return self

    def transform(self, X):
        if self.columns is None:
            return X
        return X[self.columns]


class ProbToLogitScaler(BaseEstimator, TransformerMixin):
    def __init__(self, eps: float = 1e-6):
        self.eps = float(eps)

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        Xc = np.clip(X, self.eps, 1.0 - self.eps)
        L = np.log(Xc / (1.0 - Xc))
        self.mu_ = np.nanmean(L, axis=0)
        self.sd_ = np.nanstd(L, axis=0, ddof=0)
        bad = ~np.isfinite(self.sd_) | (self.sd_ == 0)
        if np.any(bad):
            self.sd_[bad] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        nan_mask = np.isnan(X)
        Xc = np.clip(X, self.eps, 1.0 - self.eps)
        L = np.log(Xc / (1.0 - Xc))
        Z = (L - self.mu_) / self.sd_
        Z[nan_mask] = np.nan
        return Z


# Expose classes to the top-level namespace so joblib can find them during load
import __main__
__main__.ColumnSubsetter = ColumnSubsetter
__main__.ProbToLogitScaler = ProbToLogitScaler

# Register all unimodal custom classes/objects in __main__ so joblib can
# unpickle FINAL models whose steps were defined in unimodal.py
import unimodal as _unimodal_mod
for _nm in dir(_unimodal_mod):
    if not _nm.startswith('__'):
        setattr(__main__, _nm, getattr(_unimodal_mod, _nm))


# =====================================================================
# 3. Helper Evaluation Functions
# =====================================================================
def compute_and_log_auc(task: str, approach: str, target: str, mt: str, y_val: pd.Series, y_prob: np.ndarray, labeled_mask: pd.Series, st: Optional[str] = None) -> float:
    """Helper method to isolate AUC scoring safely across all models."""
    context_str = f"[{task} · {approach} · {target} · {mt}" + (f" · {st}]" if st else "]")
    if labeled_mask.any():
        try:
            auc = roc_auc_score(y_val[labeled_mask].astype(int), y_prob[labeled_mask.to_numpy()])
            print(f"{context_str}  AUC={auc:.3f}")
            return auc
        except Exception as e:
            logging.error(f"Failed scoring for {context_str}: {e}")
            return np.nan
    else:
        print(f"{context_str}  AUC=NA (No ground truth labels found)")
        return np.nan


def get_prediction_probabilities(model, X_df: pd.DataFrame) -> np.ndarray:
    """Safely extracts probabilistic output across multiple model types."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_df)[:, 1]
    elif hasattr(model, "decision_function"):
        y_dec = model.decision_function(X_df)
        return 1 / (1 + np.exp(-y_dec))
    else:
        raise AttributeError(f"Model {type(model).__name__} lacks predict_proba or decision_function methods.")


# =====================================================================
# 4. Pipeline Phase Execution
# =====================================================================

def run_unimodal_validation():
    print("\n" + "="*60 + "\nSTARTING UNIMODAL VALIDATION\n" + "="*60)
    
    def load_validation_set(modality: str) -> pd.DataFrame:
        path = Path(f"../../data/processed/PASS-01/PASS01_{modality}_Validation.csv")
        df = pd.read_csv(path)
        logging.info(f"Loaded validation set with shape {df.shape} from {path}")
        return df

    def split_validation_set(validation_set: pd.DataFrame, target: str):
        df = validation_set.copy()
        ids = df[[SUBJECT_COL]].reset_index(drop=True)
        X_val = df.drop(columns=[SUBJECT_COL, "dcr", "orr", "1yOS"], errors="ignore").reset_index(drop=True)
        if target in df.columns:
            y_val = df[target].astype("Int64").reset_index(drop=True)
            labeled_mask = y_val.notna()
        else:
            y_val = pd.Series(pd.array([pd.NA] * len(df), dtype="Int64"))
            labeled_mask = pd.Series([False] * len(df))
        return ids, X_val, y_val, labeled_mask

    def align_to_expected_columns(X: pd.DataFrame, expected_columns: list) -> pd.DataFrame:
        X_aligned = X.copy()
        for c in expected_columns:
            if c not in X_aligned.columns:
                X_aligned[c] = np.nan
        return X_aligned.reindex(columns=expected_columns)

    for TASK in TASKS:
        for modality in MODALITIES:
            for tar in TARGETS:
                for mt in MODEL_TYPES:
                    RESULTS_DIR = BASE_RESULTS / TASK / modality / tar / mt / DATASTR
                    final_tag = f"{modality}_{tar}_{mt}_FINAL"

                    model_path    = RESULTS_DIR / f"{final_tag}.pkl"
                    manifest_path = RESULTS_DIR / f"{final_tag}_manifest.json"

                    if not model_path.exists():
                        logging.warning(f"Missing Unimodal model: {model_path}")
                        continue
                    if not manifest_path.exists():
                        logging.warning(f"Missing Unimodal manifest: {manifest_path}")
                        continue

                    model = joblib.load(model_path)
                    with open(manifest_path) as _fh:
                        meta = json.load(_fh)
                    expected = meta.get("expected_columns")
                    if not expected:
                        raise ValueError(f"`expected_columns` missing in meta for {final_tag}")

                    validation_set = load_validation_set(modality)
                    ids_val, X_val, y_val, labeled_mask = split_validation_set(validation_set, tar)
                    X_ext = align_to_expected_columns(X_val, expected)
                    
                    y_prob = get_prediction_probabilities(model, X_ext)
                    compute_and_log_auc(TASK, modality, tar, mt, y_val, y_prob, labeled_mask)

                    df_preds = pd.DataFrame({
                        SUBJECT_COL: ids_val[SUBJECT_COL].values,
                        "y_true": y_val.values,
                        "y_pred": y_prob,
                    })
                    out_path = BASE_RESULTS / "PASS-01" / TASK / f"{modality}_{tar}_{mt}_ext_preds.csv"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    df_preds.to_csv(out_path, index=False)
                    print(f"Saved: {out_path}")


def run_early_fusion_validation():
    print("\n" + "="*60 + "\nSTARTING EARLY FUSION VALIDATION\n" + "="*60)
    
    if not EF_VAL_PATH.exists():
        logging.error(f"EarlyFusion validation file missing: {EF_VAL_PATH}. Skipping Phase.")
        return

    def split_ef_validation(df: pd.DataFrame, target: str, subject_col: str):
        df_ = df.copy()
        ids = df_[[subject_col]].reset_index(drop=True)
        y_num = pd.to_numeric(df_[target], errors="coerce")
        labeled_mask = y_num.notna().reset_index(drop=True)
        y = y_num.astype("Int64").reset_index(drop=True)
        drop_cols = {subject_col, "dcr", "orr", "1yOS"} & set(df_.columns)
        X = df_.drop(columns=list(drop_cols)).reset_index(drop=True)
        return ids, X, y, labeled_mask

    def columns_expected_by_ct(model) -> list:
        """Return input columns expected by the CT step, or [] if no CT step."""
        if "ct" not in model.named_steps:
            return []   # simple pipeline — caller uses all feature columns
        ct = model.named_steps["ct"]
        cols_in_order = []
        for name, transformer, colsel in ct.transformers_:
            if isinstance(colsel, list):
                cols_in_order.extend(colsel)
        return cols_in_order

    def align_to_ct_columns(df: pd.DataFrame, ct_cols: list[str]) -> pd.DataFrame:
        X = df.copy()
        for c in ct_cols:
            if c not in X.columns:
                X[c] = np.nan
        return X.reindex(columns=ct_cols)

    for TASK in TASKS:
        ef_val = pd.read_csv(EF_VAL_PATH)
        for tar in TARGETS:
            if tar not in ef_val.columns:
                logging.warning(f"[val] Target '{tar}' not found in EarlyFusion layout; skipping.")
                continue

            ids_val, X_raw, y_val, labeled_mask = split_ef_validation(
                ef_val, target=tar, subject_col=SUBJECT_COL
            )

            for mt in MODEL_TYPES:
                out_dir = BASE_RESULTS / TASK / "EarlyFusion" / tar / mt / DATASTR
                tag = f"EarlyFusion_{tar}_{mt}_FINAL"
                savetag = f"EarlyFusion_{tar}_{mt}"
                model_path = out_dir / f"{tag}.pkl"

                if not model_path.exists():
                    logging.warning(f"[fusion] Missing EarlyFusion model: {model_path}")
                    continue

                model = joblib.load(model_path)
                ct_cols = columns_expected_by_ct(model)
                if ct_cols:
                    X_val = align_to_ct_columns(X_raw, ct_cols)
                else:
                    # Simple pipeline (no CT step): pass all feature columns as-is
                    X_val = X_raw.copy()

                y_prob = get_prediction_probabilities(model, X_val)
                compute_and_log_auc(TASK, "EarlyFusion", tar, mt, y_val, y_prob, labeled_mask)

                out_path = BASE_RESULTS / "PASS-01" / TASK / f"{savetag}_ext_preds.csv"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({
                    SUBJECT_COL: ids_val[SUBJECT_COL].values,
                    "y_true": y_val,
                    "y_pred": y_prob,
                }).to_csv(out_path, index=False)
                print(f"Saved: {out_path}")


def run_late_fusion_validation():
    print("\n" + "="*60 + "\nSTARTING LATE FUSION VALIDATION\n" + "="*60)
    
    def load_unimodal_val_pred(modality: str, tar: str, mt: str, pred_dir: Path) -> Optional[pd.DataFrame]:
        fp = pred_dir / f"{modality}_{tar}_{mt}_ext_preds.csv"
        if not fp.exists():
            logging.warning(f"[latefusion] Missing dependent unimodal validation prediction: {fp}")
            return None
        df = pd.read_csv(fp)
        needed = {SUBJECT_COL, "y_true", "y_pred"}
        if not needed.issubset(df.columns):
            logging.warning(f"[latefusion] Structured column failure in {fp}; missing elements of {needed}")
            return None
        df = df[[SUBJECT_COL, "y_true", "y_pred"]].copy()
        df[SUBJECT_COL] = df[SUBJECT_COL].astype(str)
        df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
        df = df.rename(columns={"y_pred": f"p_{modality}"})
        return df

    def build_val_meta(modalities: List[str], tar: str, mt: str, pred_dir: Path) -> pd.DataFrame:
        dfs = []
        for m in modalities:
            d = load_unimodal_val_pred(m, tar, mt, pred_dir=pred_dir)
            if d is not None:
                dfs.append(d)
        if not dfs:
            raise FileNotFoundError(f"Missing all upstream unimodal validation maps for late fusion stack {tar}/{mt}.")

        meta = dfs[0]
        for d in dfs[1:]:
            meta = meta.merge(d.drop(columns=["y_true"]), on=SUBJECT_COL, how="outer")

        if "y_true" not in meta.columns:
            raise ValueError("Upstream truth tensor 'y_true' lost in downstream compilation merge.")
        meta = meta.rename(columns={"y_true": "y"})

        for m in modalities:
            col = f"p_{m}"
            if col in meta.columns:
                meta[f"has_{m}"] = ~meta[col].isna()

        has_cols = [c for c in meta.columns if c.startswith("has_")]
        have_any = meta[has_cols].any(axis=1) if has_cols else pd.Series(False, index=meta.index)
        meta = meta.loc[have_any].copy()
        meta["y"] = pd.to_numeric(meta["y"], errors="coerce").astype("Int64")
        return meta.sort_values(SUBJECT_COL).reset_index(drop=True)

    def align_X_to_expected(X: pd.DataFrame, expected_cols: List[str]) -> pd.DataFrame:
        Xa = X.copy()
        for c in expected_cols:
            if c not in Xa.columns:
                Xa[c] = (np.nan if not c.startswith("has_") else False)
        return Xa.reindex(columns=expected_cols)

    for TASK in TASKS:
        PRED_DIR_TASK = BASE_RESULTS / "PASS-01" / TASK

        for tar in TARGETS:
            for mt in MODEL_TYPES:
                try:
                    meta_val = build_val_meta(MODALITIES, tar, mt, pred_dir=PRED_DIR_TASK)
                except FileNotFoundError as e:
                    logging.warning(str(e))
                    continue
                
                subj = meta_val[SUBJECT_COL].values
                y_ser = meta_val["y"]
                labeled_mask = y_ser.notna()

                for st in STACKERS:
                    lf_dir = BASE_RESULTS / TASK / "Latefusion" / tar
                    tag_base = f"Latefusion_FINAL_{tar}_{mt}_{st}"
                    savetag_base = f"Latefusion_{tar}_{mt}_{st}"
                    model_path = lf_dir / f"{tag_base}.joblib"
                    manifest_json = lf_dir / f"{tag_base}_manifest.json"
                    manifest_jb = lf_dir / f"{tag_base}_manifest.joblib"

                    manifest = None
                    if manifest_json.exists():
                        with open(manifest_json, "r") as fh:
                            manifest = json.load(fh)
                    elif manifest_jb.exists():
                        manifest = joblib.load(manifest_jb)
                    else:
                        logging.warning(f"[latefusion] Missing manifest descriptor for {TASK}/{tar}/{mt}/{st}; skipping.")
                        continue

                    expected_cols = manifest.get("expected_columns", [])
                    flags_used = manifest.get("flags_used", True)

                    if not expected_cols:
                        expected_cols = [f"p_{m}" for m in MODALITIES if f"p_{m}" in meta_val.columns]

                    if flags_used:
                        for m in MODALITIES:
                            flag = f"has_{m}"
                            if flag not in meta_val.columns and f"p_{m}" in meta_val.columns:
                                meta_val[flag] = ~meta_val[f"p_{m}"].isna()

                    X_val = align_X_to_expected(meta_val.drop(columns=["y"]), expected_cols)

                    if st == "avg":
                        prob_cols = [c for c in expected_cols if c.startswith("p_")]
                        if not prob_cols:
                            logging.warning("[latefusion] Target probability series columns empty for ensemble average; skipping.")
                            continue
                        eps = 1e-6
                        probs = np.clip(X_val[prob_cols].values, eps, 1 - eps)
                        logits = np.log(probs / (1 - probs))
                        y_prob = 1 / (1 + np.exp(-np.nanmean(logits, axis=1)))
                    else:
                        if not model_path.exists():
                            logging.warning(f"[latefusion] Missing model checkpoint file: {model_path}")
                            continue
                        meta_model = joblib.load(model_path)
                        y_prob = get_prediction_probabilities(meta_model, X_val)

                    compute_and_log_auc(TASK, "LateFusion", tar, mt, y_ser, y_prob, labeled_mask, st=st)

                    out_path = BASE_RESULTS / "PASS-01" / TASK / f"{savetag_base}_ext_preds.csv"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame({
                        SUBJECT_COL: subj,
                        "y_true": y_ser,
                        "y_pred": y_prob,
                        "is_labeled": labeled_mask.to_numpy(),
                }).to_csv(out_path, index=False)
                print(f"Saved: {out_path}")


# =====================================================================
# 5. Main Execution Entry Point
# =====================================================================
if __name__ == "__main__":
    run_unimodal_validation()
    run_early_fusion_validation()
    run_late_fusion_validation()
    print("\n" + "="*60 + "\nALL PIPELINE VALIDATIONS COMPLETE\n" + "="*60)