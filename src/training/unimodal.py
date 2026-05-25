#!/usr/bin/env python3
"""
Unimodal Nested-CV Runner (registry-based)
CLI version for cluster runs.

Examples:
  python unimodal.py --modality RNA --targets orr --models lr --seeds 7270 --task prognosis
  python unimodal.py --modality Clinical --targets orr,1yOS --models lr,xgb --seeds 7270,860 --task prognosis
  python unimodal.py --modality Clinical --targets orr --models lr --seeds 7270 --task DTE-FFX --drop-treatment
  python unimodal.py --modality RNA --targets orr --models lr --seeds 7270 --task DTE-GNP --drop-treatment
"""
from __future__ import annotations

import warnings
import logging

logging.basicConfig(
    filename="warnings.log",
    filemode="w",
    level=logging.WARNING,
    format="%(levelname)s: %(message)s"
)

def custom_showwarning(message, category, filename, lineno, file=None, line=None):
    logging.warning(f"{category.__name__}: {message} (from {filename}:{lineno})")

warnings.showwarning = custom_showwarning

import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import re
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, Normalizer

try:
    from xgboost import XGBClassifier
    _HAVE_XGB = True
except Exception:
    _HAVE_XGB = False

try:
    from tabpfn import TabPFNClassifier
    _HAVE_TABPFN = True
except Exception:
    _HAVE_TABPFN = False

from split_helpers import iter_outer_folds_from_registry
from utils import load_data_tabpfn
from config import RANDOM_STATES, DATE_STR, BASE_RESULTS, BASE_DATA


# =======================
# Clinical column schema
# =======================
CLIN_CONT = ["Age", "Ca_19_9"]

# Base binary columns always included regardless of task
CLIN_BIN_BASE = [
    "Gender_Male",
    "Race_American Indian Or Alaska Native", "Race_Asian", "Race_Black Or African American", "Race_Unknown",
    "ECOG_1", "ECOG_2"
]

# Treatment-related columns included for prognosis, excluded for DTE
# (within a DTE arm, treatment is constant — adds no signal)
CLIN_BIN_TREATMENT = [
    "Treatment_GA", "Treatment_Other"
]

def get_clin_bin(drop_treatment: bool) -> List[str]:
    """Return the appropriate CLIN_BIN list based on task."""
    if drop_treatment:
        return CLIN_BIN_BASE
    return CLIN_BIN_TREATMENT + CLIN_BIN_BASE

def get_clin_all(drop_treatment: bool) -> List[str]:
    return CLIN_CONT + get_clin_bin(drop_treatment)


# =======================
# Utilities
# =======================

def align_to_expected_columns(X_new: pd.DataFrame, expected: List[str]) -> pd.DataFrame:
    return X_new.reindex(columns=expected, fill_value=np.nan)

class RedundancyFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.support_mask_: np.ndarray | None = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        std = X.std(axis=0, ddof=0)
        safe = std > 0
        Xs = X[:, safe]
        if Xs.shape[1] == 0:
            keep = np.zeros(X.shape[1], dtype=bool)
            if X.shape[1] > 0:
                keep[0] = True
            self.support_mask_ = keep
            return self
        corr = np.corrcoef(Xs, rowvar=False)
        n = corr.shape[0]
        keep_small = np.ones(n, dtype=bool)
        for i in range(n):
            if not keep_small[i]:
                continue
            for j in range(i):
                if keep_small[j] and abs(corr[i, j]) >= self.threshold:
                    keep_small[i] = False
                    break
        keep = np.zeros(X.shape[1], dtype=bool)
        keep[np.where(safe)[0][keep_small]] = True
        self.support_mask_ = keep
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_mask_]

    def get_support(self, indices: bool = False):
        if indices:
            return np.where(self.support_mask_)[0]
        return self.support_mask_

def make_clinical_preprocessor(drop_treatment: bool = False) -> ColumnTransformer:
    """Typed imputation + log1p for Ca_19_9. CLIN_BIN varies by task."""
    clin_bin = get_clin_bin(drop_treatment)

    idx_age = CLIN_CONT.index("Age")
    idx_ca  = CLIN_CONT.index("Ca_19_9")

    cont_imp = ColumnTransformer(
        transformers=[
            ("age", Pipeline([("imp", SimpleImputer(strategy="median"))]), [idx_age]),
            ("ca",  Pipeline([
                ("imp",   SimpleImputer(strategy="median")),
                ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ]), [idx_ca]),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    pre = ColumnTransformer(
        transformers=[
            ("cont", cont_imp, CLIN_CONT),
            ("bin",  SimpleImputer(strategy="most_frequent"), clin_bin),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return pre

RAD_BL_PAT = re.compile(r'^(primary|L1)_.+_bl$')

def split_radiomics_cols(cols):
    cols_bl = [c for c in cols if RAD_BL_PAT.match(c)]
    primary_cols = [c for c in cols_bl if c.startswith("primary_")]
    l1_cols      = [c for c in cols_bl if c.startswith("L1_")]
    return primary_cols, l1_cols, cols_bl

class RadiomicsPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, sentinel=0.0):
        self.sentinel = float(sentinel)
        self.primary_cols_ = None
        self.l1_cols_ = None
        self.cols_out_ = None
        self.median_primary_ = None
        self.median_l1_ = None

    def fit(self, X, y=None):
        df = X.copy()
        prim, l1, cols_bl = split_radiomics_cols(df.columns)
        self.primary_cols_, self.l1_cols_, self.cols_out_ = prim, l1, cols_bl
        self.median_primary_ = df[prim].median(axis=0, skipna=True) if prim else None
        self.median_l1_      = df[l1].median(axis=0,  skipna=True) if l1  else None
        return self

    def transform(self, X):
        df = X.copy()
        for c in self.cols_out_:
            if c not in df:
                df[c] = np.nan
        df = df[self.cols_out_]
        if self.primary_cols_:
            df[self.primary_cols_] = df[self.primary_cols_].fillna(self.median_primary_)
        if self.l1_cols_:
            l1_block = df[self.l1_cols_].copy()
            row_all_nan = l1_block.isna().all(axis=1).values
            l1_block = l1_block.fillna(self.median_l1_)
            if np.any(row_all_nan):
                l1_block.loc[row_all_nan, :] = self.sentinel
            df[self.l1_cols_] = l1_block
        return df.values.astype(float)

class DNAPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self,
                 burden_cols: Optional[List[str]] = None,
                 repair_cols: Optional[List[str]] = None,
                 signature_prefix: str = "csnnls_sig",
                 eps: float = 1e-6):
        self.burden_cols = burden_cols or ["snv_count", "del_count", "ins_count", "sv_count"]
        self.repair_cols = repair_cols or ["dsbr_score", "mmr_score"]
        self.signature_prefix = signature_prefix
        self.eps = float(eps)

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("DNAPreprocessor expects a pandas DataFrame.")
        self._all_cols_ = list(X.columns)
        self.sig_cols_    = [c for c in self._all_cols_ if c.startswith(self.signature_prefix)]
        self.burden_cols_ = [c for c in self.burden_cols if c in self._all_cols_]
        self.repair_cols_ = [c for c in self.repair_cols if c in self._all_cols_]
        self.output_columns_ = self.sig_cols_ + self.burden_cols_ + self.repair_cols_
        if "donor" in self.output_columns_:
            self.output_columns_.remove("donor")
        return self

    def transform(self, X):
        X = X.copy()
        if "donor" in X.columns:
            X = X.drop(columns=["donor"])
        for c in self.output_columns_:
            if c not in X.columns:
                X[c] = 0.0
        if self.sig_cols_:
            sig_df   = X[self.sig_cols_].astype(float)
            row_sums = sig_df.sum(axis=1)
            denom    = np.where(row_sums.values == 0.0, self.eps, row_sums.values)
            X[self.sig_cols_] = sig_df.div(denom, axis=0)
        if self.burden_cols_:
            X[self.burden_cols_] = np.log1p(X[self.burden_cols_].astype(float))
        if self.repair_cols_:
            X[self.repair_cols_] = X[self.repair_cols_].astype(float)
        return X[self.output_columns_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.output_columns_, dtype=object)

def _dna_summary_selector(X: pd.DataFrame) -> List[str]:
    burden = {"snv_count", "del_count", "ins_count", "sv_count"}
    repair = {"dsbr_score", "mmr_score"}
    sig    = [c for c in X.columns if c.startswith("csnnls_sig")]
    cols   = set(sig) | burden | repair
    return [c for c in X.columns if c in cols and c != "donor"]

def _dna_panel_selector(X: pd.DataFrame) -> List[str]:
    summary = set(_dna_summary_selector(X)) | {"donor"}
    return [c for c in X.columns if c not in summary]


# =======================
# Selector factory (per modality)
# =======================

def make_selector_and_pre(
    modality: str,
    seed: int,
    drop_treatment: bool = False,
) -> Tuple[object, Dict[str, List[Any]], object, object]:
    m   = modality.lower()
    pre = SimpleImputer(strategy="median")
    scl = StandardScaler(with_mean=True, with_std=True)

    if m == "clinical":
        sel  = "passthrough"
        grid = {}
        pre  = make_clinical_preprocessor(drop_treatment=drop_treatment)
        return sel, grid, scl, pre

    if m == "rna":
        sel  = Pipeline([
            ("vt", VarianceThreshold(threshold=0.0)),
            ("k",  SelectKBest(score_func=f_classif, k=25)),
        ])
        grid = {"sel__k__k": [10, 15, 20, 25, 30, 40, 50, 75]}
        return sel, grid, scl, pre

    if m == "dna":
        panel_branch = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("vt",  VarianceThreshold(threshold=0.0)),
            ("k",   SelectKBest(score_func=f_classif, k=25)),
        ])
        summary_branch = Pipeline([
            ("dna", DNAPreprocessor()),
            ("imp", SimpleImputer(strategy="median")),
        ])
        pre = ColumnTransformer(
            transformers=[
                ("summary", summary_branch, _dna_summary_selector),
                ("panel",   panel_branch,   _dna_panel_selector),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        sel  = "passthrough"
        grid = {"pre__panel__k__k": [10, 15, 20, 25, 30, 40, 50, 75]}
        scl  = StandardScaler(with_mean=True, with_std=True)
        return sel, grid, scl, pre

    if m == "radiomics":
        pre  = RadiomicsPreprocessor(sentinel=0.0)
        sel  = Pipeline([
            ("vt0", VarianceThreshold(threshold=0.0)),
            ("rf",  RedundancyFilter(threshold=0.95)),
            ("k",   SelectKBest(score_func=f_classif, k=25)),
        ])
        grid = {"sel__k__k": [10, 15, 20, 25, 30, 40, 50, "all"]}
        return sel, grid, StandardScaler(with_mean=True, with_std=True), pre

    if m == "histopathology":
        sel  = Pipeline([
            ("scl_pre", StandardScaler(with_mean=True, with_std=True)),
            ("norm",    "passthrough"),
            ("pca",     PCA(n_components=None, svd_solver="full", random_state=seed)),
        ])
        grid = {
            "sel__norm":           ["passthrough", Normalizer(norm="l2")],
            "sel__pca__whiten":    [False, True],
            "sel__pca__n_components": [32, 48, 64, 80, 0.90, 0.95],
        }
        return sel, grid, "passthrough", pre

    sel  = SelectKBest(score_func=f_classif, k="all")
    grid = {"sel__k": [25, 50, 75, "all"]}
    return sel, grid, scl, pre


# =======================
# Classifier grids
# =======================

def get_param_grid(modality: str, model_name: str, drop_treatment: bool = False) -> Dict[str, List[Any]]:
    _, sel_grid, _, _ = make_selector_and_pre(modality, seed=0, drop_treatment=drop_treatment)

    if model_name in {"logreg", "lr"}:
        grid = {
            "clf__C":        [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
            "clf__l1_ratio": [0, 0.1, 0.2, 0.3, 0.5],
        }
        grid.update(sel_grid)
        return grid

    if model_name in {"xgboost", "xgb"}:
        grid = {
            "clf__n_estimators":    [200, 400],
            "clf__max_depth":       [2, 3, 4],
            "clf__learning_rate":   [0.05, 0.1],
            "clf__subsample":       [0.8, 1.0],
            "clf__colsample_bytree":[0.7, 1.0],
            "clf__reg_lambda":      [1.0, 3.0],
        }
        grid.update(sel_grid)
        return grid

    if model_name == "tabpfn":
        return sel_grid

    return sel_grid


# =======================
# Feature name recovery
# =======================

def _recover_selected_features(best_estimator: Pipeline, expected_columns: List[str]) -> List[str]:
    try:
        pre = best_estimator.named_steps.get("pre")
        if isinstance(pre, ColumnTransformer) and "panel" in pre.named_transformers_:
            try:
                names = list(pre.get_feature_names_out())
                if names:
                    return names
            except Exception:
                pass
            try:
                panel_cols = None
                for name, transformer, cols in pre.transformers_:
                    if name == "panel":
                        panel_cols = list(cols)
                        panel_pipe = pre.named_transformers_["panel"]
                        break
                if panel_cols is None:
                    return expected_columns
                vt = panel_pipe.named_steps.get("vt", None)
                kb = panel_pipe.named_steps.get("k",  None)
                if vt is not None and kb is not None and hasattr(vt, "get_support") and hasattr(kb, "get_support"):
                    idx_after_vt = np.where(vt.get_support())[0]
                    idx_k        = idx_after_vt[kb.get_support()]
                    selected_panel = [panel_cols[i] for i in idx_k]
                else:
                    selected_panel = panel_cols
                summ = pre.named_transformers_["summary"]
                summary_names = getattr(summ, "output_columns_", [])
                return list(summary_names) + list(selected_panel)
            except Exception:
                pass
    except Exception:
        pass

    try:
        sel = best_estimator.named_steps.get("sel")
        if sel is None or sel == "passthrough":
            return expected_columns
        if isinstance(sel, Pipeline):
            if "vt" in sel.named_steps and "k" in sel.named_steps:
                vt = sel.named_steps["vt"]; k = sel.named_steps["k"]
                if hasattr(vt, "get_support") and hasattr(k, "get_support"):
                    mask_vt = vt.get_support(); idx_vt = np.where(mask_vt)[0]
                    mask_k  = k.get_support();  idx_k  = idx_vt[mask_k]
                    return [expected_columns[i] for i in idx_k]
            if "rf" in sel.named_steps and "k" in sel.named_steps:
                rf = sel.named_steps["rf"]; k = sel.named_steps["k"]
                if hasattr(rf, "get_support") and hasattr(k, "get_support"):
                    mask_rf = rf.get_support(); idx_rf = np.where(mask_rf)[0]
                    mask_k  = k.get_support();  idx_k  = idx_rf[mask_k]
                    return [expected_columns[i] for i in idx_k]
            if "pca" in sel.named_steps:
                return []
        if hasattr(sel, "get_support"):
            mask = sel.get_support()
            return [expected_columns[i] for i, flag in enumerate(mask) if flag]
    except Exception:
        pass
    return []


# =======================
# Main nested-CV entrypoint
# =======================

def run_unimodal_nested_cv_with_registry(
    X: pd.DataFrame,
    y: pd.Series,
    donor_ids: pd.Series,
    registry: pd.DataFrame,
    seed: int,
    modality: str,
    model_name: str = "lr",
    n_splits_inner: int = 5,
    use_class_weight: bool = True,
    drop_treatment: bool = False,
) -> Tuple[List[float], pd.DataFrame, List[Dict[str, Any]]]:
    X       = X.copy()
    y       = np.asarray(y).ravel().astype(int)
    donors  = pd.Series(donor_ids).astype(str).values
    expected_columns = list(X.columns)

    outer_aucs: List[float] = []
    fold_preds: List[pd.DataFrame] = []
    fold_models: List[Dict[str, Any]] = []

    if model_name in {"logreg", "lr"}:
        clf = LogisticRegression(
            penalty="elasticnet", solver="saga", max_iter=50000, tol=1e-3,
            n_jobs=-1, class_weight="balanced" if use_class_weight else None,
            random_state=seed,
        )
    elif model_name in {"xgboost", "xgb"}:
        if not _HAVE_XGB:
            raise ImportError("xgboost is not installed.")
        clf = XGBClassifier(
            objective="binary:logistic", eval_metric="auc", tree_method="hist",
            n_jobs=-1, subsample=1.0, colsample_bytree=1.0, random_state=seed,
        )
    else:
        if not _HAVE_TABPFN:
            raise ImportError("TabPFN is not installed.")
        clf = TabPFNClassifier(device="cpu")

    sel, _, scl_override, pre = make_selector_and_pre(modality, seed, drop_treatment=drop_treatment)
    grid = get_param_grid(modality, model_name, drop_treatment=drop_treatment)
    scl  = "passthrough" if scl_override == "passthrough" else StandardScaler(with_mean=True, with_std=True)

    pipe = Pipeline([
        ("pre", pre),
        ("sel", sel),
        ("scl", scl),
        ("clf", clf),
    ])

    for fold_number, tr_idx, te_idx in iter_outer_folds_from_registry(X, y, donors, registry, seed):
        X_tr, X_te   = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te   = y[tr_idx], y[te_idx]
        donors_te    = donors[te_idx]

        inner = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=seed)
        gs = GridSearchCV(
            estimator=pipe, param_grid=grid, scoring="roc_auc",
            cv=inner, n_jobs=-1, refit=True, verbose=0,
            return_train_score=False, error_score=np.nan,
        )
        gs.fit(X_tr, y_tr)
        best_estimator = gs.best_estimator_
        best_params    = gs.best_params_

        y_prob = best_estimator.predict_proba(X_te)[:, 1]
        auc    = roc_auc_score(y_te, y_prob)
        outer_aucs.append(auc)

        fold_preds.append(pd.DataFrame({
            "fold_number":    fold_number,
            "donor":          donors_te,
            "true_label":     y_te,
            "predicted_prob": y_prob,
        }))

        pre_fitted    = best_estimator.named_steps["pre"]
        exp_cols      = getattr(pre_fitted, "cols_out_", expected_columns)
        final_features = _recover_selected_features(best_estimator, exp_cols)
        print(f"[Fold {fold_number}] {modality} AUC = {auc:.3f} | selected={len(final_features) if final_features else 'N/A'} / {len(exp_cols)}")

        fold_models.append({
            "fold":             fold_number,
            "model":            best_estimator,
            "features":         final_features,
            "expected_columns": exp_cols,
            "model_name":       model_name,
            "best_params":      best_params,
            "modality":         modality,
        })

    fold_predictions_all = pd.concat(fold_preds, axis=0, ignore_index=True)
    mean_auc = float(np.mean(outer_aucs)) if outer_aucs else np.nan
    sd_auc   = float(np.std(outer_aucs, ddof=1)) if len(outer_aucs) > 1 else 0.0
    print(f"\nOuter AUCs: {np.round(outer_aucs, 3)} | Mean = {mean_auc:.3f} SD = {sd_auc:.3f}")

    return outer_aucs, fold_predictions_all, fold_models


# =======================
# CLI
# =======================

def _csv_list(s: str) -> List[str]:
    if s is None:
        return []
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    return [x.strip() for x in s.split() if x.strip()]

def _csv_int_list(s: str) -> List[int]:
    return [int(x) for x in _csv_list(s)]

def main():
    p = argparse.ArgumentParser(description="Unimodal nested-CV runner (registry-based)")
    p.add_argument("--modality",       type=str, default="RNA",
                   help="Clinical | DNA | Radiomics | Histopathology | RNA")
    p.add_argument("--targets",        type=str, required=True,
                   help="Target(s): orr, 1yOS (csv/space separated)")
    p.add_argument("--models",         type=str, required=True,
                   help="Model(s): lr, xgb, tabpfn (csv/space separated)")
    p.add_argument("--seeds",          type=str, default=",".join(str(s) for s in RANDOM_STATES))
    p.add_argument("--task",           type=str, default="prognosis",
                   help="prognosis | DTE-FFX | DTE-GNP (default: prognosis)")
    p.add_argument("--drop-treatment", action="store_true",
                   help="Drop Treatment columns from clinical schema. "
                        "Set automatically implied by DTE tasks.")
    p.add_argument("--registry-csv",   type=str, default=None,
                   help="Path to split registry CSV. Defaults to task-specific path under BASE_DATA.")
    p.add_argument("--n-splits-inner", type=int, default=5)
    p.add_argument("--no-class-weight",action="store_true")
    p.add_argument("--results-root",   type=str, default=None)
    p.add_argument("--drop-fu",        action="store_true",
                   help="If set and modality=Radiomics, drop '*_fu' columns")
    args = p.parse_args()

    modality = args.modality
    targets  = _csv_list(args.targets)
    models   = _csv_list(args.models)
    seeds    = _csv_int_list(args.seeds)
    use_cw   = not args.no_class_weight

    # DTE tasks always drop treatment columns from clinical schema
    drop_treatment = args.drop_treatment or args.task.startswith("DTE")

    # Resolve registry path
    if args.registry_csv is not None:
        registry_path = args.registry_csv
    else:
        registry_path = str(BASE_DATA / "processed" / "COMPASS" / args.task / "split_registry.csv")
    reg = pd.read_csv(registry_path)
    print(f"Loaded registry from: {registry_path}  shape={reg.shape}")

    for target in targets:
        print(f"\n=== Loading data: task={args.task} modality={modality} target={target} ===")
        X, y, ids = load_data_tabpfn(args.task, modality, target)

        base_root   = Path(args.results_root) if args.results_root else BASE_RESULTS
        RESULTS_DIR = base_root / args.task / modality / target  # task-namespaced results

        for model in models:
            if model in {"xgb", "xgboost"} and not _HAVE_XGB:
                print("[WARN] Skipping xgb — not installed.")
                continue
            if model == "tabpfn" and not _HAVE_TABPFN:
                print("[WARN] Skipping tabpfn — not installed.")
                continue

            out_dir = RESULTS_DIR / model / DATE_STR
            out_dir.mkdir(parents=True, exist_ok=True)

            for seed in seeds:
                print(f"\n=== {args.task} | {modality} | {target} | {model} | seed={seed} ===")
                outer_aucs, fold_predictions, fold_models = run_unimodal_nested_cv_with_registry(
                    X=X, y=y.squeeze(), donor_ids=ids.squeeze(),
                    registry=reg, seed=seed, modality=modality,
                    model_name=model, n_splits_inner=args.n_splits_inner,
                    use_class_weight=use_cw, drop_treatment=drop_treatment,
                )

                base = f"{modality}_{seed}"
                pd.DataFrame({"fold": np.arange(1, len(outer_aucs) + 1), "auc": outer_aucs}
                             ).to_csv(out_dir / f"{base}_scores.csv", index=False)
                fold_predictions.to_csv(out_dir / f"{base}_predictions.csv", index=False)
                joblib.dump(fold_models, out_dir / f"{base}_models.joblib", compress=3)
                pd.Series({
                    "task":           args.task,
                    "modality":       modality,
                    "target":         target,
                    "model_type":     model,
                    "random_state":   seed,
                    "drop_treatment": drop_treatment,
                    "n_folds_outer":  5,
                    "n_folds_inner":  args.n_splits_inner,
                    "sklearn_version":sklearn.__version__,
                }).to_json(out_dir / f"{base}_manifest.json", indent=2)

    print("\nDone.")

if __name__ == "__main__":
    main()
