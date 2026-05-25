#!/usr/bin/env python3
"""
Early-Fusion Runner (registry-based, explicit artifacts only)

Examples:
  python earlyfusion.py --targets orr --models lr --seeds 7270 --task prognosis \
    --modality-map ../../data/processed/COMPASS/modality_map.json \
    --registry-csv ../../data/processed/COMPASS/prognosis/split_registry.csv \
    --artifacts-paths-json ../../configs/artifacts_templates.json

  python earlyfusion.py --targets orr --models lr --seeds 7270 --task DTE-FFX --drop-treatment \
    --modality-map ../../data/processed/COMPASS/modality_map.json \
    --registry-csv ../../data/processed/COMPASS/DTE-FFX/split_registry.csv \
    --artifacts-paths-json ../../configs/artifacts_templates_DTE-FFX.json
"""
from __future__ import annotations

import warnings
import logging

logging.basicConfig(
    filename="warnings.log", filemode="w",
    level=logging.WARNING, format="%(levelname)s: %(message)s",
)

def custom_showwarning(message, category, filename, lineno, file=None, line=None):
    logging.warning(f"{category.__name__}: {message} (from {filename}:{lineno})")

warnings.showwarning = custom_showwarning

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

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

from unimodal import (
    get_param_grid as unimodal_get_param_grid,
    make_selector_and_pre,
    DNAPreprocessor,
    _dna_summary_selector,
    _dna_panel_selector,
    RadiomicsPreprocessor,
    RedundancyFilter,
)
from split_helpers import iter_outer_folds_from_registry
from config import RANDOM_STATES, DATE_STR, BASE_RESULTS, BASE_DATA


# =========================
# Utilities
# =========================

def get_param_grid(model_name: str) -> Dict[str, List[Any]]:
    try:
        base = unimodal_get_param_grid("RNA", model_name)
    except Exception:
        base = {}
    return {k: v for k, v in base.items() if k.startswith("clf__")}


def load_artifact_for_fold(artifact_path: Path, fold_index_1based: int) -> Dict[str, Any]:
    arts = joblib.load(artifact_path)
    if not isinstance(arts, (list, tuple)) or len(arts) < fold_index_1based:
        raise ValueError(f"Artifact {artifact_path} does not contain fold index {fold_index_1based}")
    return arts[fold_index_1based - 1]


class NameSliceAfterPre(BaseEstimator, TransformerMixin):
    def __init__(self, keep_names: List[str], expected_columns_order: Optional[List[str]] = None):
        self.keep_names = keep_names
        self.expected_columns_order = expected_columns_order

    def fit(self, X, y=None):
        self._keep_names = list(self.keep_names) if self.keep_names is not None else []
        self._expected_columns_order = (
            list(self.expected_columns_order) if self.expected_columns_order is not None else None
        )
        return self

    def transform(self, X):
        if hasattr(X, "columns"):
            X = X.copy()
            X.columns = [col.split("__")[-1] for col in X.columns]
            cols_present = [c for c in self._keep_names if c in X.columns]
            missing = [c for c in self._keep_names if c not in X.columns]
            if missing:
                print(f"[slice warn] {len(missing)} features missing after pre; e.g., {missing[:5]}")
            return X.loc[:, cols_present].to_numpy()

        if self._expected_columns_order is None:
            raise ValueError("Need expected_columns_order when pre returns numpy.")
        name_to_idx = {n: i for i, n in enumerate(self._expected_columns_order)}
        idxs = [name_to_idx.get(n) for n in self._keep_names if n in name_to_idx]
        missing = [n for n in self._keep_names if n not in name_to_idx]
        if missing:
            print(f"[slice warn] {len(missing)} features missing after pre; e.g., {missing[:5]}")
        return np.asarray(X)[:, np.asarray(idxs, dtype=int)]


def make_branch_for_non_histo(
    modality: str,
    seed: int,
    selected_feature_names: List[str],
    expected_columns_after_pre: Optional[List[str]] = None,
    drop_treatment: bool = False,
) -> Pipeline:
    """Structure: pre → slice → scl (no selector here)."""
    _sel, _grid, scl, pre = make_selector_and_pre(
        modality.lower(), seed, drop_treatment=drop_treatment
    )
    try:
        pre.set_output(transform="pandas")
    except Exception:
        pass

    slicer = NameSliceAfterPre(
        keep_names=selected_feature_names,
        expected_columns_order=expected_columns_after_pre,
    )
    return Pipeline(steps=[("pre", pre), ("slice", slicer), ("scl", scl)])


def make_branch_for_histo(
    modality: str,
    seed: int,
    n_components: int,
    norm_setting=None,
    whiten_setting=None,
    max_n_components: Optional[int] = None,
) -> Pipeline:
    """Structure: pre → sel(PCA set to n_components) → scl(passthrough)."""
    sel, _sel_grid, scl, pre = make_selector_and_pre(modality.lower(), seed)

    # Cap n_components to min(n_samples, n_features) for this fold
    if max_n_components is not None and n_components > max_n_components:
        print(
            f"[Histopathology] Capping PCA n_components from {n_components} "
            f"to {max_n_components} (min(n_samples, n_features) in this fold)."
        )
        n_components = max_n_components

    try:
        sel.set_params(**{"pca__n_components": int(n_components)})
    except Exception:
        pass
    if norm_setting is not None:
        try:
            sel.set_params(**{"norm": norm_setting})
        except Exception:
            pass
    if whiten_setting is not None:
        try:
            sel.set_params(**{"pca__whiten": bool(whiten_setting)})
        except Exception:
            pass

    try:
        pre.set_output(transform="pandas")
    except Exception:
        pass

    return Pipeline(steps=[("pre", pre), ("sel", sel), ("scl", scl)])


def extract_histopathology_pca_from_best_params(best_params: Dict[str, Any]) -> Optional[int]:
    if not best_params:
        return None
    for k, v in best_params.items():
        if k.endswith("sel__pca__n_components") or k == "sel__pca__n_components":
            try:
                return int(v)
            except Exception:
                return None
    return None


def build_fusion_pipeline_for_fold(
    X: pd.DataFrame,
    modality_map: Dict[str, List[str]],
    artifacts_by_modality: Dict[str, Dict[str, Any]],
    seed: int,
    model: str = "lr",
    drop_treatment: bool = False,
) -> Pipeline:
    """Full early-fusion pipeline (ColumnTransformer + classifier)."""
    transformers = []

    # Non-histopathology modalities
    for m in ["Clinical", "DNA", "RNA", "Radiomics"]:
        if m not in modality_map or m not in artifacts_by_modality:
            continue
        art = artifacts_by_modality[m]
        feats = art.get("features", [])
        if not feats:
            raise ValueError(f"{m}: empty 'features' list in unimodal artifact; verify unimodal run.")
        expected_cols = art.get("expected_columns", None)
        branch = make_branch_for_non_histo(
            m, seed, feats, expected_cols, drop_treatment=drop_treatment
        )
        transformers.append((m.lower(), branch, modality_map[m]))

    # Histopathology with PCA cap
    if "Histopathology" in modality_map and "Histopathology" in artifacts_by_modality:
        hart = artifacts_by_modality["Histopathology"]
        bp = hart.get("best_params", {}) or {}
        n_comp = extract_histopathology_pca_from_best_params(bp) or 128

        histo_cols = modality_map["Histopathology"]
        X_histo = X[histo_cols]
        max_pca_components = int(min(X_histo.shape[0], X_histo.shape[1]))

        norm_setting   = bp.get("sel__norm", None)
        whiten_setting = bp.get("sel__pca__whiten", None)
        branch_h = make_branch_for_histo(
            "Histopathology", seed, n_comp,
            norm_setting=norm_setting,
            whiten_setting=whiten_setting,
            max_n_components=max_pca_components,
        )
        transformers.append(("histopathology", branch_h, modality_map["Histopathology"]))

    ct = ColumnTransformer(
        transformers=transformers, remainder="drop", verbose_feature_names_out=False
    )

    m = model.lower()
    if m in {"lr", "logreg"}:
        clf = LogisticRegression(
            penalty="elasticnet", solver="saga", max_iter=50000, n_jobs=-1,
            l1_ratio=0.5, class_weight="balanced", random_state=seed,
        )
    elif m in {"xgb", "xgboost"}:
        if not _HAVE_XGB:
            raise ImportError("xgboost not installed.")
        clf = XGBClassifier(
            objective="binary:logistic", eval_metric="auc", tree_method="hist",
            n_jobs=-1, subsample=1.0, colsample_bytree=1.0, random_state=seed,
        )
    elif m == "tabpfn":
        if not _HAVE_TABPFN:
            raise ImportError("TabPFN is not installed.")
        clf = TabPFNClassifier(device="cpu")
    else:
        raise ValueError("model must be 'lr', 'xgb', or 'tabpfn' for early fusion")

    return Pipeline([("ct", ct), ("clf", clf)])


# =========================
# Main runner
# =========================

def run_earlyfusion_with_registry(
    X: pd.DataFrame,
    y: pd.Series,
    donor_ids: pd.Series,
    registry: pd.DataFrame,
    seed: int,
    target: str,
    model_name: str,
    modality_map: Dict[str, List[str]],
    tune: bool = False,
    n_splits_inner: int = 5,
    artifacts_paths: Dict[str, Path] = None,
    drop_treatment: bool = False,
) -> Tuple[List[float], pd.DataFrame, List[Dict[str, Any]]]:
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    if not artifacts_paths:
        raise ValueError("artifacts_paths is required.")

    X       = X.copy()
    y       = pd.Series(y).astype(int).values
    donors  = pd.Series(donor_ids).astype(str).values

    aucs: List[float] = []
    all_preds: List[pd.DataFrame] = []
    fold_models: List[Dict[str, Any]] = []

    try:
        fold_iter = iter_outer_folds_from_registry(
            X=X, y=y, donor_ids=donors, registry=registry, seed=seed, how="inner"
        )
    except TypeError:
        fold_iter = iter_outer_folds_from_registry(X, y, donors, registry, seed)

    for fold_number, tr_idx, te_idx in fold_iter:
        X_tr, X_te   = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te   = y[tr_idx], y[te_idx]
        donors_te    = donors[te_idx]

        needed  = {m for m in ["Clinical", "DNA", "RNA", "Radiomics", "Histopathology"] if m in modality_map}
        missing = [m for m in needed if m not in artifacts_paths]
        if missing:
            raise ValueError(f"Missing explicit artifacts for modalities: {missing}")

        artifacts_by_modality: Dict[str, Dict[str, Any]] = {}
        for mod in needed:
            p = Path(artifacts_paths[mod])
            if not p.exists():
                raise FileNotFoundError(f"Artifact path for modality '{mod}' does not exist: {p}")
            artifacts_by_modality[mod] = load_artifact_for_fold(p, fold_number)

        fusion = build_fusion_pipeline_for_fold(
            X=X_tr, modality_map=modality_map,
            artifacts_by_modality=artifacts_by_modality,
            seed=seed, model=model_name, drop_treatment=drop_treatment,
        )

        if tune:
            inner = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=seed)
            grid  = get_param_grid(model_name)
            if grid:
                gs = GridSearchCV(fusion, param_grid=grid, scoring="roc_auc",
                                  cv=inner, n_jobs=-1, refit=True)
                gs.fit(X_tr, y_tr)
                fusion = gs.best_estimator_
            else:
                fusion.fit(X_tr, y_tr)
        else:
            fusion.fit(X_tr, y_tr)

        y_prob = fusion.predict_proba(X_te)[:, 1]
        auc    = roc_auc_score(y_te, y_prob)
        aucs.append(auc)

        all_preds.append(pd.DataFrame({
            "fold_number":    fold_number,
            "donor":          donors_te,
            "true_label":     y_te,
            "predicted_prob": y_prob,
        }))

        # Introspection (best-effort)
        try:
            ct = fusion.named_steps["ct"]
            Xt = ct.transform(X_tr)
            print(f"[Fold {fold_number}] Fused train design: {Xt.shape}")
            for name, trans, cols in ct.transformers_:
                if name == "remainder":
                    continue
                part = trans.transform(X_tr[cols].iloc[:5])
                print(f"    - {name}: n_cols_in={len(cols)}, n_cols_out={part.shape[1]}")
        except Exception:
            pass

        fold_models.append({
            "fold":       fold_number,
            "model":      fusion,
            "target":     target,
            "model_name": model_name,
            "seed":       seed,
            "tuned":      tune,
        })

        print(f"[Fold {fold_number}] Early-fusion {model_name} AUC: {auc:.4f}")

    preds_df = pd.concat(all_preds, axis=0, ignore_index=True) if all_preds else pd.DataFrame()
    mean_auc = np.mean(aucs) if aucs else np.nan
    sd_auc   = np.std(aucs, ddof=1) if len(aucs) > 1 else 0.0
    print(f"\nOuter AUCs: {np.round(aucs, 3)} | Mean = {mean_auc:.3f} SD = {sd_auc:.3f}")
    return aucs, preds_df, fold_models


# =========================
# CLI
# =========================

def _csv_list(s: str) -> List[str]:
    if s is None:
        return []
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    return [x.strip() for x in s.split() if x.strip()]

def _csv_int_list(s: str) -> List[int]:
    return [int(x) for x in _csv_list(s)]


def main():
    p = argparse.ArgumentParser(description="Early-fusion runner (registry-based, explicit artifacts only)")
    p.add_argument("--targets",              type=str, required=True)
    p.add_argument("--models",               type=str, required=True)
    p.add_argument("--seeds",                type=str, default=",".join(str(s) for s in RANDOM_STATES))
    p.add_argument("--task",                 type=str, default="prognosis",
                   help="prognosis | DTE-FFX | DTE-GNP (default: prognosis)")
    p.add_argument("--drop-treatment",       action="store_true",
                   help="Drop Treatment/Liver_Met from clinical schema. Auto-set for DTE tasks.")
    p.add_argument("--data-dir",             type=str, default=None,
                   help="Directory containing EarlyFusion_{task}_{X,y,ids}.pkl. "
                        "Defaults to BASE_DATA/processed/COMPASS/{task}/")
    p.add_argument("--modality-map",         type=str, required=True)
    p.add_argument("--registry-csv",         type=str, default=None,
                   help="Split registry CSV. Defaults to BASE_DATA/processed/COMPASS/{task}/split_registry.csv")
    p.add_argument("--tune",                 action="store_true")
    p.add_argument("--n-splits-inner",       type=int, default=5)
    p.add_argument("--artifacts-paths-json", type=str, required=True)
    p.add_argument("--results-root",         type=str, default=None)
    args = p.parse_args()

    targets = _csv_list(args.targets)
    models  = _csv_list(args.models)
    seeds   = _csv_int_list(args.seeds)

    # DTE tasks always drop treatment columns
    drop_treatment = args.drop_treatment or args.task.startswith("DTE")

    # Resolve data directory
    data_dir = Path(args.data_dir) if args.data_dir else (
        BASE_DATA / "processed" / "COMPASS" / args.task
    )

    # Resolve registry
    registry_path = args.registry_csv or str(
        BASE_DATA / "processed" / "COMPASS" / args.task / "split_registry.csv"
    )
    registry = pd.read_csv(registry_path)
    print(f"Loaded registry from: {registry_path}  shape={registry.shape}")

    # Load EarlyFusion pickles — filename uses task name
    X_path   = data_dir / f"EarlyFusion_{args.task}_X.pkl"
    y_path   = data_dir / f"EarlyFusion_{args.task}_y.pkl"
    ids_path = data_dir / f"EarlyFusion_{args.task}_ids.pkl"

    X      = joblib.load(X_path)
    y_all  = joblib.load(y_path)
    ids    = joblib.load(ids_path)

    if "donor" not in ids.columns or ids.shape[1] != 1:
        raise ValueError(f"ids must have exactly one column named 'donor'; got {ids.columns.tolist()}")
    donor_ids = ids["donor"].reset_index(drop=True)

    X.index        = pd.RangeIndex(len(X))
    donor_ids.index = X.index

    with open(args.modality_map, "r") as fh:
        modality_map = json.load(fh)

    with open(args.artifacts_paths_json, "r") as f:
        artifacts_templates = json.load(f)
    if not isinstance(artifacts_templates, dict):
        raise ValueError("--artifacts-paths-json must map modality → path template string.")

    base_root = Path(args.results_root) if args.results_root else BASE_RESULTS

    for target in targets:
        if target not in y_all.columns:
            raise ValueError(f"Target '{target}' not found in y pickle columns: {list(y_all.columns)}")
        y = pd.Series(y_all[target]).astype(int)
        y.index = X.index

        # Task-namespaced output directory
        out_dir_root = base_root / args.task / "EarlyFusion" / target

        for model in models:
            if model in {"xgb", "xgboost"} and not _HAVE_XGB:
                print("[WARN] Skipping xgb — not installed.")
                continue

            out_dir = out_dir_root / model / DATE_STR
            out_dir.mkdir(parents=True, exist_ok=True)

            for seed in seeds:
                print(f"\n=== {args.task} | EarlyFusion | target={target} | model={model} | seed={seed} ===")

                artifacts_paths: Dict[str, Path] = {}
                for m, tmpl in artifacts_templates.items():
                    path_str = str(tmpl).format(seed=seed, target=target, model=model)
                    artifacts_paths[m] = Path(path_str)

                aucs, preds_df, fold_models = run_earlyfusion_with_registry(
                    X=X, y=y, donor_ids=donor_ids, registry=registry,
                    seed=seed, target=target, model_name=model,
                    modality_map=modality_map,
                    tune=args.tune,
                    n_splits_inner=args.n_splits_inner,
                    artifacts_paths=artifacts_paths,
                    drop_treatment=drop_treatment,
                )

                base = f"EarlyFusion_{seed}"
                pd.DataFrame({"fold": np.arange(1, len(aucs) + 1), "auc": aucs}
                             ).to_csv(out_dir / f"{base}_scores.csv", index=False)
                preds_df.to_csv(out_dir / f"{base}_predictions.csv", index=False)
                joblib.dump(fold_models, out_dir / f"{base}_models.joblib", compress=3)
                pd.Series({
                    "task":           args.task,
                    "fusion":         True,
                    "model_type":     model,
                    "random_state":   seed,
                    "drop_treatment": drop_treatment,
                    "n_folds_outer":  5,
                    "sklearn_version":sklearn.__version__,
                    "tuned":          args.tune,
                    "n_folds_inner":  args.n_splits_inner if args.tune else None,
                }).to_json(out_dir / f"{base}_manifest.json", indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
