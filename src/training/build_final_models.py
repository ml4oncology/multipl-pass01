#!/usr/bin/env python3
"""
Refit Script for Unimodal, Early Fusion, and Late Fusion Pipelines.
Aggregates hyperparameter choices and stable features across cross-validation folds/seeds,
performs a final fit on the full multi-modal dataset, and saves production-ready artifacts.
"""

from __future__ import annotations
import os
import re
import json
import joblib
import warnings
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Any, List, Optional, Tuple

# Sklearn imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin

# ==============================================================================
# 1. LOGGING & WARNINGS SETUP
# ==============================================================================
logging.basicConfig(
    filename="warnings.log",
    filemode="w",
    level=logging.WARNING,
    format="%(levelname)s: %(message)s"
)

def custom_showwarning(message, category, filename, lineno, file=None, line=None):
    logging.warning(f"{category.__name__}: {message} (from {filename}:{lineno})")

warnings.showwarning = custom_showwarning

# Try imports for optional dependencies
try:
    from xgboost import XGBClassifier
    _HAVE_XGB = True
except ImportError:
    _HAVE_XGB = False

try:
    from tabpfn import TabPFNClassifier
    _HAVE_TABPFN = True
except ImportError:
    _HAVE_TABPFN = False

# Project-specific internal imports
from unimodal import make_selector_and_pre
from utils import load_data_tabpfn
from config import RANDOM_STATES, BASE_RESULTS, DATE_STR

# Register unimodal and earlyfusion attributes in __main__ so joblib can unpickle
# fold models saved when those scripts ran as __main__ (classes, functions, and
# module-level objects all carry __module__ == "__main__" at pickle time).
import unimodal as _unimodal_mod
import earlyfusion as _ef_mod
import __main__ as _main_mod
for _nm in dir(_unimodal_mod):
    if not _nm.startswith('__'):
        setattr(_main_mod, _nm, getattr(_unimodal_mod, _nm))
for _nm in dir(_ef_mod):
    if not _nm.startswith('__'):
        setattr(_main_mod, _nm, getattr(_ef_mod, _nm))


class ColumnSubsetter(BaseEstimator, TransformerMixin):
    """Subset a DataFrame to a fixed list of columns (used in final pipelines)."""
    def __init__(self, columns):
        self.columns = list(columns) if columns is not None else None

    def fit(self, X, y=None):
        if self.columns is not None:
            missing = [c for c in self.columns if c not in X.columns]
            if missing:
                raise ValueError(
                    f"[ColumnSubsetter] Missing columns: {missing[:8]}"
                    f"{'...' if len(missing) > 8 else ''}"
                )
        return self

    def transform(self, X):
        if self.columns is None:
            return X
        return X[self.columns]

# ==============================================================================
# 2. SHARED REPRODUCIBILITY & DEGENERACY UTILITIES
# ==============================================================================
def _load_auc_map(scores_csv: Path, preds_csv: Path) -> Dict[int, float]:
    """Extracts performance maps from cross-validation output files."""
    if scores_csv.exists():
        df = pd.read_csv(scores_csv)
        if {"fold", "auc"}.issubset(df.columns):
            return {int(r["fold"]): float(r["auc"]) for _, r in df.iterrows()}
        if {"fold_number", "auc"}.issubset(df.columns):
            return {int(r["fold_number"]): float(r["auc"]) for _, r in df.iterrows()}
    if preds_csv.exists():
        dfp = pd.read_csv(preds_csv)
        if {"fold_number", "true_label", "predicted_prob"}.issubset(dfp.columns):
            return {int(f): roc_auc_score(g["true_label"], g["predicted_prob"])
                    for f, g in dfp.groupby("fold_number")}
    return {}

def _is_degenerate_predictions(pipe: Pipeline, X_all: pd.DataFrame, coef_eps: float = 1e-6, proba_var_eps: float = 1e-6) -> bool:
    """Checks if a trained pipeline outputs constant or zeroed coefficients."""
    if "clf" in pipe.named_steps:
        clf = pipe.named_steps["clf"]
        if hasattr(clf, "coef_") and clf.coef_ is not None:
            if np.all(np.abs(clf.coef_) < coef_eps):
                return True
    proba = pipe.predict_proba(X_all)[:, 1]
    return bool(np.var(proba) < proba_var_eps)

def _relax_lr_regularization_until_non_degenerate(pipe: Pipeline, X_all: pd.DataFrame, y_all: pd.Series, max_tries: int = 5, C_scale: float = 10.0) -> Pipeline:
    """Iteratively scales up C inverse regularization parameters if LR collapses."""
    if "clf" not in pipe.named_steps or not isinstance(pipe.named_steps["clf"], LogisticRegression):
        return pipe
    
    C_key = "clf__C" if "clf__C" in pipe.get_params() else next((k for k in pipe.get_params() if k.endswith("clf__C")), None)
    if not C_key or not _is_degenerate_predictions(pipe, X_all):
        return pipe

    current_C = pipe.get_params()[C_key]
    for _ in range(max_tries):
        new_C = current_C * C_scale
        print(f"[Refit Warning] LR degenerate at C={current_C}. Scaling up to C={new_C}...")
        pipe.set_params(**{C_key: new_C})
        pipe.fit(X_all, np.asarray(y_all).astype(int))
        if not _is_degenerate_predictions(pipe, X_all):
            return pipe
        current_C = new_C
    return pipe

def _build_clf(model_name: str, seed: int):
    """Unified classifier instantiation mapping."""
    m_name = model_name.lower()
    if m_name in {"logreg", "lr"}:
        # l1_ratio=0.5 is the default; overridden by set_params when best_params include it
        return LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5,
                                  max_iter=10000, n_jobs=-1, class_weight="balanced", random_state=seed)
    if m_name in {"xgboost", "xgb"}:
        if not _HAVE_XGB: raise ImportError("XGBoost is not installed.")
        return XGBClassifier(objective="binary:logistic", eval_metric="auc", tree_method="hist", n_jobs=-1, random_state=seed)
    if m_name in {"tabpfn"}:
        if not _HAVE_TABPFN: raise ImportError("TabPFN is not installed.")
        return TabPFNClassifier(device="cpu", ignore_pretraining_limits=True)
    raise ValueError(f"Unknown classifier string archetype: {model_name}")

# ==============================================================================
# 3. UNIMODAL REFIT LOGIC
# ==============================================================================
def choose_stable_features_across_seeds(seed_items: List[Dict[str, Any]], expected_columns: List[str], vote_level: str = "fold", thresh: float = 0.60) -> Optional[List[str]]:
    by_seed_lists = {it["seed"]: [art.get("features") for art in it["fold_models"] if art.get("features")] for it in seed_items}
    by_seed_lists = {k: v for k, v in by_seed_lists.items() if v}
    if not by_seed_lists: return None

    vote: Counter = Counter()
    denom = len(by_seed_lists) if vote_level == "seed" else sum(len(L) for L in by_seed_lists.values())
    
    for lists_ in by_seed_lists.values():
        if vote_level == "seed":
            vote.update(set().union(*map(set, lists_)))
        else:
            for L in lists_: vote.update(L)

    chosen = [f for f, c in vote.items() if (c / max(denom, 1)) >= thresh]
    return [c for c in expected_columns if c in set(chosen)] or None

def aggregate_best_params_across_seeds(seed_items: List[Dict[str, Any]], tie_break_by_auc: bool = True) -> Dict[str, Any]:
    per_param_vals = defaultdict(list)
    for it in seed_items:
        auc_map = it["auc_map"] or {}
        for art in it["fold_models"]:
            for k, v in dict(art.get("best_params", {})).items():
                per_param_vals[k].append((v, float(auc_map.get(int(art["fold"]), np.nan))))
    
    final_params = {}
    for k, rows in per_param_vals.items():
        counts = Counter([v for (v, _) in rows])
        top_count = counts.most_common(1)[0][1]
        tied = [v for v, c in counts.items() if c == top_count]
        if len(tied) == 1 or not tie_break_by_auc:
            final_params[k] = tied[0]
        else:
            best_v, best_mean = None, -np.inf
            for v in tied:
                vscores = [auc for (vv, auc) in rows if vv == v and not np.isnan(auc)]
                mean_s = np.mean(vscores) if vscores else -np.inf
                if mean_s > best_mean:
                    best_mean, best_v = mean_s, v
            final_params[k] = best_v if best_v is not None else tied[0]
    return final_params

def build_final_unimodal(X_all: pd.DataFrame, y_all: pd.Series, task: str, modality: str, target: str, model_type: str, date_str: str, drop_treatment: bool) -> Tuple[Pipeline, Dict[str, Any]]:
    results_dir = BASE_RESULTS / task / modality / target / model_type / date_str
    
    # Collect items — skip seeds whose artifact files don't exist yet
    items = []
    for seed in RANDOM_STATES:
        base = results_dir / f"{modality}_{seed}"
        models_f = base.with_name(f"{modality}_{seed}_models.joblib")
        if not models_f.exists():
            continue
        auc_map = _load_auc_map(base.with_name(f"{modality}_{seed}_scores.csv"), base.with_name(f"{modality}_{seed}_predictions.csv"))
        items.append({"seed": seed, "fold_models": joblib.load(models_f), "auc_map": auc_map})
    if not items:
        raise FileNotFoundError(f"No model artifacts found in {results_dir}")

    # Columns and Params
    all_lists = [tuple(art.get("expected_columns", [])) for it in items for art in it["fold_models"] if art.get("expected_columns")]
    expected_columns = list(Counter(all_lists).most_common(1)[0][0])
    final_params = aggregate_best_params_across_seeds(items, tie_break_by_auc=True)
    stable_features = choose_stable_features_across_seeds(items, expected_columns, vote_level="fold", thresh=0.1)

    # Reconstruct pipeline architectural steps
    sel, _, scl_override, pre = make_selector_and_pre(modality, RANDOM_STATES[0], drop_treatment=drop_treatment)
    scl = "passthrough" if scl_override == "passthrough" else StandardScaler(with_mean=True, with_std=True)
    clf = _build_clf(model_type, RANDOM_STATES[0])

    steps = []
    if stable_features is not None: steps += [("subset", ColumnSubsetter(stable_features))]
    steps += [("pre", pre)]
    if stable_features is None: steps += [("sel", sel)]
    steps += [("scl", scl), ("clf", clf)]
    
    pipe = Pipeline(steps)
    # Safely drop incompatible parameter assignments if schema drift discovered
    valid_p = set(pipe.get_params().keys())
    pipe.set_params(**{k: v for k, v in final_params.items() if k in valid_p})

    # Transform Namespace 
    X_aligned = X_all.copy()
    for c in expected_columns:
        if c not in X_aligned.columns: X_aligned[c] = np.nan
    X_aligned = X_aligned[expected_columns]

    pipe.fit(X_aligned, np.asarray(y_all).astype(int))
    pipe = _relax_lr_regularization_until_non_degenerate(pipe, X_aligned, y_all)

    meta = {"modality": modality, "target": target, "model_name": model_type, "final_params": final_params, "stable_features": stable_features, "expected_columns": expected_columns}

    # Save outputs — stringify any non-JSON-serializable param values (e.g. sklearn Normalizer)
    def _json_safe(v):
        try: json.dumps(v); return v
        except (TypeError, ValueError): return repr(v)

    meta_safe = {**meta, "final_params": {k: _json_safe(v) for k, v in final_params.items()}}
    out_tag = f"{modality}_{target}_{model_type}_FINAL"
    joblib.dump(pipe, results_dir / f"{out_tag}.pkl")
    with open(results_dir / f"{out_tag}_manifest.json", "w") as f: json.dump(meta_safe, f, indent=2)
    return pipe, meta

# ==============================================================================
# 4. EARLY FUSION REFIT LOGIC
# ==============================================================================
def build_final_earlyfusion(X_all: pd.DataFrame, y_all: pd.Series, task: str, target: str, model_type: str, date_str: str) -> Tuple[Pipeline, Dict[str, Any]]:
    results_dir = BASE_RESULTS / task / "EarlyFusion" / target / model_type / date_str
    results_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate best params from available fold model artifacts
    items = []
    for seed in RANDOM_STATES:
        models_f = results_dir / f"EarlyFusion_{seed}_models.joblib"
        if not models_f.exists():
            continue
        auc_map = _load_auc_map(
            results_dir / f"EarlyFusion_{seed}_scores.csv",
            results_dir / f"EarlyFusion_{seed}_predictions.csv",
        )
        items.append({"seed": seed, "fold_models": joblib.load(models_f), "auc_map": auc_map})

    clf = _build_clf(model_type, RANDOM_STATES[0])
    pipe = Pipeline([("scl", StandardScaler()), ("clf", clf)])

    if items:
        final_params = aggregate_best_params_across_seeds(items, tie_break_by_auc=True)
        valid_p = set(pipe.get_params().keys())
        pipe.set_params(**{k: v for k, v in final_params.items() if k in valid_p})

    pipe.fit(X_all, np.asarray(y_all).astype(int))
    pipe = _relax_lr_regularization_until_non_degenerate(pipe, X_all, y_all)

    meta = {"strategy": "EarlyFusion", "target": target, "model_type": model_type}
    out_tag = f"EarlyFusion_{target}_{model_type}_FINAL"
    joblib.dump(pipe, results_dir / f"{out_tag}.pkl")
    return pipe, meta

# ==============================================================================
# 5. LATE FUSION REFIT LOGIC
# ==============================================================================
def build_final_latefusion(
    task: str,
    target: str,
    model_type: str,
    stacker: str,
    modalities: List[str],
    date_str: str,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Refit late fusion stacker for a given (task, target, model_type, stacker).
    Saves to results/{task}/Latefusion/{target}/ as expected by validate_pipelines.py.
      avg        → manifest only  (no fitted model needed)
      lr/xgb/tabpfn → fit stacker on training OOF predictions + save model
    """
    # Path matches validate_pipelines.py expectation (note: lowercase 'fusion')
    lf_dir = BASE_RESULTS / task / "Latefusion" / target
    lf_dir.mkdir(parents=True, exist_ok=True)
    tag_base = f"Latefusion_FINAL_{target}_{model_type}_{stacker}"

    # ── Build meta-feature matrix from training OOF prediction CSVs ──────────
    pred_cols_found: List[str] = []
    donor_dfs: List[pd.DataFrame] = []
    y_series: Optional[pd.Series] = None

    for mod in modalities:
        mod_dir = BASE_RESULTS / task / mod / target / model_type / date_str
        seed_files = sorted(mod_dir.glob(f"{mod}_*_predictions.csv"))
        if not seed_files:
            continue

        # Logit-average across available seeds
        frames = []
        for fp in seed_files:
            tmp = pd.read_csv(fp, dtype={"donor": str})
            if not {"donor", "true_label", "predicted_prob"}.issubset(tmp.columns):
                continue
            frames.append(tmp[["donor", "true_label", "predicted_prob"]].rename(
                columns={"predicted_prob": f"p_{mod}_{fp.stem}"}
            ))
        if not frames:
            continue

        merged = frames[0].rename(columns={f"p_{mod}_{frames[0].columns[-1].split('_')[-1]}": f"_p0"})
        # simpler: just stack all prob columns and average
        base = pd.read_csv(seed_files[0], dtype={"donor": str})[["donor", "true_label"]]
        prob_frames = []
        for i, fp in enumerate(seed_files):
            tmp = pd.read_csv(fp, dtype={"donor": str})[["donor", "predicted_prob"]]
            prob_frames.append(tmp.rename(columns={"predicted_prob": f"_s{i}"}))
        base = base.copy()
        for pf in prob_frames:
            base = base.merge(pf, on="donor", how="left")
        scols = [c for c in base.columns if c.startswith("_s")]
        eps = 1e-6
        probs = base[scols].clip(eps, 1 - eps)
        logits = np.log(probs / (1 - probs))
        base[f"p_{mod}"] = 1 / (1 + np.exp(-logits.mean(axis=1, skipna=True)))
        base = base[["donor", "true_label", f"p_{mod}"]]

        if y_series is None:
            y_series = base.set_index("donor")["true_label"]
        donor_dfs.append(base[["donor", f"p_{mod}"]])
        pred_cols_found.append(f"p_{mod}")

    expected_columns = pred_cols_found
    manifest: Dict[str, Any] = {
        "ensemble_strategy": stacker,
        "target": target,
        "model_type": model_type,
        "stacker": stacker,
        "expected_columns": expected_columns,
        "flags_used": False,
    }

    manifest_path = lf_dir / f"{tag_base}_manifest.json"

    if stacker == "avg" or not donor_dfs or y_series is None:
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"[Refit] LateFusion {task}/{target}/{model_type}/{stacker} — avg manifest saved.")
        return manifest, manifest

    # ── Merge meta-features and fit stacker model ─────────────────────────────
    meta = donor_dfs[0].copy()
    for d in donor_dfs[1:]:
        meta = meta.merge(d, on="donor", how="outer")
    meta = meta.set_index("donor")
    meta["y"] = y_series.reindex(meta.index)

    valid = meta["y"].notna()
    X_meta = meta.loc[valid, pred_cols_found].fillna(0.5)
    y_meta = meta.loc[valid, "y"].astype(int)

    if len(y_meta) < 4 or len(np.unique(y_meta)) < 2:
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"[Refit] LateFusion {task}/{target}/{model_type}/{stacker} — insufficient data, manifest only.")
        return manifest, manifest

    clf = _build_clf(stacker, RANDOM_STATES[0])
    stacker_pipe = Pipeline([("scl", StandardScaler()), ("clf", clf)])
    stacker_pipe.fit(X_meta, y_meta)

    model_path = lf_dir / f"{tag_base}.joblib"
    joblib.dump(stacker_pipe, model_path)
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[Refit] LateFusion {task}/{target}/{model_type}/{stacker} — stacker saved.")
    return stacker_pipe, manifest

# ==============================================================================
# 6. CENTRAL COMMAND EXECUTION INTERFACE (MAIN)
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Refit all execution variants for publication release pipeline execution.")
    parser.add_argument("--date", type=str, default=DATE_STR, help="Date string tag directory.")
    parser.add_argument("--strategy", type=str, choices=["all", "unimodal", "early", "late"], default="all", help="Target architecture group selection filter.")
    args = parser.parse_args()

    tasks = ["DTE-FFX", "DTE-GNP"]
    modalities = ["Clinical", "DNA", "Histopathology", "RNA"]
    targets = ["orr", "1yOS"]
    models = ["lr", "xgb", "tabpfn"]

    for task in tasks:
        drop_treatment = task.startswith("DTE")
        
        # --- RUN UNIMODAL ---
        if args.strategy in ["all", "unimodal"]:
            print(f"\n>>> Running Unimodal Refits for Task: {task}")
            for mod in modalities:
                for tar in targets:
                    for mt in models:
                        try:
                            X_all, y_all, _ = load_data_tabpfn(task, mod, tar)
                            _, meta = build_final_unimodal(X_all, y_all, task, mod, tar, mt, args.date, drop_treatment)
                            print(f"[Success Unimodal] {mod}-{tar}-{mt} Shape Matrix: {X_all.shape}")
                        except Exception as e:
                            print(f"[Skipped/Failed Unimodal] {mod}-{tar}-{mt}: {e}")

        # --- RUN EARLY FUSION ---
        if args.strategy in ["all", "early"]:
            print(f"\n>>> Running Early Fusion Refits for Task: {task}")
            for tar in targets:
                for mt in models:
                    try:
                        X_all, y_all, _ = load_data_tabpfn(task, "EarlyFusion", tar)
                        build_final_earlyfusion(X_all, y_all, task, tar, mt, args.date)
                        print(f"[Success Early Fusion] {tar}-{mt}")
                    except Exception as e:
                        print(f"[Skipped/Failed Early Fusion] {tar}-{mt}: {e}")

        # --- RUN LATE FUSION ---
        if args.strategy in ["all", "late"]:
            stackers = ["avg", "lr", "xgb", "tabpfn"]
            print(f"\n>>> Running Late Fusion Refits for Task: {task}")
            for tar in targets:
                for mt in models:
                    for st in stackers:
                        try:
                            build_final_latefusion(task, tar, mt, st, list(modalities), args.date)
                            print(f"[Success Late Fusion] {tar}-{mt}-{st}")
                        except Exception as e:
                            print(f"[Skipped/Failed Late Fusion] {tar}-{mt}-{st}: {e}")

if __name__ == "__main__":
    main()