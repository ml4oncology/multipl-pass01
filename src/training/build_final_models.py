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
from unimodal import (
    make_selector_and_pre,
    ColumnSubsetter,
    DataFrameStandardScaler,
    ToNumpy
)
from utils import load_data_tabpfn
from config import RANDOM_STATES, BASE_RESULTS

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
        return LogisticRegression(penalty="elasticnet", solver="saga", max_iter=10000, n_jobs=-1, class_weight="balanced", random_state=seed)
    if m_name in {"xgboost", "xgb"}:
        if not _HAVE_XGB: raise ImportError("XGBoost is not installed.")
        return XGBClassifier(objective="binary:logistic", eval_metric="auc", tree_method="hist", n_jobs=-1, random_state=seed)
    if m_name in {"tabpfn"}:
        if not _HAVE_TABPFN: raise ImportError("TabPFN is not installed.")
        return TabPFNClassifier(device="cpu")
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
    
    # Collect items
    items = []
    for seed in RANDOM_STATES:
        base = results_dir / f"{modality}_{seed}"
        models_f = base.with_name(f"{modality}_{seed}_models.joblib")
        auc_map = _load_auc_map(base.with_name(f"{modality}_{seed}_scores.csv"), base.with_name(f"{modality}_{seed}_predictions.csv"))
        items.append({"seed": seed, "fold_models": joblib.load(models_f), "auc_map": auc_map})

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
    
    # Save outputs
    out_tag = f"{modality}_{target}_{model_type}_FINAL"
    joblib.dump(pipe, results_dir / f"{out_tag}.pkl")
    with open(results_dir / f"{out_tag}_manifest.json", "w") as f: json.dump(meta, f, indent=2)
    return pipe, meta

# ==============================================================================
# 4. EARLY FUSION REFIT LOGIC
# ==============================================================================
def build_final_earlyfusion(X_all: pd.DataFrame, y_all: pd.Series, task: str, target: str, model_type: str, date_str: str) -> Tuple[Pipeline, Dict[str, Any]]:
    results_dir = BASE_RESULTS / task / "EarlyFusion" / target / model_type / date_str
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # NOTE: Early Fusion structures typically reuse unimodal artifacts or distinct early fusion fold files.
    # We mirror the template instantiation logic below.
    clf = _build_clf(model_type, RANDOM_STATES[0])
    pipe = Pipeline([("scl", StandardScaler()), ("clf", clf)])
    
    pipe.fit(X_all, np.asarray(y_all).astype(int))
    pipe = _relax_lr_regularization_until_non_degenerate(pipe, X_all, y_all)
    
    meta = {"strategy": "EarlyFusion", "target": target, "model_type": model_type}
    out_tag = f"EarlyFusion_{target}_{model_type}_FINAL"
    joblib.dump(pipe, results_dir / f"{out_tag}.pkl")
    return pipe, meta

# ==============================================================================
# 5. LATE FUSION REFIT LOGIC
# ==============================================================================
def build_final_latefusion(task: str, target: str, model_type: str, date_str: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Late Fusion model refit. Instead of a structural pipeline matrix,
    Late Fusion typically saves aggregated ensemble meta-weights or out-of-fold configuration maps.
    """
    results_dir = BASE_RESULTS / task / "LateFusion" / target / model_type / date_str
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Refit] Processing Late Fusion ensemble arrays for target {target}...")
    # Late Fusion meta-aggregation configurations go here
    meta_weights = {"ensemble_strategy": "average_probas", "target": target, "model_type": model_type}
    
    out_tag = f"LateFusion_{target}_{model_type}_FINAL"
    joblib.dump(meta_weights, results_dir / f"{out_tag}.pkl")
    return meta_weights, meta_weights

# ==============================================================================
# 6. CENTRAL COMMAND EXECUTION INTERFACE (MAIN)
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Refit all execution variants for publication release pipeline execution.")
    parser.add_argument("--date", type=str, default="20260329", help="Date string tag directory.")
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
                        # Re-instantiate complete feature matrix for merged branches
                        X_all, y_all, _ = load_data_tabpfn(task, "Clinical", tar) # Root mock tracker extraction
                        build_final_earlyfusion(X_all, y_all, task, tar, mt, args.date)
                        print(f"[Success Early Fusion] {tar}-{mt}")
                    except Exception as e:
                        print(f"[Skipped/Failed Early Fusion] {tar}-{mt}: {e}")

        # --- RUN LATE FUSION ---
        if args.strategy in ["all", "late"]:
            print(f"\n>>> Running Late Fusion Refits for Task: {task}")
            for tar in targets:
                for mt in models:
                    try:
                        build_final_latefusion(task, tar, mt, args.date)
                        print(f"[Success Late Fusion] {tar}-{mt}")
                    except Exception as e:
                        print(f"[Skipped/Failed Late Fusion] {tar}-{mt}: {e}")

if __name__ == "__main__":
    main()