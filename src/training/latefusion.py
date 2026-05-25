#!/usr/bin/env python3
"""
Late Fusion Runner (registry-based)

Examples:
  python latefusion.py --targets orr --base-set lr --stacker lr --seeds 7270 --task prognosis \
    --paths-json latefusion_paths_sets.json

  python latefusion.py --targets orr --base-set lr --stacker lr --seeds 7270 --task DTE-FFX \
    --paths-json latefusion_paths_sets_DTE-FFX.json
"""
from __future__ import annotations
import argparse, json, warnings, logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import joblib, numpy as np, pandas as pd, sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline

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

from config import RANDOM_STATES, DATE_STR, BASE_RESULTS, BASE_DATA

# ---- logging/warnings ----
logging.basicConfig(filename="warnings.log", filemode="w",
                    level=logging.WARNING, format="%(levelname)s: %(message)s")
def custom_showwarning(message, category, filename, lineno, file=None, line=None):
    logging.warning(f"{category.__name__}: {message} (from {filename}:{lineno})")
warnings.showwarning = custom_showwarning

# ---- helpers ----
def _csv_list(s: str | None) -> List[str]:
    if not s: return []
    return [x.strip() for x in (s.split(",") if "," in s else s.split()) if x.strip()]

def _csv_int_list(s: str | None) -> List[int]:
    return [int(x) for x in _csv_list(s or "")]

def load_registry(path: Path) -> pd.DataFrame:
    reg = pd.read_csv(path)
    reg["donor"]      = reg["donor"].astype(str)
    reg["seed"]       = reg["seed"].astype(int)
    reg["outer_fold"] = reg["outer_fold"].astype(int)
    return reg

def load_oof_csv(dirpath: Path, modality: str, seed: int) -> pd.DataFrame:
    fp = dirpath / f"{modality}_{seed}_predictions.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing OOF file: {fp}")
    df = pd.read_csv(fp).rename(columns={
        "fold_number":    "outer_fold",
        "true_label":     "y",
        "predicted_prob": f"p_{modality}"
    })
    df["donor"]      = df["donor"].astype(str)
    df["outer_fold"] = df["outer_fold"].astype(int)
    return df[["donor", "outer_fold", "y", f"p_{modality}"]]


class ProbToLogitScaler(BaseEstimator, TransformerMixin):
    def __init__(self, eps: float = 1e-6):
        self.eps = float(eps)

    def fit(self, X, y=None):
        X  = np.asarray(X, dtype=float)
        Xc = np.clip(X, self.eps, 1 - self.eps)
        L  = np.log(Xc / (1 - Xc))
        self.mu_ = np.nanmean(L, axis=0)
        self.sd_ = np.nanstd(L, axis=0, ddof=0)
        bad = ~np.isfinite(self.sd_) | (self.sd_ == 0)
        if np.any(bad):
            self.sd_[bad] = 1.0
        return self

    def transform(self, X):
        X   = np.asarray(X, dtype=float)
        nan = np.isnan(X)
        Xc  = np.clip(X, self.eps, 1 - self.eps)
        L   = np.log(Xc / (1 - Xc))
        Z   = (L - self.mu_) / self.sd_
        Z[nan] = np.nan
        return Z


def build_meta_for_seed(
    registry: pd.DataFrame,
    seed: int,
    modality_dirs: Dict[str, Path],
    modalities: List[str],
) -> pd.DataFrame:
    base    = registry.loc[registry["seed"] == seed, ["donor", "outer_fold"]].drop_duplicates()
    meta    = base.copy()
    y_added = False
    for m in modalities:
        dfm  = load_oof_csv(modality_dirs[m], m, seed)
        dfm  = dfm.drop(columns=["outer_fold"]).merge(base, on="donor", how="right")
        meta = meta.merge(dfm[["donor", f"p_{m}"]], on="donor", how="left")
        if not y_added:
            meta    = meta.merge(dfm[["donor", "y"]], on="donor", how="left")
            y_added = True
    for m in modalities:
        meta[f"has_{m}"] = ~meta[f"p_{m}"].isna()
    meta = meta.loc[meta[[f"has_{m}" for m in modalities]].any(axis=1)].reset_index(drop=True)
    meta["y"]          = meta["y"].astype(int)
    meta["outer_fold"] = meta["outer_fold"].astype(int)
    return meta


def make_stacker_pipeline(
    modalities: List[str],
    stacker: str = "lr",
    impute: str = "mean",
    random_state: int = 42,
) -> Pipeline | None:
    prob_cols = [f"p_{m}" for m in modalities]
    flag_cols = [f"has_{m}" for m in modalities]
    st = stacker.lower()
    if st == "avg":
        return None

    prob_steps_lr_or_tabpfn = []
    if impute == "0.5":
        prob_steps_lr_or_tabpfn.append(
            ("pre_impute_p", SimpleImputer(strategy="constant", fill_value=0.5))
        )
    prob_steps_lr_or_tabpfn.append(("logit_scale", ProbToLogitScaler(eps=1e-6)))
    if impute != "0.5":
        prob_steps_lr_or_tabpfn.append(
            ("imp_after", SimpleImputer(strategy="constant", fill_value=0.0))
        )

    prob_steps_xgb = [("logit_scale", ProbToLogitScaler(eps=1e-6))]

    if st == "lr":
        pre = ColumnTransformer([
            ("prob_pipe",         Pipeline(prob_steps_lr_or_tabpfn), prob_cols),
            ("flags_passthrough", "passthrough",                      flag_cols),
        ], remainder="drop")
        clf = LogisticRegression(
            penalty="l2", solver="lbfgs", max_iter=5000,
            C=1.0, class_weight="balanced", random_state=random_state,
        )
        return Pipeline([("pre", pre), ("clf", clf)])

    if st == "xgb":
        if not _HAVE_XGB:
            raise ImportError("xgboost not installed.")
        pre = ColumnTransformer([
            ("prob_pipe",         Pipeline(prob_steps_xgb), prob_cols),
            ("flags_passthrough", "passthrough",             flag_cols),
        ], remainder="drop")
        clf = XGBClassifier(
            objective="binary:logistic", eval_metric="auc", tree_method="hist",
            n_estimators=400, max_depth=2, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9,
            n_jobs=-1, random_state=random_state,
        )
        return Pipeline([("pre", pre), ("clf", clf)])

    if st == "tabpfn":
        if not _HAVE_TABPFN:
            raise ImportError("TabPFN not installed.")
        pre = ColumnTransformer([
            ("prob_pipe",         Pipeline(prob_steps_lr_or_tabpfn), prob_cols),
            ("flags_passthrough", "passthrough",                      flag_cols),
        ], remainder="drop")
        clf = TabPFNClassifier(device="cpu")
        return Pipeline([("pre", pre), ("clf", clf)])

    raise ValueError("stacker must be one of {'lr','xgb','tabpfn','avg'}")


def get_stacker_param_grid(stacker: str = "lr") -> List[Dict[str, Any]]:
    st = stacker.lower()
    if st == "lr":
        return [{"clf__C": [0.2, 0.5, 1.0, 2.0]}]
    if st == "xgb":
        return [{"clf__n_estimators": [300, 500], "clf__max_depth": [2, 3, 4],
                 "clf__learning_rate": [0.05, 0.1], "clf__subsample": [0.8, 1.0],
                 "clf__colsample_bytree": [0.7, 0.9, 1.0]}]
    return []


def run_fusion_nested_cv_for_seed(
    registry: pd.DataFrame,
    modality_dirs: Dict[str, Path],
    modalities: List[str],
    seed: int,
    stacker: str = "lr",
    impute: str = "mean",
    tune_model: bool = False,
    n_splits_inner: int = 5,
) -> Tuple[List[float], pd.DataFrame, List[Dict[str, Any]], List[str]]:
    meta          = build_meta_for_seed(registry, seed, modality_dirs, modalities)
    prob_cols     = [f"p_{m}" for m in modalities]
    flag_cols     = [f"has_{m}" for m in modalities]
    expected_cols = prob_cols + flag_cols
    folds         = sorted(meta["outer_fold"].unique().tolist())
    outer_aucs, fold_preds, fold_models = [], [], []
    st = stacker.lower()

    for f in folds:
        tr_mask   = meta["outer_fold"] != f
        te_mask   = meta["outer_fold"] == f
        X_tr      = meta.loc[tr_mask, expected_cols].copy()
        y_tr      = meta.loc[tr_mask, "y"].astype(int).values
        X_te      = meta.loc[te_mask, expected_cols].copy()
        y_te      = meta.loc[te_mask, "y"].astype(int).values
        donors_te = meta.loc[te_mask, "donor"].values

        if st == "avg":
            # Logit-average then sigmoid (more principled than raw prob average)
            eps    = 1e-6
            probs  = X_te[prob_cols].clip(eps, 1 - eps)
            logits = np.log(probs / (1 - probs))
            avg_logit = logits.mean(axis=1, skipna=True).values
            p_te   = 1 / (1 + np.exp(-avg_logit))
            auc    = roc_auc_score(y_te, p_te)
            fold_preds.append(pd.DataFrame({
                "fold_number": f, "donor": donors_te,
                "true_label": y_te, "predicted_prob": p_te,
            }))
            outer_aucs.append(float(auc))
            fold_models.append({
                "fold": f, "model": None, "expected_columns": prob_cols,
                "modalities": modalities, "stacker": "avg",
                "impute": None, "best_params": None,
            })
            print(f"[Seed {seed} | Fold {f}] AUC = {auc:.3f}  |  stacker=avg (logit)  N_te={len(y_te)}")
            continue

        pipe        = make_stacker_pipeline(modalities, stacker=st, impute=impute, random_state=seed)
        best_params = None
        if tune_model and st in {"lr", "xgb"}:
            inner    = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=seed + 137)
            gs       = GridSearchCV(estimator=pipe, param_grid=get_stacker_param_grid(st),
                                    scoring="roc_auc", cv=inner, refit=True,
                                    n_jobs=-1, verbose=0, return_train_score=False)
            gs.fit(X_tr, y_tr)
            best_est    = gs.best_estimator_
            best_params = gs.best_params_
        else:
            best_est = pipe.fit(X_tr, y_tr)

        p_te = best_est.predict_proba(X_te)[:, 1]
        auc  = roc_auc_score(y_te, p_te)
        outer_aucs.append(float(auc))
        fold_preds.append(pd.DataFrame({
            "fold_number": f, "donor": donors_te,
            "true_label": y_te, "predicted_prob": p_te,
        }))
        fold_models.append({
            "fold": f, "model": best_est, "expected_columns": expected_cols,
            "modalities": modalities, "stacker": st,
            "impute": (impute if st in {"lr"} else None),
            "best_params": best_params,
        })
        print(f"[Seed {seed} | Fold {f}] AUC = {auc:.3f}  |  stacker={st}  N_te={len(y_te)}")

    return outer_aucs, pd.concat(fold_preds, axis=0, ignore_index=True), fold_models, expected_cols


def save_fusion_artifacts_per_seed(
    RESULTS_DIR: Path,
    seed: int,
    outer_aucs: List[float],
    fold_predictions: pd.DataFrame,
    fold_models: List[Dict[str, Any]],
    task: str,
    stacker_name: str,
    modalities: List[str],
    extra_manifest: Dict[str, Any] | None = None,
):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base = f"Latefusion_{seed}"
    pd.DataFrame({"fold": np.arange(1, len(outer_aucs) + 1), "auc": outer_aucs}
                 ).to_csv(RESULTS_DIR / f"{base}_scores.csv", index=False)
    fold_predictions.to_csv(RESULTS_DIR / f"{base}_predictions.csv", index=False)
    joblib.dump(fold_models, RESULTS_DIR / f"{base}_models.joblib", compress=3)
    manifest = {
        "artifact": "late_fusion", "task": task, "modalities": modalities,
        "stacker": stacker_name, "n_folds_outer": len(outer_aucs),
        "sklearn_version": sklearn.__version__, "random_state": seed,
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    pd.Series(manifest).to_json(RESULTS_DIR / f"{base}_manifest.json", indent=2)


def load_paths_sets_json(path_json: Path) -> Dict[str, Dict[str, str]]:
    with open(path_json, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("paths JSON must be an object: {'lr': {...}, 'xgb': {...}, 'tabpfn': {...}}")
    return data


def resolve_modality_dirs_from_set(
    paths_sets: Dict[str, Dict[str, str]],
    base_set: str,
    target: str,
    modalities: List[str] | None,
    strict: bool = True,
) -> Tuple[List[str], Dict[str, Path]]:
    base_set = base_set.lower()
    if base_set not in paths_sets:
        raise ValueError(f"Base set '{base_set}' not found. Keys: {list(paths_sets.keys())}")
    templ               = paths_sets[base_set]
    available_modalities = list(templ.keys())
    mods                = modalities or available_modalities
    missing             = [m for m in mods if m not in templ]
    if missing and strict:
        raise ValueError(f"Missing modalities in base set '{base_set}': {missing}")
    mods     = [m for m in mods if m in templ]
    mod_dirs = {m: Path(templ[m].replace("${target}", target)) for m in mods}
    for m, path in mod_dirs.items():
        if not path.exists():
            print(f"[WARN] Modality dir does not exist: {m} -> {path}")
    return mods, mod_dirs


def main():
    p = argparse.ArgumentParser(description="Late Fusion over unimodal OOF predictions")
    p.add_argument("--targets",        type=str, required=True)
    p.add_argument("--modalities",     type=str, default=None)
    p.add_argument("--base-set",       type=str, required=True, choices=["lr", "xgb", "tabpfn"])
    p.add_argument("--stacker",        type=str, default="lr")
    p.add_argument("--impute",         type=str, default="mean")
    p.add_argument("--tune-model",     action="store_true")
    p.add_argument("--n-splits-inner", type=int, default=5)
    p.add_argument("--seeds",          type=str, default=",".join(str(s) for s in RANDOM_STATES))
    p.add_argument("--task",           type=str, default="prognosis",
                   help="prognosis | DTE-FFX | DTE-GNP (default: prognosis)")
    p.add_argument("--registry-csv",   type=str, default=None,
                   help="Path to split registry CSV. Defaults to BASE_DATA/processed/COMPASS/{task}/split_registry.csv")
    p.add_argument("--results-root",   type=str, default=None)
    p.add_argument("--paths-json",     type=str, required=True)
    p.add_argument("--strict",         action="store_true")
    args = p.parse_args()

    targets        = _csv_list(args.targets)
    modalities_req = _csv_list(args.modalities) if args.modalities else None
    seeds          = _csv_int_list(args.seeds)
    stacker        = args.stacker.lower()

    if stacker in {"xgb", "xgboost"} and not _HAVE_XGB:
        print("[WARN] xgboost not installed; cannot use stacker='xgb'.")
    if stacker == "tabpfn" and not _HAVE_TABPFN:
        print("[WARN] TabPFN not installed; cannot use stacker='tabpfn'.")

    # Resolve registry path
    registry_path = args.registry_csv or str(
        BASE_DATA / "processed" / "COMPASS" / args.task / "split_registry.csv"
    )
    reg = load_registry(Path(registry_path))
    print(f"Loaded registry from: {registry_path}  shape={reg.shape}")

    paths_sets = load_paths_sets_json(Path(args.paths_json))
    base_root  = Path(args.results_root) if args.results_root else BASE_RESULTS

    for target in targets:
        mods, mod_dirs = resolve_modality_dirs_from_set(
            paths_sets, args.base_set, target, modalities_req,
            strict=bool(args.strict or True),
        )

        # Task-namespaced output directory
        out_dir = base_root / args.task / "Latefusion" / target / args.base_set / stacker / DATE_STR
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Late Fusion | task={args.task} | target={target} | "
              f"base-set={args.base_set} | stacker={stacker} | modalities={mods} ===")
        for m in mods:
            print(f"  - {m}: {mod_dirs[m]}")

        for seed in seeds:
            print(f"\n=== {args.task} | {target} | stacker={stacker} | seed={seed} ===")
            outer_aucs, fold_preds, fold_models, expected_cols = run_fusion_nested_cv_for_seed(
                registry=reg, modality_dirs=mod_dirs, modalities=mods,
                seed=seed, stacker=stacker, impute=args.impute,
                tune_model=args.tune_model, n_splits_inner=args.n_splits_inner,
            )
            save_fusion_artifacts_per_seed(
                RESULTS_DIR=out_dir, seed=seed, outer_aucs=outer_aucs,
                fold_predictions=fold_preds, fold_models=fold_models,
                task=args.task, stacker_name=stacker, modalities=mods,
                extra_manifest={
                    "impute":            args.impute if stacker in {"lr", "tabpfn"} else None,
                    "tuned":             bool(args.tune_model and stacker in {"lr", "xgb"}),
                    "expected_columns":  expected_cols,
                    "paths_json":        str(Path(args.paths_json).resolve()),
                    "base_set":          args.base_set,
                },
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
