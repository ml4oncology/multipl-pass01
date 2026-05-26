"""
Assemble per-model PASS-01 validation prediction CSVs into one wide DataFrame
per (task, target), matching the format expected by auc_analysis.py.

Input:  ../../results/PASS-01/{task}/*_ext_preds.csv
Output: ../../results/PASS-01/{task}/PASS01_{task}_{target}_preds.csv

Column naming mirrors build_final_preds.py so that model_selection.py and
auc_analysis.py parse predictor names identically for training and validation.

Usage (from src/validation/):
    python build_val_preds.py
"""
from pathlib import Path
import pandas as pd

# ============================================================
# Config
# ============================================================
BASE_RESULTS  = Path("../../results/PASS-01")
TASKS         = ["DTE-FFX", "DTE-GNP"]
TARGETS       = ["orr", "1yOS"]
MODALITIES    = ["Clinical", "DNA", "Histopathology", "RNA"]
MODEL_TYPES   = ["lr", "xgb", "tabpfn"]
LF_STACKERS   = ["avg", "lr", "xgb", "tabpfn"]
SUBJECT_COL   = "Subject"


# ============================================================
# Helpers
# ============================================================
def _load(path: Path, col_name: str) -> pd.DataFrame | None:
    """Load a prediction CSV and return (Subject, col_name) DataFrame."""
    if not path.exists():
        print(f"  [SKIP] {path.name}")
        return None
    df = pd.read_csv(path)[[SUBJECT_COL, "y_pred"]].copy()
    df = df.rename(columns={"y_pred": col_name})
    return df


def build_wide(task: str, target: str) -> pd.DataFrame:
    base = BASE_RESULTS / task

    # ── Ground truth from the first available unimodal file ──────────
    wide = None
    for mod in MODALITIES:
        p = base / f"{mod}_{target}_lr_ext_preds.csv"
        if p.exists():
            tmp = pd.read_csv(p)[[SUBJECT_COL, "y_true"]].copy()
            wide = tmp
            break
    if wide is None:
        raise FileNotFoundError(
            f"No unimodal prediction file found for {task}/{target} "
            f"— run validate_pipelines.py first."
        )

    def merge(df):
        nonlocal wide
        if df is not None:
            wide = wide.merge(df, on=SUBJECT_COL, how="left")

    # ── Unimodal ─────────────────────────────────────────────────────
    for mod in MODALITIES:
        for mt in MODEL_TYPES:
            p = base / f"{mod}_{target}_{mt}_ext_preds.csv"
            merge(_load(p, f"y_pred_{mod.lower()}_{mt}"))

    # ── Early Fusion ─────────────────────────────────────────────────
    for mt in MODEL_TYPES:
        p = base / f"EarlyFusion_{target}_{mt}_ext_preds.csv"
        merge(_load(p, f"y_pred_earlyfusion_{mt}"))

    # ── Late Fusion ──────────────────────────────────────────────────
    for mt in MODEL_TYPES:
        for st in LF_STACKERS:
            p = base / f"Latefusion_{target}_{mt}_{st}_ext_preds.csv"
            merge(_load(p, f"y_pred_latefusion_{mt}_{st}"))

    return wide


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    for task in TASKS:
        for target in TARGETS:
            print(f"\n=== {task} / {target} ===")
            wide = build_wide(task, target)
            out  = BASE_RESULTS / task / f"PASS01_{task}_{target}_preds.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            wide.to_csv(out, index=False)
            n_labeled = wide["y_true"].notna().sum()
            print(f"Saved: {out}  shape={wide.shape}  labeled={n_labeled}/{len(wide)}")
