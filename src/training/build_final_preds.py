from pathlib import Path
import pandas as pd
from config import BASE_RESULTS, DATE_STR

# ============================================================
# Config
# ============================================================
DATE_STR_UNIMODAL = DATE_STR
DATE_STR_EF       = DATE_STR
DATE_STR_LF       = DATE_STR

UNIMODAL_MODS = ["Clinical", "DNA", "RNA", "Histopathology"]
MODEL_TYPES   = ["lr", "xgb", "tabpfn"]
LF_STACKERS   = ["avg", "lr", "xgb", "tabpfn"]

TASKS = ["DTE-FFX", "DTE-GNP"]

TARGETS = ["orr", "1yOS"]

# PurIST scores path (prognosis only — not applicable for DTE)
PURIST_PATH = "../../data/processed/COMPASS/COMPASS_PurIST_scores.csv"


# ============================================================
# Core merge function
# ============================================================
def build_final_preds_csv_simple(csv_entries, output_csv):
    """
    Merge multiple prediction CSVs into one wide DataFrame.

    Entry formats:
      ("y_true",  path)                      — ground truth
      ("purist",  path)                      — PurIST scores
      (modality,  model_type,          path) — unimodal / early fusion
      (modality,  model_type, stacker, path) — late fusion
    """
    base_df = None

    for entry in csv_entries:
        if len(entry) == 2:
            tag, path = entry
            tag_l = tag.lower()
            if tag_l == "purist":
                df   = pd.read_csv(path).rename(columns={"predicted_prob": "y_pred_purist"})
                cols = ["donor", "y_pred_purist"]
            elif tag_l == "y_true":
                df   = pd.read_csv(path).rename(columns={"true_label": "y_true"})
                cols = ["donor", "y_true"]
            else:
                raise ValueError(f"Unknown two-tuple entry: {entry}")

        elif len(entry) == 3:
            modality, model_type, path = entry
            df      = pd.read_csv(path)
            new_col = f"y_pred_{modality}_{model_type}"
            df      = df.rename(columns={"y_pred": new_col})
            cols    = ["donor", new_col]

        elif len(entry) == 4:
            modality, model_type, stacker, path = entry
            df      = pd.read_csv(path)
            new_col = f"y_pred_{modality}_{model_type}_{stacker}"
            df      = df.rename(columns={"y_pred": new_col})
            cols    = ["donor", new_col]

        else:
            raise ValueError(f"Invalid entry format: {entry}")

        df = df[cols]
        base_df = df if base_df is None else base_df.merge(df, on="donor", how="left")

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    base_df.to_csv(output_csv, index=False)
    return base_df


# ============================================================
# Path builder
# ============================================================
def oof_avg_path(base: Path, task: str, modality: str, target: str,
                 model_type: str, stacker: str, date_str: str) -> str:
    if modality == "Latefusion":
        return str(
            base / task / modality / target / model_type / stacker / date_str
            / f"{modality}_{target}_{model_type}_{stacker}_oof_avg.csv"
        )
    else:
        return str(
            base / task / modality / target / model_type / date_str
            / f"{modality}_{target}_{model_type}_oof_avg.csv"
        )


# ============================================================
# Entry generator
# ============================================================
def build_csv_entries(task: str, target: str, base: Path) -> list:
    entries = []

    # y_true — from Clinical lr (has all donors)
    entries.append((
        "y_true",
        oof_avg_path(base, task, "Clinical", target, "lr", "", DATE_STR_UNIMODAL)
    ))

    # PurIST — prognosis only
    if task == "prognosis" and Path(PURIST_PATH).exists():
        entries.append(("purist", PURIST_PATH))

    # Unimodal
    for mod in UNIMODAL_MODS:
        for mt in MODEL_TYPES:
            entries.append((
                mod.lower(), mt,
                oof_avg_path(base, task, mod, target, mt, "", DATE_STR_UNIMODAL)
            ))

    # Early Fusion
    for mt in MODEL_TYPES:
        entries.append((
            "earlyfusion", mt,
            oof_avg_path(base, task, "EarlyFusion", target, mt, "", DATE_STR_EF)
        ))

    # Late Fusion
    for mt in MODEL_TYPES:
        for st in LF_STACKERS:
            entries.append((
                "latefusion", mt, st,
                oof_avg_path(base, task, "Latefusion", target, mt, st, DATE_STR_LF)
            ))

    return entries


# ============================================================
# Run for all tasks and targets
# ============================================================
if __name__ == "__main__":
    base = Path(BASE_RESULTS)

    for task in TASKS:
        for target in TARGETS:
            print(f"\n=== Building preds CSV: task={task} target={target} ===")
            entries    = build_csv_entries(task, target, base)
            output_csv = base / task / f"COMPASS_{task}_{target}_preds.csv"
            final_df   = build_final_preds_csv_simple(entries, output_csv)
            print(f"Saved: {output_csv}  shape={final_df.shape}")
