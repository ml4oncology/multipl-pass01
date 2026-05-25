from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional, List

from config import RANDOM_STATES, BASE_RESULTS


def build_seed_path(
    base: Path,
    TASK: str,
    MODALITY: str,
    TARGET: str,
    MODEL_TYPE: str,
    stacker: str,
    DATE_STR: str,
    seed: int,
) -> Path:
    """
    <BASE_RESULTS>/<TASK>/<MODALITY>/<TARGET>/<MODEL_TYPE>/<DATE_STR>/<MODALITY>_{seed}_predictions.csv
    Latefusion adds an extra stacker subdirectory.
    """
    if MODALITY != "Latefusion":
        return (
            Path(base) / TASK / MODALITY / TARGET / MODEL_TYPE / DATE_STR
            / f"{MODALITY}_{seed}_predictions.csv"
        )
    else:
        return (
            Path(base) / TASK / MODALITY / TARGET / MODEL_TYPE / stacker / DATE_STR
            / f"{MODALITY}_{seed}_predictions.csv"
        )


def average_oof_predictions(
    TASK: str,
    MODALITY: str,
    TARGET: str,
    MODEL_TYPE: str,
    stacker: str,
    DATE_STR: str,
    BASE_RESULTS: Path,
    RANDOM_STATES: List[int],
    save_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: donor, y_true, y_pred
    where y_pred is the logit-averaged predicted probability across seeds.
    """
    base_dir = Path(BASE_RESULTS)
    merged = None
    used_seeds, missing = [], []

    for seed in RANDOM_STATES:
        f = build_seed_path(base_dir, TASK, MODALITY, TARGET, MODEL_TYPE, stacker, DATE_STR, seed)
        if not f.exists():
            missing.append((seed, str(f)))
            continue

        df = pd.read_csv(f, dtype={"donor": str})
        df = df[["donor", "true_label", "predicted_prob"]].rename(
            columns={"predicted_prob": f"p_{seed}"}
        )
        df = df.sort_values("donor").reset_index(drop=True)

        if merged is None:
            merged = df.copy()
        else:
            merged = pd.merge(merged, df, on=["donor", "true_label"], how="inner")

        used_seeds.append(seed)

    if merged is None:
        raise FileNotFoundError(
            f"No predictions found for any RANDOM_STATES. "
            f"Check TASK={TASK} MODALITY={MODALITY} TARGET={TARGET} MODEL_TYPE={MODEL_TYPE}"
        )

    # Logit-average across seeds then sigmoid
    prob_cols = [c for c in merged.columns if c.startswith("p_")]
    eps    = 1e-6
    probs  = merged[prob_cols].clip(eps, 1 - eps)
    logits = np.log(probs / (1 - probs))
    merged["y_pred"] = 1 / (1 + np.exp(-logits.mean(axis=1, skipna=True)))

    out = merged[["donor", "true_label", "y_pred"]].rename(columns={"true_label": "y_true"})

    print(f"[{TASK} | {MODALITY} | {TARGET} | {MODEL_TYPE} | {DATE_STR}]")
    print(f"Used seeds ({len(used_seeds)}): {used_seeds}")
    if missing:
        print("Missing seeds/files:")
        for sd, path in missing:
            print(f"  - seed {sd}: {path}")
    print(f"Rows in output: {len(out)}\n")

    if save_csv is not None:
        save_csv = Path(save_csv)
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(save_csv, index=False)
        print(f"Saved: {save_csv}")

    return out


# ============================================================
# Run aggregation for all tasks, modalities, targets, models
# ============================================================
if __name__ == "__main__":

    DATE_STR_UNIMODAL = "20260329"
    DATE_STR_EF       = "20260329"
    DATE_STR_LF       = "20260329"

    # Run all tasks — comment out as needed
    TASKS = ["prognosis", "DTE-FFX", "DTE-GNP"]
    # TASKS = ["DTE-FFX", "DTE-GNP"]  # DTE only

    for TASK in TASKS:
        print(f"\n{'='*60}")
        print(f"TASK: {TASK}")
        print(f"{'='*60}")

        # ── Unimodal ──────────────────────────────────────────────
        for mod in ["Clinical", "DNA", "RNA", "Histopathology"]:
            for tar in ["orr", "1yOS"]:
                for mt in ["lr", "xgb", "tabpfn"]:
                    save_csv = (
                        Path(BASE_RESULTS) / TASK / mod / tar / mt / DATE_STR_UNIMODAL
                        / f"{mod}_{tar}_{mt}_oof_avg.csv"
                    )
                    average_oof_predictions(
                        TASK=TASK, MODALITY=mod, TARGET=tar, MODEL_TYPE=mt,
                        stacker="", DATE_STR=DATE_STR_UNIMODAL,
                        BASE_RESULTS=BASE_RESULTS, RANDOM_STATES=RANDOM_STATES,
                        save_csv=save_csv,
                    )

        # ── Early Fusion ──────────────────────────────────────────
        for tar in ["orr", "1yOS"]:
            for mt in ["lr", "xgb", "tabpfn"]:
                save_csv = (
                    Path(BASE_RESULTS) / TASK / "EarlyFusion" / tar / mt / DATE_STR_EF
                    / f"EarlyFusion_{tar}_{mt}_oof_avg.csv"
                )
                average_oof_predictions(
                    TASK=TASK, MODALITY="EarlyFusion", TARGET=tar, MODEL_TYPE=mt,
                    stacker="", DATE_STR=DATE_STR_EF,
                    BASE_RESULTS=BASE_RESULTS, RANDOM_STATES=RANDOM_STATES,
                    save_csv=save_csv,
                )

        # ── Late Fusion ───────────────────────────────────────────
        for tar in ["orr", "1yOS"]:
            for mt in ["lr", "xgb", "tabpfn"]:
                for stacker in ["avg", "lr", "xgb", "tabpfn"]:
                    save_csv = (
                        Path(BASE_RESULTS) / TASK / "Latefusion" / tar / mt / stacker / DATE_STR_LF
                        / f"Latefusion_{tar}_{mt}_{stacker}_oof_avg.csv"
                    )
                    average_oof_predictions(
                        TASK=TASK, MODALITY="Latefusion", TARGET=tar, MODEL_TYPE=mt,
                        stacker=stacker, DATE_STR=DATE_STR_LF,
                        BASE_RESULTS=BASE_RESULTS, RANDOM_STATES=RANDOM_STATES,
                        save_csv=save_csv,
                    )
