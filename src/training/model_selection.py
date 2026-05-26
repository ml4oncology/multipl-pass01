"""
Model selection: pick best late fusion, early fusion, and unimodal models
based on auc_max from the pre-computed auc_ci_summary CSV.

Logic:
  1) Best late fusion by auc_max → determines base_type
  2) base_type locks unimodal picks
  3) Best early fusion selected independently
"""
import re
import numpy as np
import pandas as pd


# ============================================================
# Config
# ============================================================
UNIMODAL     = ["clinical", "rna", "dna", "histopathology"]
TYPE_PREF    = ["lr", "xgb", "tabpfn"]
STACKER_PREF = ["avg", "lr", "xgb", "tabpfn"]


# ============================================================
# Helpers
# ============================================================
def parse_ci(s):
    m = re.match(r"\s*\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]\s*$", str(s))
    return (float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)


def norm_base(tok):
    tok = tok.lower()
    if tok in {"lr", "logreg", "logistic"}: return "lr"
    if tok == "xgb":                        return "xgb"
    if "tabpfn" in tok:                     return "tabpfn"
    return None


def norm_stacker(tok):
    tok = tok.lower()
    if tok in {"avg", "average"}: return "avg"
    if tok in {"lr", "logreg"}:   return "lr"
    if tok == "xgb":              return "xgb"
    if "tabpfn" in tok:           return "tabpfn"
    return None


def infer_fields(name: str):
    p    = name.lower()
    toks = p.split("_")

    if any(t.startswith("latef") or "latefusion" in t for t in toks):
        modality = "latefusion"
    elif any("earlyfusion" in t for t in toks):
        modality = "earlyfusion"
    elif any("clinic" in t for t in toks):
        modality = "clinical"
    elif "rna" in toks:
        modality = "rna"
    elif "dna" in toks:
        modality = "dna"
    elif any("histo" in t for t in toks):
        modality = "histopathology"
    else:
        modality = "other"

    base_type = stacker = None
    if modality == "latefusion":
        try:
            i         = [i for i, t in enumerate(toks) if t.startswith("latef") or "latefusion" in t][0]
            base_type = norm_base(toks[i + 1])    if i + 1 < len(toks) else None
            stacker   = norm_stacker(toks[i + 2]) if i + 2 < len(toks) else None
        except Exception:
            pass
    else:
        for t in toks[::-1]:
            bt = norm_base(t)
            if bt:
                base_type = bt
                break

    return modality, base_type, stacker


def best_row(g):
    return g.sort_values("auc_max", ascending=False).iloc[0]


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Pick best late fusion, early fusion, and unimodal models."
    )
    parser.add_argument(
        "--file", "-f",
        default="../../results/DTE-FFX/COMPASS_DTE-FFX_orr_auc_ci_summary.csv",
        help=(
            "Path to the auc_ci_summary CSV produced by auc_analysis.py. Examples:\n"
            "  ../../results/DTE-FFX/COMPASS_DTE-FFX_orr_auc_ci_summary.csv\n"
            "  ../../results/DTE-GNP/COMPASS_DTE-GNP_1yOS_auc_ci_summary.csv"
        ),
    )
    args = parser.parse_args()
    csv_path = args.file

    # Load
    df = pd.read_csv(csv_path)
    df[["ci_low", "ci_high"]] = df["ci95_cc"].apply(parse_ci).apply(pd.Series)
    df["se"] = (df["ci_high"] - df["ci_low"]) / (2 * 1.96)
    df[["modality", "base_type", "stacker"]] = (
        df["predictor"].apply(infer_fields).apply(pd.Series)
    )

    # 1) Best late fusion → determines base_type
    lf = df[df["modality"] == "latefusion"].copy()
    if lf.empty:
        raise ValueError("No late-fusion rows found.")
    lf_best          = lf.sort_values("auc_max", ascending=False).iloc[0]
    chosen_base_type = lf_best["base_type"]
    chosen_latefusion = lf_best
    if chosen_base_type is None:
        raise ValueError(f"Could not parse base_type from: {lf_best['predictor']}")

    # 2) Best early fusion (independent)
    ef     = df[df["modality"] == "earlyfusion"].copy()
    ef_best = ef.sort_values("auc_max", ascending=False).iloc[0] if not ef.empty else None

    # 3) Lock unimodals to chosen base_type
    unimodal_picks = []
    for m in UNIMODAL:
        g = df[(df["modality"] == m) & (df["base_type"] == chosen_base_type)]
        if g.empty:
            continue
        r = best_row(g)
        unimodal_picks.append((m, r["predictor"], r["auc_max"], r["ci_low"], r["ci_high"]))

    # 4) Print
    print(f"Chosen base type:  {chosen_base_type}")
    print(f"Chosen late fusion: {chosen_latefusion['predictor']}  "
          f"AUC={chosen_latefusion['auc_max']:.3f}\n")

    if ef_best is not None:
        print(f"Chosen early fusion: {ef_best['predictor']}  "
              f"AUC={ef_best['auc_max']:.3f}  "
              f"CI=[{ef_best['ci_low']:.3f},{ef_best['ci_high']:.3f}]\n")
    else:
        print("No early-fusion models found.\n")

    print("Unimodal picks (locked to chosen base type):")
    for m, p, a, lo, hi in unimodal_picks:
        print(f"{m:15s}  {p:40s}  AUC={a:.3f}  CI=[{lo:.3f},{hi:.3f}]")
