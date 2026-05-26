"""
AUC + 95% CI summary and pairwise DeLong tests.

Usage:
    Set file_path below and run. Outputs two CSVs:
      {base}_auc_ci_summary.csv
      {base}_pairwise_auc_tests_complete_case.csv
"""
import warnings
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy import stats
from scipy.stats import bootstrap as scipy_bootstrap


# ============================================================
# Helpers
# ============================================================
def compute_auc_ci(truth: np.ndarray, pred: np.ndarray, n_boot: int = 2000):
    mask  = ~(np.isnan(truth) | np.isnan(pred))
    truth = truth[mask].astype(int)
    pred  = pred[mask]
    n     = len(truth)

    if n == 0 or len(np.unique(truth)) < 2:
        return {"n": n, "auc": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    auc = roc_auc_score(truth, pred)

    def _auc_stat(truth_b, pred_b):
        if len(np.unique(truth_b)) < 2:
            return np.nan
        return roc_auc_score(truth_b.astype(int), pred_b)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = scipy_bootstrap(
            (truth, pred),
            statistic=lambda t, p: _auc_stat(t, p),
            n_resamples=n_boot,
            confidence_level=0.95,
            random_state=42,
            method="percentile",
            paired=True,
        )
    return {"n": n, "auc": auc,
            "ci_low": res.confidence_interval.low,
            "ci_high": res.confidence_interval.high}


def delong_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    """
    One-sided DeLong test: H1 = AUC(a) > AUC(b).
    Based on: DeLong et al. (1988), implemented via variance of U-statistics.
    """
    def _auc_and_kernel(y, pred):
        pos = pred[y == 1]
        neg = pred[y == 0]
        n1, n0 = len(pos), len(neg)
        if n1 == 0 or n0 == 0:
            return np.nan, None, None, n1, n0
        kernel = np.zeros((n1, n0))
        for i in range(n1):
            kernel[i] = (pos[i] > neg) + 0.5 * (pos[i] == neg)
        auc = kernel.mean()
        return auc, kernel.mean(axis=1), kernel.mean(axis=0), n1, n0

    y = y_true.astype(int)
    auc_a, v10_a, v01_a, n1, n0 = _auc_and_kernel(y, pred_a)
    auc_b, v10_b, v01_b, _, _   = _auc_and_kernel(y, pred_b)

    if np.isnan(auc_a) or np.isnan(auc_b):
        return np.nan

    s10 = np.cov(v10_a, v10_b, ddof=1) / n1
    s01 = np.cov(v01_a, v01_b, ddof=1) / n0
    cov_mat  = s10 + s01
    var_diff = cov_mat[0, 0] + cov_mat[1, 1] - 2 * cov_mat[0, 1]

    if var_diff <= 0:
        return np.nan

    z = (auc_a - auc_b) / np.sqrt(var_diff)
    return 1 - stats.norm.cdf(z)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    import argparse
    np.random.seed(42)

    parser = argparse.ArgumentParser(
        description="AUC + 95% CI summary and pairwise DeLong tests."
    )
    parser.add_argument(
        "--file", "-f",
        default="../../results/DTE-FFX/COMPASS_DTE-FFX_orr_preds.csv",
        help=(
            "Path to the preds CSV produced by build_final_preds.py. "
            "Examples:\n"
            "  ../../results/DTE-FFX/COMPASS_DTE-FFX_orr_preds.csv\n"
            "  ../../results/DTE-FFX/COMPASS_DTE-FFX_1yOS_preds.csv\n"
            "  ../../results/DTE-GNP/COMPASS_DTE-GNP_orr_preds.csv\n"
            "  ../../results/DTE-GNP/COMPASS_DTE-GNP_1yOS_preds.csv"
        ),
    )
    args = parser.parse_args()
    file_path = args.file

    # ── Load ──────────────────────────────────────────────────────
    df = pd.read_csv(file_path)

    truth_candidates = ["y_true", "label", "target", "y", "outcome", "response"]
    truth_col = next((c for c in df.columns if c.lower() in truth_candidates), None)
    if truth_col is None:
        raise ValueError(f"Could not find ground-truth column. Expected one of: {truth_candidates}")

    df[truth_col] = pd.to_numeric(df[truth_col], errors="coerce")
    pred_cols     = [c for c in df.columns if c.startswith("y_pred_")]
    if not pred_cols:
        raise ValueError("No predictor columns found (expected columns starting with 'y_pred_').")

    df_cc = df[[truth_col] + pred_cols].dropna()
    print(f"Complete-case N: {len(df_cc)} (from {len(df)} total)")

    # ── Part 1: AUC + CI summary ──────────────────────────────────
    print("\nComputing AUC + CI for each predictor...")
    rows = []
    for col in pred_cols:
        ok_max   = df[[truth_col, col]].dropna()
        max_stat = compute_auc_ci(ok_max[truth_col].values, ok_max[col].values)
        cc_stat  = compute_auc_ci(df_cc[truth_col].values, df_cc[col].values)

        rows.append({
            "predictor": col,
            "n_max":    max_stat["n"],
            "auc_max":  round(max_stat["auc"], 3) if not np.isnan(max_stat["auc"]) else np.nan,
            "ci95_max": f"[{max_stat['ci_low']:.3f}, {max_stat['ci_high']:.3f}]"
                        if not np.isnan(max_stat["auc"]) else None,
            "n_cc":     cc_stat["n"],
            "auc_cc":   round(cc_stat["auc"], 3) if not np.isnan(cc_stat["auc"]) else np.nan,
            "ci95_cc":  f"[{cc_stat['ci_low']:.3f}, {cc_stat['ci_high']:.3f}]"
                        if not np.isnan(cc_stat["auc"]) else None,
        })

    summary_df = pd.DataFrame(rows).sort_values("auc_cc", ascending=False).reset_index(drop=True)
    print(summary_df.to_string(index=False))

    # ── Part 2: Pairwise DeLong tests ─────────────────────────────
    if len(df_cc) > 0:
        y_cc  = df_cc[truth_col].values.astype(int)
        pairs = list(itertools.combinations(pred_cols, 2))
        print(f"\nRunning {len(pairs)} pairwise DeLong tests...")

        pairwise_rows = []
        for a, b in pairs:
            pred_a = df_cc[a].values
            pred_b = df_cc[b].values
            auc_a  = roc_auc_score(y_cc, pred_a)
            auc_b  = roc_auc_score(y_cc, pred_b)

            if np.isnan(auc_a) or np.isnan(auc_b):
                continue

            if auc_a >= auc_b:
                winner, loser = a, b
                auc_w, auc_l  = auc_a, auc_b
                p_val         = delong_test(y_cc, pred_a, pred_b)
            else:
                winner, loser = b, a
                auc_w, auc_l  = auc_b, auc_a
                p_val         = delong_test(y_cc, pred_b, pred_a)

            pairwise_rows.append({
                "winner":      winner,
                "loser":       loser,
                "auc_w":       round(auc_w, 3),
                "auc_l":       round(auc_l, 3),
                "delta":       round(auc_w - auc_l, 3),
                "p_val":       float(f"{p_val:.3g}") if not np.isnan(p_val) else np.nan,
                "significant": (not np.isnan(p_val)) and (p_val < 0.05),
            })

        pairwise_df = pd.DataFrame(pairwise_rows)

        sig    = pairwise_df[pairwise_df["significant"]].sort_values("p_val")
        nonsig = pairwise_df[~pairwise_df["significant"]].sort_values("p_val")

        print("\n=== Pairwise AUC comparisons (complete-case, one-sided DeLong) ===")
        if len(sig) > 0:
            print(f"\nSignificant (alpha=0.05): {len(sig)} pairs")
            for _, r in sig.iterrows():
                print(f"  {r['winner']} > {r['loser']}: ΔAUC={r['delta']:.3f} "
                      f"({r['auc_w']:.3f} vs {r['auc_l']:.3f}), p={r['p_val']}")
        else:
            print("No significant superiority at alpha=0.05.")

        if len(nonsig) > 0:
            print(f"\nNon-significant: {len(nonsig)} pairs (showing up to 10)")
            for _, r in nonsig.head(10).iterrows():
                print(f"  {r['winner']} > {r['loser']}: ΔAUC={r['delta']:.3f} "
                      f"({r['auc_w']:.3f} vs {r['auc_l']:.3f}), p={r['p_val']}")
    else:
        pairwise_df = pd.DataFrame()
        print("No complete-case rows — skipping pairwise tests.")

    # ── Save ──────────────────────────────────────────────────────
    file_path_obj = Path(file_path)
    base_clean    = file_path_obj.stem
    for pat in ["_ext_preds", "_preds", "_oof_avg", "_oof_mean", "_oof"]:
        base_clean = base_clean.replace(pat, "")

    out_summary  = file_path_obj.parent / f"{base_clean}_auc_ci_summary.csv"
    out_pairwise = file_path_obj.parent / f"{base_clean}_pairwise_auc_tests_complete_case.csv"

    summary_df.to_csv(out_summary, index=False)
    pairwise_df.to_csv(out_pairwise, index=False)

    print(f"\nSaved AUC/CI summary to:    {out_summary}")
    print(f"Saved pairwise tests to:    {out_pairwise}")
