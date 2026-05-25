# =========================
# Split Registry Helpers (robust + group-aware)
# =========================
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Tuple

import numpy as np
import pandas as pd

# Prefer group-wise stratification to keep donors intact across folds
try:  # sklearn ≥ 1.1
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:  # pragma: no cover
    StratifiedGroupKFold = None  # type: ignore


# -------------------------
# Registry construction
# -------------------------
def make_split_registry(
    donor_ids: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    seeds: Iterable[int],
    n_splits: int = 5,
    shuffle: bool = True,
) -> pd.DataFrame:
    """
    Build a *donor-level* outer-fold registry with stratification by label and grouping by donor.

    Returns a long-form DataFrame with columns: [seed, donor, outer_fold].
    Each row indicates which outer fold a donor belongs to (1..n_splits) for a given seed.

    Notes
    -----
    - Uses StratifiedGroupKFold (sklearn ≥ 1.1). If unavailable, raises an informative error.
    - Fold indices start at 1 (not 0) for readability and backwards-compat.
    """
    if StratifiedGroupKFold is None:
        raise ImportError(
            "StratifiedGroupKFold is not available. Please upgrade scikit-learn (>=1.1)."
        )

    donors = pd.Series(donor_ids, copy=False).astype(str).values
    labels = pd.Series(y, copy=False).astype(int).values
    assert len(donors) == len(labels), "donor_ids and y must align"

    rows: list[tuple[int, str, int]] = []
    for seed in seeds:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=int(seed))
        # X can be dummy; sgkf uses y (labels) and groups (donors)
        for fold_idx, (_, te_idx) in enumerate(
            sgkf.split(np.zeros(len(labels)), labels, groups=donors), start=1
        ):
            # Assign each donor appearing in the test set to this fold
            for d in np.unique(donors[te_idx]):
                rows.append((int(seed), str(d), int(fold_idx)))

    reg = pd.DataFrame(rows, columns=["seed", "donor", "outer_fold"])

    # Sanity: each (seed, donor) appears exactly once
    if reg.duplicated(["seed", "donor"]).any():
        dup = reg[reg.duplicated(["seed", "donor"], keep=False)].sort_values(["seed", "donor"])  # pragma: no cover
        raise ValueError(
            "Registry contains duplicate (seed, donor) assignments.\n" + str(dup.head())
        )

    return reg


def save_registry(registry: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(path, index=False)


def load_registry(path: Path) -> pd.DataFrame:
    # Keep dtypes stable
    return pd.read_csv(path, dtype={"seed": int, "donor": str, "outer_fold": int})


# -------------------------
# Registry ↔ modality alignment
# -------------------------

def attach_registry_for_seed(
    donor_ids: pd.Series | np.ndarray,
    registry: pd.DataFrame,
    seed: int,
    how: str = "left",
) -> pd.DataFrame:
    """
    Return a per-row mapping DataFrame aligned to the provided donor_ids order, with columns:
        [donor, row_idx, outer_fold]

    Parameters
    ----------
    donor_ids : sequence of donor IDs for the current modality (same length as X/y)
    registry  : master registry from make_split_registry
    seed      : which seed row to select from the registry
    how       : pandas merge method: "left" (keep all local rows, outer_fold may be NaN)
                or "inner" (drop rows not in registry for this seed).

    Notes
    -----
    - We validate m:1 (many local rows per donor to one registry row) to avoid cartesian blowups.
    - Includes a local positional index column (row_idx) so downstream splitters can work without re-merge.
    """
    donors = pd.Series(donor_ids, copy=False).astype(str).values

    local = pd.DataFrame({
        "donor": donors,
        "row_idx": np.arange(len(donors), dtype=int),
    })

    reg_seed = registry.loc[registry["seed"] == int(seed), ["donor", "outer_fold"]]

    # Validate uniqueness of (seed, donor) in registry
    if reg_seed.duplicated("donor").any():  # pragma: no cover
        dup = reg_seed[reg_seed.duplicated("donor", keep=False)].sort_values("donor")
        raise ValueError(
            "Registry has duplicate donors for this seed; expected one row per donor.\n" + str(dup.head())
        )

    out = local.merge(reg_seed, on="donor", how=how, validate="m:1")
    return out


# -------------------------
# Fold iterator (outer CV)
# -------------------------

def iter_outer_folds_from_registry(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    donor_ids: pd.Series | np.ndarray,
    registry: pd.DataFrame,
    seed: int,
    how: str = "inner",
) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
    """
    Yield (fold_number, train_index, test_index) as *positional* indices into X/y/ids for the
    provided modality, respecting the donor-level registry for the given seed.

    - No cartesian merges; mapping is strictly per-row with a stored local position (row_idx).
    - If `how` is "inner", only donors present in the registry are kept (recommended).
      If "left", rows without an assignment are ignored in all folds.
    - Fold numbers are the 1..n_splits values coming from the registry; they are not re-labeled.
    """
    n = len(X)
    assert n == len(y) == len(donor_ids), "X/y/donor_ids must have equal length"

    map_df = attach_registry_for_seed(donor_ids, registry, seed, how=how)

    # Restrict to rows that actually have an assigned outer_fold
    if "outer_fold" not in map_df:
        raise ValueError("No 'outer_fold' column after attach; check registry/seed alignment.")

    have_fold = map_df["outer_fold"].notna().to_numpy()
    if not np.any(have_fold):
        raise ValueError("No rows have a fold assignment for this seed.")

    folds = map_df.loc[have_fold, "outer_fold"].astype(int).to_numpy()
    idx_all = map_df.loc[have_fold, "row_idx"].astype(int).to_numpy()

    # Safety: ensure indices are valid positional indices for X
    if idx_all.min() < 0 or idx_all.max() >= n:
        raise IndexError(
            f"Row positions out-of-bounds for n={n}. min={idx_all.min()}, max={idx_all.max()}"
        )

    for fold_num in np.unique(folds):
        te_mask = (folds == fold_num)
        te_idx = idx_all[te_mask]
        tr_idx = idx_all[~te_mask]

        # Optional guards
        if te_idx.size == 0:  # pragma: no cover
            raise ValueError(f"Fold {fold_num} has empty test set.")
        if np.intersect1d(tr_idx, te_idx).size:  # pragma: no cover
            raise AssertionError("Train/Test overlap detected.")

        yield int(fold_num), tr_idx.astype(int), te_idx.astype(int)


# -------------------------
# Convenience / diagnostics
# -------------------------

def check_registry_coverage(
    donor_ids: pd.Series | np.ndarray,
    registry: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """Return a small report on which donors in `donor_ids` are covered by the registry for `seed`."""
    donors = pd.Series(donor_ids, copy=False).astype(str)
    reg_seed = registry.loc[registry["seed"] == int(seed), ["donor", "outer_fold"]]
    merged = donors.to_frame("donor").merge(reg_seed, on="donor", how="left")
    merged["in_registry"] = merged["outer_fold"].notna()
    return merged


__all__ = [
    "make_split_registry",
    "save_registry",
    "load_registry",
    "attach_registry_for_seed",
    "iter_outer_folds_from_registry",
    "check_registry_coverage",
]


# # =========================
# # Split Registry Helpers
# # =========================
# from pathlib import Path
# import numpy as np
# import pandas as pd
# from typing import Iterable, Iterator, Tuple, Dict, List, Optional
# from sklearn.model_selection import StratifiedKFold

# def make_split_registry(
#     donor_ids: pd.Series,
#     y: pd.Series,
#     seeds: Iterable[int],
#     n_splits: int = 5,
#     shuffle: bool = True,
# ) -> pd.DataFrame:
#     """
#     Build a long-form registry: one row per (seed, donor) with the donor's OUTER test fold index.
#     Train = all folds != outer_fold for that (seed, donor).
#     """
#     donors = pd.Series(donor_ids).astype(str).values
#     labels = pd.Series(y).astype(int).values
#     assert len(donors) == len(labels), "donor_ids and y must align"

#     rows = []
#     for seed in seeds:
#         skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
#         # Assign fold numbers to the test indices
#         for fold_idx, (_, te_idx) in enumerate(skf.split(donors, labels), start=1):
#             for idx in te_idx:
#                 rows.append((seed, donors[idx], fold_idx))
#     reg = pd.DataFrame(rows, columns=["seed", "donor", "outer_fold"])
#     return reg

# def save_registry(registry: pd.DataFrame, path: Path) -> None:
#     path.parent.mkdir(parents=True, exist_ok=True)
#     registry.to_csv(path, index=False)

# def load_registry(path: Path) -> pd.DataFrame:
#     return pd.read_csv(path, dtype={"seed": int, "donor": str, "outer_fold": int})

# def attach_registry_for_seed(
#     donor_ids: pd.Series,
#     registry: pd.DataFrame,
#     seed: int,
#     how: str = "inner"  # "inner" recommended: drop donors not present in this modality
# ) -> pd.DataFrame:
#     """
#     Return a DataFrame with columns: donor, outer_fold (for the chosen seed).
#     """
#     df = pd.DataFrame({"donor": pd.Series(donor_ids).astype(str).values})
#     reg_seed = registry.loc[registry["seed"] == seed, ["donor", "outer_fold"]]
#     out = df.merge(reg_seed, on="donor", how=how, validate="m:1")
#     return out

# def iter_outer_folds_from_registry(
#     X: pd.DataFrame,
#     y: pd.Series,
#     donor_ids: pd.Series,
#     registry: pd.DataFrame,
#     seed: int,
# ) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
#     """
#     Yield (fold_number, train_index, test_index) **within the provided modality**,
#     respecting the master registry for the given seed.
#     Index arrays are relative to X/y/ids (i.e., local row indices).
#     """
#     # Map local rows -> outer_fold via donor merge
#     map_df = attach_registry_for_seed(donor_ids, registry, seed, how="inner")
#     # Align to X/y order by merging back with an index
#     local = pd.DataFrame({"donor": pd.Series(donor_ids).astype(str).values})
#     local = local.merge(map_df, on="donor", how="left")
#     # Identify which rows have a fold assignment (i.e., exist in registry for this seed)
#     have_fold = local["outer_fold"].notna().values
#     if not np.all(have_fold):
#         # Rows without assignment are simply excluded from this seed’s run
#         pass

#     folds = local.loc[have_fold, "outer_fold"].astype(int).values
#     idx_all = np.where(have_fold)[0]

#     for fold_num in np.unique(folds):
#         te_mask = (folds == fold_num)
#         te_idx = idx_all[te_mask]
#         tr_idx = idx_all[~te_mask]
#         yield fold_num, tr_idx, te_idx
