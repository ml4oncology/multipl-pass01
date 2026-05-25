import joblib
import pandas as pd


def load_data_tabpfn(task: str, modality: str, target: str):
    """
    Load processed COMPASS data for a given task, modality, and target.

    Parameters
    ----------
    task     : "prognosis" | "DTE-FFX" | "DTE-GNP"
    modality : "Clinical" | "DNA" | "RNA" | "Histopathology" | "EarlyFusion"
    target   : "orr" | "1yOS"

    Returns
    -------
    X   : pd.DataFrame  — feature matrix
    y   : pd.DataFrame  — target column
    ids : pd.DataFrame  — donor IDs
    """
    supported_tasks = {"prognosis", "DTE-FFX", "DTE-GNP"}
    if task not in supported_tasks:
        raise ValueError(f"Unknown task '{task}'. Must be one of: {supported_tasks}")

    base     = f"../../data/processed/COMPASS/{task}"
    X        = joblib.load(f"{base}/{modality}_{task}_X.pkl")
    y        = joblib.load(f"{base}/{modality}_{task}_y.pkl")
    y        = y[[target]]
    ids      = joblib.load(f"{base}/{modality}_{task}_ids.pkl")

    return X, y, ids