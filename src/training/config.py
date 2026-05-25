from pathlib import Path
from datetime import date

# For reproducibility
RANDOM_STATES = [7270, 860, 5390, 5191, 5734, 6265, 466, 4426, 5578, 8322]


# ─── Paths ──────────────────────────────────────────────────────────────────────

# Base folder under which all data files live
BASE_DATA = Path("../../data")

# Base folder under which all date-stamped results live
BASE_RESULTS = Path("../../results")

# Auto-generate today’s date folder (YYYYMMDD)
DATE_STR = date.today().strftime("%Y%m%d")  # e.g. "20250728"