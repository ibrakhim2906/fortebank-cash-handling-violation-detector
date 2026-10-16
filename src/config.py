"""Project configuration. Set these right after the case is revealed (PLAYBOOK P2)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

# --- data -------------------------------------------------------------------
TRAIN_FILE = DATA_RAW / "train.csv"
TEST_FILE = DATA_RAW / "test.csv"   # set to None if there is no test file

ID_COL = "client_id"     # one row per prediction unit is identified by this
TARGET = "target"
TIME_COL = "application_date"  # None if no time dimension → validation becomes stratified / grouped
GROUP_COL = None         # e.g. "client_id" if one client has several rows
DROP_COLS = []           # columns that must never be features (leaky, IDs, free text)

# --- task -------------------------------------------------------------------
TASK = "binary"          # "binary" or "regression"
METRIC = "roc_auc"       # roc_auc | pr_auc | gini | f1 | rmse | mae
N_FOLDS = 5
SEED = 42

# --- business assumptions (binary tasks) -------------------------------------
# Agree these with ForteBank mentors; they are shown on the impact slide.
VALUE_TP = 400_000       # ₸ gained by correctly flagging a positive (e.g. loss avoided on a default)
COST_FP = 60_000         # ₸ lost by flagging a negative (e.g. margin lost by rejecting a good client)
