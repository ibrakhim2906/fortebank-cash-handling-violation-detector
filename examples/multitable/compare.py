"""Compare: application-only vs point-in-time history features vs naive (leaky) history features.

Run from the repo root:
  uv run python examples/multitable/make_data.py
  uv run python -m examples.multitable.compare
"""
from src import config as C

# configure for this example (the real case sets these in src/config.py)
C.ID_COL, C.TARGET, C.TIME_COL, C.GROUP_COL = "application_id", "target", "application_date", None
C.DROP_COLS = ["client_id"]

from src import models as M  # noqa: E402
from src.eval import cross_validate, make_folds  # noqa: E402
from src.features import to_model_frame  # noqa: E402
from examples.multitable.features_multitable import (  # noqa: E402
    _load, build_dataset, build_dataset_naive)


def run(label, train, test):
    train = train.sort_values(C.ID_COL).reset_index(drop=True)
    X, _ = to_model_frame(train, test)
    y = train[C.TARGET]
    res = cross_validate(M.fit_lgbm, X, y, make_folds(train))
    print(f"{label:38s} AUC={res['score']:.4f}  features={X.shape[1]}")
    return res["score"]


if __name__ == "__main__":
    tr, te, _, _ = _load()
    base = run("1. application table only", tr, te)
    honest = run("2. + point-in-time history (correct)", *build_dataset())
    naive = run("3. + ALL history (naive, leaky)", *build_dataset_naive())
    print()
    print(f"Correct history features add {honest - base:+.4f} AUC.")
    print(f"The naive version looks {naive - honest:+.4f} AUC better — that gap is pure leakage.")
    print("It would collapse on new clients, and a bank jury will ask about it. Never ship it.")
