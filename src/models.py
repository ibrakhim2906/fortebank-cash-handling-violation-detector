"""Model zoo. Each fit_* has the signature fit_predict(X_tr, y_tr, X_va, y_va) -> predictions."""
import warnings
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config as C

warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")
warnings.filterwarnings("ignore", message="X does not have valid feature names")


def cat_cols(X):
    return [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]


def _predict(model, X):
    return model.predict_proba(X)[:, 1] if C.TASK == "binary" else model.predict(X)


def _as_str(X, cats):
    X = X.copy()
    for c in cats:
        X[c] = X[c].astype(str)
    return X


def _early_stop_split(X_tr, y_tr, frac=0.1):
    """Carve an early-stopping set out of the TRAINING fold, so the validation fold is never
    used to choose the number of trees (that would inflate scores).
    Time-based runs: the training fold arrives sorted by time, so the latest rows are used."""
    n = len(X_tr)
    if C.TIME_COL:
        k = max(1, int(n * frac))
        fit_idx, es_idx = np.arange(n - k), np.arange(n - k, n)
    else:
        fit_idx, es_idx = train_test_split(np.arange(n), test_size=frac, random_state=C.SEED,
                                           stratify=y_tr if C.TASK == "binary" else None)
    return X_tr.iloc[fit_idx], y_tr.iloc[fit_idx], X_tr.iloc[es_idx], y_tr.iloc[es_idx]


# --- floor ------------------------------------------------------------------
def fit_dummy(X_tr, y_tr, X_va, y_va):
    # Binary: one constant for every fold, so the pooled score is exactly chance (AUC 0.5).
    return np.full(len(X_va), 0.5 if C.TASK == "binary" else float(y_tr.mean()))


# --- linear -----------------------------------------------------------------
def fit_linear(X_tr, y_tr, X_va, y_va):
    cats = cat_cols(X_tr)
    nums = [c for c in X_tr.columns if c not in cats]
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), nums),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cats),
    ])
    est = LogisticRegression(max_iter=2000, random_state=C.SEED) if C.TASK == "binary" else Ridge()
    model = make_pipeline(pre, est)
    model.fit(_as_str(X_tr, cats), y_tr)
    return _predict(model, _as_str(X_va, cats))


# --- LightGBM ---------------------------------------------------------------
LGB_PARAMS = dict(n_estimators=5000, learning_rate=0.03, num_leaves=31, min_child_samples=50,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                  random_state=C.SEED, n_jobs=int(os.environ.get("DATATHON_THREADS", "-1")), verbose=-1)
LGB_BEST_ITERS = []


def make_lgbm(**overrides):
    params = {**LGB_PARAMS, "random_state": C.SEED, **overrides}
    return lgb.LGBMClassifier(**params) if C.TASK == "binary" else lgb.LGBMRegressor(**params)


def fit_lgbm(X_tr, y_tr, X_va, y_va):
    X_fit, y_fit, X_es, y_es = _early_stop_split(X_tr, y_tr)
    model = make_lgbm()
    model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)],
              callbacks=[lgb.early_stopping(200, verbose=False)])
    LGB_BEST_ITERS.append(model.best_iteration_ or LGB_PARAMS["n_estimators"])
    return _predict(model, X_va)


def fit_lgbm_full(X, y, n_iter):
    model = make_lgbm(n_estimators=max(50, int(n_iter * 1.1)))
    model.fit(X, y)
    return model


# --- CatBoost ---------------------------------------------------------------
def fit_cat(X_tr, y_tr, X_va, y_va):
    cats = cat_cols(X_tr)
    cls = CatBoostClassifier if C.TASK == "binary" else CatBoostRegressor
    model = cls(iterations=3000, learning_rate=0.05, depth=6, random_seed=C.SEED,
                od_type="Iter", od_wait=200, verbose=0, allow_writing_files=False,
                thread_count=int(os.environ.get("DATATHON_THREADS", "-1")))
    X_fit, y_fit, X_es, y_es = _early_stop_split(X_tr, y_tr)
    model.fit(_as_str(X_fit, cats), y_fit, cat_features=cats,
              eval_set=(_as_str(X_es, cats), y_es), use_best_model=True)
    return _predict(model, _as_str(X_va, cats))


FIT = {"dummy": fit_dummy, "linear": fit_linear, "lgbm": fit_lgbm, "cat": fit_cat}
