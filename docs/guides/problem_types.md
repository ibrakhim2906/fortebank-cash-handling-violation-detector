# Guide: recipes by problem type

The starter code covers tabular binary classification and regression. For anything else, propose a plan first using these recipes, reusing `eval.py`-style frozen validation and `experiments.md` logging.

| Case | Baseline first | Main approach | Validation | Metric | Show the jury |
|---|---|---|---|---|---|
| Credit default / scoring | Logistic regression | LightGBM/CatBoost on aggregated client features | Time or group by client | ROC-AUC / Gini | Reasons per client, ₸ at threshold |
| Churn / attrition | Activity-drop rule | Same as above, with trend + recency features | Time | ROC-AUC / PR-AUC | Who to contact, retention ₸ |
| Fraud / anomaly (labels) | Rules | LightGBM, threshold at fixed alert budget | Time | PR-AUC, recall at fixed FPR | Fraud caught per 1,000 alerts |
| Fraud / anomaly (no labels) | Z-score rules | Isolation Forest + engineered features | Manual review of top-N | Precision@N on reviewed cases | Examples of caught anomalies |
| Forecasting (volumes, cash, demand) | Seasonal naive (same day last week) | LightGBM on lags, rolling means, calendar features | Last period(s) held out, rolling origin | MAE / MAPE / WAPE | Forecast vs actual chart, ₸ saved on buffers |
| Segmentation (no target) | RFM quantiles | Scaled features → KMeans (k by silhouette + business sense) | Stability across seeds | Interpretability | Persona per segment + action per segment |
| Cross-sell / next best offer | Popularity | LightGBM per product, or ranker | Time | Recall@k / MAP@k | Offer list for a sample client |
| Text (complaints, chats, reviews) | TF-IDF + logistic regression | Embeddings or LLM API (cache every call) | Stratified K-fold | Macro F1 | Routed examples, time saved |
| Credit limit / amount (regression) | Median by segment | LightGBM regression | Time or group | MAE / RMSE | Error in ₸, examples |

## General rules
- Start with the simplest baseline and log it. The jury wants to see improvement over something.
- LLM APIs: budget cost and rate limits, cache responses to disk, set temperature 0, keep a fallback.
- Whatever the problem, finish with: explanation of why, ₸ impact, one clear demo flow.
