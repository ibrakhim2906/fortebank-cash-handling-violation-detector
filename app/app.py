"""Demo: pick a client → prediction → reasons → business impact.  Run: uv run streamlit run app/app.py"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src import config as C
from src.explain import load_bundle, predict, reasons
from src.features import build_dataset, to_model_frame

st.set_page_config(page_title="ForteBank datathon demo", layout="centered")


@st.cache_resource
def bundle():
    return load_bundle()


@st.cache_data
def data():
    train, test = build_dataset()
    train = train.reset_index(drop=True)
    rows = (test if test is not None else train).reset_index(drop=True)
    _, X = to_model_frame(train, rows)
    return rows, X


def threshold():
    path = C.REPORTS / "impact.json"
    return json.loads(path.read_text())["threshold"] if path.exists() else 0.5


rows, X = data()
b = bundle()
st.title("Client risk check")  # rename to the real use case

ids = rows[C.ID_COL].tolist() if C.ID_COL in rows.columns else list(range(len(rows)))
choice = st.selectbox(f"Choose {C.ID_COL}", ids[:500])
i = ids.index(choice)
x = X.iloc[[i]]
score = float(predict(b, x)[0])

if C.TASK == "binary":
    t = threshold()
    flagged = score >= t
    c1, c2 = st.columns(2)
    c1.metric("Predicted probability", f"{score:.1%}")
    c2.metric("Decision", "Flag for review" if flagged else "OK", f"threshold {t:.2f}", delta_color="off")
else:
    st.metric("Prediction", f"{score:,.2f}")

st.subheader("Why")
st.dataframe(reasons(b, x), hide_index=True)

img = C.REPORTS / "impact.png"
if img.exists():
    st.subheader("Business impact")
    st.image(str(img))
