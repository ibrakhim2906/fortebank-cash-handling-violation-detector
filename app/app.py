"""One application, its risk, its reasons and the value scenario."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import streamlit as st

from src import artifacts as A
from src import config as C
from src.explain import predict, reasons
from src.features import build_dataset

A.configure_run()
st.set_page_config(page_title="ForteBank datathon demo", layout="centered")


@st.cache_resource
def bundle(run_id):
    return A.load_bundle()


@st.cache_data
def data(run_id):
    train, test = build_dataset()
    rows = (test if test is not None else train).reset_index(drop=True)
    return rows, A.inputs(rows)


try:
    manifest = A.load_manifest()
    fitted = bundle(manifest["run_id"])
    rows, frame = data(manifest["run_id"])
except (ValueError, FileNotFoundError) as error:
    st.error(str(error))
    st.stop()

st.title("Client risk check")
if (C.REPORTS.parent / "synthetic.json").exists():
    st.caption("Synthetic rehearsal data · monetary values are illustrative scenarios")

probabilities = predict(fitted, frame)
threshold = manifest["policy"]["threshold"]
borderline = int(np.argmin(np.abs(probabilities - threshold))) if threshold is not None else len(rows) // 2
examples = {"Low risk": int(np.argmin(probabilities)), "High risk": int(np.argmax(probabilities)),
            "Borderline": borderline}
example = st.selectbox("Prepared example", list(examples))
ids = rows[C.ID_COL].tolist()
choice = st.selectbox(f"Choose {C.ID_COL}", ids, index=examples[example])
position = ids.index(choice)
row = frame.iloc[[position]]
prediction = float(predict(fitted, row)[0])

if fitted["task"] == "binary":
    current_threshold = st.slider("Review threshold", 0.0, 1.0, float(threshold), 0.01)
    left, right = st.columns(2)
    left.metric("Predicted probability", f"{prediction:.1%}")
    right.metric("Decision", "Flag for review" if prediction >= current_threshold else "OK")
else:
    st.metric("Prediction", f"{prediction:,.2f}")

st.subheader("Why")
st.dataframe(reasons(fitted, row), hide_index=True)
image_path = C.REPORTS / "impact.png"
if image_path.exists():
    st.subheader("Business impact")
    st.caption("Development value curve; the final holdout uses the policy frozen before evaluation.")
    st.image(str(image_path))
