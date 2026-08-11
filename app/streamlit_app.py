"""
GM hybrid-ownership predictor.

A small interactive demo built on the corrected, leakage-free model from this project.
The original Random Forest (96.3% accuracy) turned out to be reading the answer off the
vehicle's own fuel-type code. This app uses the retrained, honest model instead: household
demographics only, 0.68 ROC-AUC. See reference/CHALLENGES.md in the main repo for the full
writeup of what went wrong and how it was fixed.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "gm_hybrid_model.joblib"
META_PATH = APP_DIR / "model_meta.json"

st.set_page_config(page_title="GM Hybrid Ownership Predictor", page_icon="🚗", layout="centered")


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    return model, meta


model, meta = load_model()
metrics = meta["metrics"]

st.title("Household hybrid-ownership predictor")
st.caption(
    "A National Household Travel Survey model estimating the probability a household owns "
    "a hybrid or electric vehicle, from demographics and geography alone."
)

with st.expander("Why this app exists, and why it's honest about a mediocre result", expanded=False):
    st.markdown(
        f"""
The Random Forest in the original project write-up reached **96.3% accuracy**. It turned out
to be inflated by data leakage: the training data included the vehicle's own fuel-type code,
and every vehicle coded as hybrid/plug-in/electric fuel is, unsurprisingly, a hybrid 100% of
the time. The model was mostly reading the answer off the vehicle record, not learning
anything about the household.

This app runs the retrained, leakage-free model instead, demographics and geography only,
no vehicle-level fields. Its honest performance:

- **ROC-AUC: {metrics['model_roc_auc']:.2f}** (0.50 is chance, 1.00 is perfect)
- Accuracy: {metrics['model_accuracy']*100:.1f}% (the model is weighted toward catching hybrid
  owners rather than maximizing raw accuracy, since always guessing "no hybrid" already scores
  {metrics['baseline_accuracy']*100:.1f}%)
- Precision / recall on hybrid-owning households: {metrics['model_precision_class1']*100:.0f}% /
  {metrics['model_recall_class1']*100:.0f}%

A real signal, well short of a reliable classifier. Treat the number below as "how much this
household's profile resembles a hybrid-owning one," not a confident prediction.
        """
    )

st.divider()
st.subheader("Household profile")

col1, col2 = st.columns(2)

INCOME_LABELS = {
    1: "Less than $10,000", 2: "$10,000-$14,999", 3: "$15,000-$24,999",
    4: "$25,000-$34,999", 5: "$35,000-$49,999", 6: "$50,000-$74,999",
    7: "$75,000-$99,999", 8: "$100,000-$124,999", 9: "$125,000-$149,999",
    10: "$150,000-$199,999", 11: "$200,000 or more",
}
HOMEOWN_LABELS = {
    1: "Owned, with mortgage/loan", 2: "Owned, free and clear",
    3: "Rented", 4: "Occupied without payment",
}
LIFCYC_LABELS = {
    1: "One adult, no children", 2: "2+ adults, no children",
    3: "One adult, youngest child 0-5", 4: "2+ adults, youngest child 0-5",
    5: "One adult, youngest child 6-15", 6: "2+ adults, youngest child 6-15",
    7: "One adult, youngest child 16-21", 8: "2+ adults, youngest child 16-21",
    9: "One adult, retired, no children", 10: "2+ adults, retired, no children",
}
MSASIZE_LABELS = {
    1: "Metro area under 250,000", 2: "Metro area 250,000-499,999",
    3: "Metro area 500,000-999,999", 4: "Metro area 1-3 million",
    5: "Metro area 3 million+", 6: "Not in a metro area",
}
URBANSIZE_LABELS = {
    1: "Urban area 50,000-199,999", 2: "Urban area 200,000-499,999",
    3: "Urban area 500,000-999,999", 4: "Urban area 1M+ with heavy rail",
    5: "Urban area 1M+ without heavy rail", 6: "Not in an urbanized area",
}

with col1:
    income = st.selectbox("Household income", list(INCOME_LABELS), format_func=lambda x: INCOME_LABELS[x], index=5)
    hhsize = st.slider("Household size (people)", 1, 8, 3)
    vehcnt = st.slider("Vehicles owned", 0, 6, 2)
    drvrcnt = st.slider("Number of drivers", 0, 5, 2)
    wrkcount = st.slider("Number of workers", 0, 4, 1)

with col2:
    urban = st.selectbox("Urban or rural", [1, 0], format_func=lambda x: "Urban" if x == 1 else "Rural")
    homeown = st.selectbox("Home ownership", list(HOMEOWN_LABELS), format_func=lambda x: HOMEOWN_LABELS[x])
    lifcyc = st.selectbox("Life-cycle stage", list(LIFCYC_LABELS), format_func=lambda x: LIFCYC_LABELS[x], index=1)
    msasize = st.selectbox("Metro area size", list(MSASIZE_LABELS), format_func=lambda x: MSASIZE_LABELS[x])
    urbansize = st.selectbox("Urban area size", list(URBANSIZE_LABELS), format_func=lambda x: URBANSIZE_LABELS[x])

cnttdhh = st.slider("Household trips taken on the survey travel day", 0, 20, 4)

input_row = pd.DataFrame([{
    "HHFAMINC_IMP": income,
    "HHSIZE": hhsize,
    "HHVEHCNT": vehcnt,
    "URBRUR_BIN": urban,
    "URBANSIZE": urbansize,
    "DRVRCNT": drvrcnt,
    "HOMEOWN": homeown,
    "LIF_CYC": lifcyc,
    "WRKCOUNT": wrkcount,
    "MSASIZE": msasize,
    "CNTTDHH": cnttdhh,
}])[meta["features"]]

if st.button("Predict", type="primary"):
    proba = model.predict_proba(input_row)[0, 1]
    st.metric("Predicted probability of hybrid/EV ownership", f"{proba:.1%}")
    st.progress(min(max(proba, 0.0), 1.0))
    if proba > 0.5:
        st.write("Above the model's decision threshold, this profile leans toward hybrid ownership.")
    else:
        st.write("Below the model's decision threshold, this profile leans toward conventional-vehicle ownership.")
    st.caption(
        f"For reference, {metrics['baseline_accuracy']*100:.1f}% of households in the training "
        "data owned no hybrid at all, hybrid ownership is a rare outcome, so treat this as a "
        "relative signal rather than a confident yes/no."
    )

st.divider()
st.caption(
    "Data: 2022 National Household Travel Survey (NHTS). Model: class-weighted Logistic "
    "Regression on 11 household/person demographic and geographic features. Full writeup, "
    "including the leakage bug this app corrects for, in reference/CHALLENGES.md."
)
