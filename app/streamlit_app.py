"""
GM hybrid-ownership predictor.

A small interactive demo built on the strongest model from this project: a tuned, calibrated
XGBoost classifier trained on household demographics, vehicle-fleet aggregates, brand-tier
signal, and person-level features, 0.75 ROC-AUC on 5-fold cross-validation. See
preprocessing/honest_household_model.ipynb for the full build, including the data leakage bug
in the original 96.3%-accuracy result and every feature-engineering step that closed part of
that gap honestly.
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
FEATURES = meta["features"]
DEFAULTS = meta["background_defaults"]

st.title("Household hybrid-ownership predictor")
st.caption(
    "A National Household Travel Survey model estimating the probability a household owns "
    "a hybrid or electric vehicle, from demographics, vehicle fleet, and household composition."
)

with st.expander("Why this app exists, and the honest numbers behind it", expanded=False):
    st.markdown(
        f"""
The Random Forest in the original project write-up reached **96.3% accuracy**. It turned out
to be inflated by data leakage: the training data included the vehicle's own fuel-type code,
and every vehicle coded as hybrid/plug-in/electric fuel is, unsurprisingly, a hybrid 100% of
the time. The model was mostly reading the answer off the vehicle record, not learning
anything about the household.

This app runs the model that was rebuilt from scratch after removing that leak, and then
strengthened step by step: household demographics alone got to 0.68 ROC-AUC, adding vehicle-fleet
aggregates (fleet age, mileage), household composition, a brand-tier proxy, and person-level
features pushed a tuned XGBoost to **{metrics['model_roc_auc_cv']:.2f} ROC-AUC** ({metrics['model_roc_auc_cv_std']:.2f}
std across 5 folds). Every one of those additions was tested against a cross-validated baseline
before being kept, several tested ideas (SMOTE rebalancing, trip-level data, respondent sex) made
things worse or did nothing and were left out, documented as negative results rather than hidden.

Raw XGBoost probabilities on an imbalanced target like this one are overconfident, a household the
model scores at "70% probability" doesn't actually own a hybrid 70% of the time. This app uses the
**calibrated** version (sigmoid/Platt scaling), which cut the Brier score from
{metrics['brier_score_uncalibrated']:.3f} to {metrics['brier_score_calibrated']:.3f} with no
change to the model's ranking ability. That's also why the predicted probability below usually
looks lower than you might expect: only {metrics['baseline_rate']*100:.1f}% of households in the
data actually own a hybrid, and a properly calibrated model reflects that.
        """
    )

st.divider()
st.subheader("Household profile")

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
CENSUS_R_LABELS = {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"}

col1, col2 = st.columns(2)

with col1:
    income = st.selectbox("Household income", list(INCOME_LABELS), format_func=lambda x: INCOME_LABELS[x], index=5)
    hhsize = st.slider("Household size (people)", 1, 8, 3)
    vehcnt = st.slider("Vehicles owned", 0, 6, 2)
    wrkcount = st.slider("Number of workers", 0, 4, 1)
    urban = st.selectbox("Urban or rural", [1, 0], format_func=lambda x: "Urban" if x == 1 else "Rural")
    homeown = st.selectbox("Home ownership", list(HOMEOWN_LABELS), format_func=lambda x: HOMEOWN_LABELS[x])
    census_r = st.selectbox("Region", list(CENSUS_R_LABELS), format_func=lambda x: CENSUS_R_LABELS[x], index=2)

with col2:
    lifcyc = st.selectbox("Life-cycle stage", list(LIFCYC_LABELS), format_func=lambda x: LIFCYC_LABELS[x], index=1)
    msasize = st.selectbox("Metro area size", list(MSASIZE_LABELS), format_func=lambda x: MSASIZE_LABELS[x])
    min_vehicle_age = st.slider("Newest vehicle's age (years)", 0, 30, 5)
    max_vehicle_age = st.slider("Oldest vehicle's age (years)", 0, 30, 11)
    avg_annual_miles = st.slider("Average annual miles per vehicle", 0, 30000, 7500, step=500)
    premium_brand = st.selectbox(
        "Owns a premium-brand vehicle?",
        [0, 1],
        format_func=lambda x: "Yes (Lincoln, Cadillac, Audi, BMW, Mercedes, Volvo, Acura, Infiniti, Lexus)" if x == 1 else "No",
    )

if max_vehicle_age < min_vehicle_age:
    max_vehicle_age = min_vehicle_age

row = dict(DEFAULTS)
row.update({
    "HHFAMINC_IMP": income,
    "HHSIZE": hhsize,
    "HHVEHCNT": vehcnt,
    "URBRUR_BIN": urban,
    "HOMEOWN": homeown,
    "WRKCOUNT": wrkcount,
    "LIF_CYC": lifcyc,
    "MSASIZE": msasize,
    "CENSUS_R": census_r,
    "min_vehicle_age": min_vehicle_age,
    "max_vehicle_age": max_vehicle_age,
    "avg_annual_miles": avg_annual_miles,
    "pct_premium_brand": premium_brand,
})
input_row = pd.DataFrame([row])[FEATURES]

with st.expander("Fields not shown above (held at typical household values)", expanded=False):
    st.caption(
        "This model uses 28 features in total. The 13 above are the most interpretable and "
        "highest-importance ones, exposed as inputs. The rest (household trip count, driver "
        "count, urban-area size, rail access, number of distinct vehicle types, commercial "
        "vehicle share, region/division grouping, presence of young children, average "
        "respondent age and commute distance, and household education level) are held at their "
        "dataset median or most common value, shown below, so the form stays usable without "
        "quietly ignoring 15 of the 28 features the model was actually trained on."
    )
    st.json({k: v for k, v in DEFAULTS.items() if k not in row})

if st.button("Predict", type="primary"):
    proba = model.predict_proba(input_row)[0, 1]
    st.metric("Predicted probability of hybrid/EV ownership", f"{proba:.1%}")
    st.progress(min(max(proba, 0.0), 1.0))
    ratio = proba / metrics["baseline_rate"] if metrics["baseline_rate"] else 0
    st.write(
        f"That's **{ratio:.1f}x** the average household's rate ({metrics['baseline_rate']*100:.1f}%). "
        "Hybrid ownership is a rare outcome in this data, so probabilities well under 50% can "
        "still represent a meaningfully above-average household."
    )

st.divider()
st.caption(
    "Data: 2022 National Household Travel Survey (NHTS). Model: XGBoost, tuned via "
    "RandomizedSearchCV, sigmoid-calibrated, 28 features across demographics, vehicle fleet, "
    "household composition, and person-level aggregates. Full build and every tested "
    "alternative in preprocessing/honest_household_model.ipynb."
)
