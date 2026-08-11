# GM hybrid-ownership predictor (Streamlit)

Interactive demo built on the strongest model from this project. See the main
[README](../README.md), [reference/CHALLENGES.md](../reference/CHALLENGES.md), and
`preprocessing/honest_household_model.ipynb` for the full build: how the original
96.3%-accuracy Random Forest turned out to be a data leakage artifact, how a demographics-only
retrain landed at a modest but honest 0.68 ROC-AUC, and how vehicle-fleet aggregates, household
composition, a brand-tier proxy, and person-level features pushed a tuned, calibrated XGBoost to
0.75 ROC-AUC, tested against a cross-validated baseline at every step.

## Files

- `streamlit_app.py` — the app
- `gm_hybrid_model.joblib` — `CalibratedClassifierCV` wrapping the tuned XGBoost classifier (sigmoid calibration)
- `model_meta.json` — full 28-feature list, the 13 exposed as app inputs, background defaults for the rest, value ranges, and the honest evaluation metrics the app displays
- `requirements.txt` — dependencies for local runs and Streamlit Cloud

## Run locally

```bash
cd app
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy to Streamlit Community Cloud

1. Push this repo (including `app/`) to GitHub if you haven't already.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in.
3. Click "New app", pick this repo and branch.
4. Set the **main file path** to `app/streamlit_app.py`.
5. Deploy. Streamlit Cloud installs from `app/requirements.txt` automatically.
6. Once live, copy the app URL into the portfolio's GM project card (`demo` field in
   `projects-section.tsx`) so the "Live Demo" link on the site points at it.

The model file is a calibrated XGBoost pipeline, not a deep model, about 1.7MB, so it's
committed directly to the repo rather than downloaded at runtime.
