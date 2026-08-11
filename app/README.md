# GM hybrid-ownership predictor (Streamlit)

Interactive demo built on the corrected, leakage-free model from this project. See the main
[README](../README.md) and [reference/CHALLENGES.md](../reference/CHALLENGES.md) for how the
original 96.3%-accuracy Random Forest turned out to be a data leakage artifact, and how this
model was retrained honestly on demographics alone (0.68 ROC-AUC).

## Files

- `streamlit_app.py` — the app
- `gm_hybrid_model.joblib` — the trained, class-weighted Logistic Regression pipeline (scaler + model)
- `model_meta.json` — feature list, value ranges, and the honest evaluation metrics the app displays
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

The model file is small (a Logistic Regression pipeline, not a deep model) so it's committed
directly to the repo rather than downloaded at runtime.
