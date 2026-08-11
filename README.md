# GM EV Adoption Prediction

A data challenge General Motors brought to Texas A&M's Aggie Data Science, predicting which households are likely to own a hybrid or electric vehicle, with a specific lens on GM-brand vehicles (Buick, Cadillac, Chevrolet, GMC), using the National Household Travel Survey (NHTS).

## Objective

Predict household-level hybrid/EV ownership (`HAS_HYBRID`) from household, person, and vehicle-level NHTS survey data, and identify which factors matter most.

## Data

Three 2022 NHTS public-use files, joined on `HOUSEID`:

- `data/hhpub.csv` — household-level records (income, size, vehicle count, urban/rural classification)
- `data/vehpub.csv` — vehicle-level records (make, fuel type, hybrid flag)
- `data/perpub.csv` — person-level records

Vehicle and household data are merged on `HOUSEID`, then grouped to one row per household with a binary target: whether any vehicle in the household is a hybrid.

`reference/` holds the public NHTS documentation used while building the feature set (data dictionary, codebook, user's guide, derived variables guide).

## Approach

`preprocessing/preprocessing.ipynb` handles the merge and target construction.
`preprocessing/linear_not_neural.ipynb` compares four models on the same train/test split, but
trains on 36 raw merged columns rather than a clean feature set:

| Model | Accuracy | Notes |
|---|---|---|
| Logistic Regression | 90.0% | Baseline |
| K-Nearest Neighbors (k=6) | 90.1% | Marginal gain over baseline |
| Random Forest | 96.3% | Inflated, see below |
| Neural Network (Keras) | 90.8% | Best validation accuracy across 1000 epochs |

That Random Forest number doesn't hold up. Two of the 36 columns, `VEHFUEL` and `VEHTYPE`,
describe the specific vehicle's own fuel and body type, and every vehicle with `VEHFUEL` in
{4, 5, 6} is a hybrid 100% of the time. The model was largely reading the answer off the vehicle
record rather than learning anything about the household. Retraining on only legitimate household-
and person-level demographic features (income, household size, vehicle count, urban/rural status,
urban area size, driver count, home ownership, life-cycle stage, worker count, MSA size, household
member count), with no vehicle-specific columns, gives a class-weighted Logistic Regression at
**0.68 ROC-AUC**, a real but modest signal, nowhere near the original headline. `figures/model_comparison.png`
and `app/` reflect this corrected result; `reference/CHALLENGES.md` walks through the leakage
finding in full, including the raw evidence.

`reference/variable_scoping_notes.txt` has the original variable-by-variable notes from scoping the project against the NHTS codebook. `reference/CHALLENGES.md` and `reference/ASSUMPTIONS.md` cover the rest of what didn't make it into the numbers above, including that the model predicts hybrid ownership generally rather than GM-brand ownership specifically.

## Try it

`app/streamlit_app.py` is a small Streamlit app built on the corrected, leakage-free model:
input household income, size, vehicle count, and a handful of other demographic fields, and it
returns a predicted hybrid-ownership probability plus the honest ROC-AUC/precision/recall context
behind that number. See `app/README.md` for how to run it locally or deploy it to Streamlit Cloud.

## Running it

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn scipy
jupyter notebook preprocessing/preprocessing.ipynb
```

## Status

Built as part of an Aggie Data Science project challenge sponsored by General Motors. Our team was not selected to present to GM directly, but the modeling work stands on its own.
