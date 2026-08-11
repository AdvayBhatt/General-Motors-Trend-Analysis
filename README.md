# GM EV Adoption Prediction

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-tuned%20%2B%20calibrated-006400)
![scikit--learn](https://img.shields.io/badge/scikit--learn-pipeline-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)

A data challenge General Motors brought to Texas A&M's Aggie Data Science, predicting which households are likely to own a hybrid or electric vehicle, with a specific lens on GM-brand vehicles (Buick, Cadillac, Chevrolet, GMC), using the National Household Travel Survey (NHTS).

**[Try the live demo →](https://general-motors-trend-analysis-mubdfc3jyaom8mdzybxjga.streamlit.app/)**

## Results at a glance

| | |
|---|---|
| Final model | XGBoost, tuned (RandomizedSearchCV) + sigmoid-calibrated |
| Cross-validated ROC-AUC | **0.75** (up from a 0.68 demographics-only baseline) |
| Original headline number | 96.3% accuracy, Random Forest, **found to be data leakage** and corrected |
| Features | 28, spanning demographics, vehicle fleet, household composition, and brand tier |
| What didn't work, tested and reported anyway | SMOTE rebalancing, trip-level aggregates, respondent sex |

The short version: the original 96.3% Random Forest was reading the answer off a column that
encodes the vehicle's own fuel type. Once that leak is removed, real demographic signal exists
but is modest (0.68 ROC-AUC), and it took real feature engineering, tested against a
cross-validated baseline at every step, to responsibly get that up to 0.75.

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
record rather than learning anything about the household.

`preprocessing/honest_household_model.ipynb` is the real rebuild: household-level deduplication
done safely (verified column-by-column, not assumed), then a demographics-only baseline
(**0.68 ROC-AUC**, class-weighted Logistic Regression), then eight further rounds of honestly
tested feature engineering. Each round is compared against a 5-fold cross-validated baseline
before being kept:

| Addition | CV ROC-AUC | Kept? |
|---|---|---|
| Demographics only (income, size, urban/rural, etc.) | 0.68 | baseline |
| + vehicle-fleet aggregates (fleet age, mileage, vehicle-type diversity) | 0.71 | yes |
| + household composition (adults, kids, rail access, region) | 0.72 | yes |
| + tuned XGBoost hyperparameters (RandomizedSearchCV) | 0.73 | yes |
| + brand-tier proxy and person-level features (age, education, commute) | **0.75** | yes, biggest single jump |
| SMOTE rebalancing | 0.69 | no, hurt XGBoost |
| Trip-level aggregates, respondent sex | 0.75 | no, no improvement |
| Probability calibration (sigmoid) | 0.75 AUC, Brier 0.184 → 0.074 | yes, same ranking, honest probabilities |

`app/` runs the final 0.75-AUC calibrated model, not the 0.68 baseline. `figures/model_comparison.png`
and `reference/CHALLENGES.md` cover the original leakage finding in full, including the raw
evidence; `reference/variable_scoping_notes.txt` has the original variable-by-variable notes from
scoping the project against the NHTS codebook. `reference/ASSUMPTIONS.md` covers what's still out
of scope, including that the model predicts hybrid ownership generally rather than GM-brand
ownership specifically.

## Try it

**[Live demo →](https://general-motors-trend-analysis-mubdfc3jyaom8mdzybxjga.streamlit.app/)**

`app/streamlit_app.py` runs the final tuned, calibrated XGBoost model: input household income,
vehicle fleet age, region, brand tier, and a handful of other fields, and it returns a calibrated
hybrid-ownership probability, shown against the dataset's real base rate rather than as a bare
number. See `app/README.md` for how to run it locally or deploy it to Streamlit Cloud.

## Running it

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn tensorflow matplotlib seaborn scipy
jupyter notebook preprocessing/honest_household_model.ipynb
```

## Status

Built as part of an Aggie Data Science project challenge sponsored by General Motors. Our team was not selected to present to GM directly, but the modeling work stands on its own.
