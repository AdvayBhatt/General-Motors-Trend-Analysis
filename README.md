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

`preprocessing/preprocessing.ipynb` handles the merge, feature selection (urban/rural status, household income, household size, vehicle count, urban area size), and target construction.

`preprocessing/linear_not_neural.ipynb` compares four models on the same train/test split:

| Model | Accuracy | Notes |
|---|---|---|
| Logistic Regression | 90.0% | Baseline |
| K-Nearest Neighbors (k=6) | 90.1% | Marginal gain over baseline |
| Random Forest | 96.3% | 99% precision, 62% recall on the hybrid-owner class |
| Neural Network (Keras) | 90.8% | Best validation accuracy across 1000 epochs |

Random Forest was the standout, but the precision/recall split matters. It is very confident when it predicts hybrid ownership, but still misses a meaningful share of actual hybrid-owning households, expected given hybrid ownership is a rare class in the data. `figures/model_comparison.png` charts all four models against the always-predict-majority-class baseline, which most of them barely beat.

`reference/variable_scoping_notes.txt` has the original variable-by-variable notes from scoping the project against the NHTS codebook. `reference/CHALLENGES.md` and `reference/ASSUMPTIONS.md` cover what didn't make it into the numbers above, including that the model predicts hybrid ownership generally rather than GM-brand ownership specifically, worth reading before taking the 96.3% at face value.

## Running it

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn scipy
jupyter notebook preprocessing/preprocessing.ipynb
```

## Status

Built as part of an Aggie Data Science project challenge sponsored by General Motors. Our team was not selected to present to GM directly, but the modeling work stands on its own.
