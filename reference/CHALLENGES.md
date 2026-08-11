# Challenges

## The original 96.3% result was inflated by data leakage

`preprocessing/linear_not_neural.ipynb` trained Random Forest on 36 raw columns from the merged
vehicle/household file, not the five demographic features described in the original write-up.
Two of those columns, `VEHFUEL` and `VEHTYPE`, describe the specific vehicle's own fuel and body
type. Checking the raw data directly: every vehicle with `VEHFUEL` in {4, 5, 6} is a hybrid, 100%
of the time. Those codes mean hybrid, plug-in hybrid, and electric in the NHTS coding scheme, so
handing the model that column is close to handing it the answer. That's why Random Forest reached
96.3% while Logistic Regression and KNN stayed near the 90% baseline: it wasn't learning that
income or geography predict hybrid ownership, it was mostly reading the fuel-type code off the
vehicle record.

Retraining on only legitimate household- and person-level features, income, household size,
vehicle count, urban/rural status, urban area size, driver count, home ownership, life-cycle
stage, worker count, MSA size, and household member count, with no vehicle-specific columns,
gives a materially different and more honest picture: a class-weighted Logistic Regression reaches
0.68 ROC-AUC, clearly better than chance but nowhere near the original 96.3% accuracy /
0.99 precision headline. Accuracy on its own is a poor stand-in for this at 62.8% given how the
balanced weighting trades accuracy for the recall that actually matters on a rare class; ROC-AUC is
the more honest number to report here.

That demographics-only number was the starting point, not the final one.
`preprocessing/honest_household_model.ipynb` builds on it step by step (vehicle-fleet aggregates,
extra household-composition variables, hyperparameter tuning, a brand-tier proxy, person-level
features), each addition tested against a cross-validated baseline before being kept, landing at
0.75 ROC-AUC with a tuned, sigmoid-calibrated XGBoost. `app/` holds that final model, not the
0.68 baseline; the notebook is the record of every step, including the ones that didn't help
(SMOTE, trip-level data, respondent sex), see the takeaways table at the end of that notebook.

The takeaway isn't that GM-brand hybrid adoption is unpredictable, it's that this particular
survey's household demographics carry a modest signal (AUC 0.68, better than a coin flip, well
short of a reliable classifier) and that the original headline number measured something closer to
"can a model detect that a hybrid is a hybrid from its own fuel code" than "can household
demographics predict hybrid ownership."

## Severe class imbalance

Only 286 of 2,937 households in the modeling set (9.7%) own a hybrid or electric vehicle. A model
that always predicts "no hybrid" already scores 90.3% accuracy, which is higher than most models
this project trained once leakage is removed. Precision and recall on the minority class are the
numbers that actually say something here, not accuracy.

## The target ended up broader than the original brief

The brief GM gave Aggie Data Science asked specifically about GM-brand EV adoption (Buick,
Cadillac, Chevrolet, GMC) out of GM vehicle owners. The modeling target that was actually built,
`HAS_HYBRID`, flags a household if *any* vehicle it owns is a hybrid, regardless of make. The
`MAKE` column was loaded and checked for missing values but never used to filter down to
GM-brand owners, and it was left in as a raw feature in the original leaky model rather than used
to scope the target. Narrowing to GM-brand hybrid owners specifically would have shrunk an already
small positive class (286 households) to something likely too small to model reliably, that's the
probable reason for the broader framing, but it means the finished model answers "does this
household own a hybrid" rather than the more specific GM-brand question it was scoped to answer.

## Merging data collected at three different levels

NHTS ships household-, vehicle-, and person-level files. A household can have multiple vehicles
and multiple people, so `HAS_HYBRID` is computed per household (true if any vehicle qualifies)
and the dataset is grouped down to one row per household before modeling. That grouping is
necessary to get a clean household-level target, but it also discards vehicle-level detail, model
year, ownership length, annual mileage, that could plausibly matter for predicting adoption
(without also reintroducing the same leakage problem, since those fields describe the vehicle
that's already hybrid or not).

## A coarse, demographics-only feature set

The honest feature set is eleven household- and person-level variables, income, household size,
vehicle count, urban/rural classification, urban area size, driver count, home ownership,
life-cycle stage, worker count, MSA size, and household member count. Real-world EV/hybrid
adoption research usually points to purchase price, available incentives, and charging access as
major factors, none of which are in the NHTS public-use files used here. The 0.68 ROC-AUC should
be read as the ceiling on what these particular demographic variables can predict alone, not a
ceiling on what's predictable given richer data.
