# Assumptions

- **Any hybrid vehicle in a household is a reasonable stand-in for a GM-brand hybrid vehicle.**
  The original brief asked about GM-brand EV adoption specifically; the model predicts hybrid
  ownership of any make. This assumes the factors that predict hybrid ownership generally
  (income, household size, urban/rural status) transfer to the GM-brand-specific question, which
  was never directly tested since GM-brand hybrid owners weren't isolated as their own group.

- **A household-level target is the right level of analysis.** Collapsing person- and
  vehicle-level NHTS records to one row per household with a binary "has hybrid" flag assumes the
  household, not the individual vehicle or person, is the meaningful unit for predicting adoption.
  A household with five vehicles and one hybrid is treated identically to a household with one
  vehicle that's a hybrid.

- **The eleven demographic/geographic features used in the corrected model (income, household
  size, vehicle count, urban/rural status, urban area size, driver count, home ownership,
  life-cycle stage, worker count, MSA size, household member count) are the right ones to keep
  once vehicle-specific columns are removed for leakage.** Other NHTS variables (commute patterns,
  trip counts) were dropped for scope, this assumes they wouldn't meaningfully change the 0.68
  ROC-AUC result, which wasn't tested directly. See `reference/CHALLENGES.md` for why the original
  36-column feature set (including `VEHFUEL`/`VEHTYPE`) was replaced.

- **Class-weighted Logistic Regression, rather than Random Forest, is the fairer model to report
  once leakage is removed.** Balancing the loss trades overall accuracy for recall on the rare
  hybrid-owning class, which assumes that catching more true hybrid owners (at the cost of more
  false positives) is more useful for this project's purpose than optimizing raw accuracy against
  a 90%-majority baseline.

- **The 2022 NHTS sample is representative of the households GM would actually want to target.**
  NHTS is a national household travel survey, not a GM customer or prospect database, so this
  assumes the demographic patterns in a general household travel survey generalize to GM's actual
  buyer population.

- **A 90/10 train/test split with `random_state=42` gives a reliable read on model performance.**
  With only 286 positive examples total, the test set has roughly 57 hybrid-owning households.
  That's a small enough number that accuracy, precision, and recall on the minority class could
  shift meaningfully with a different random split, no cross-validation was run to check how
  stable these numbers are.
