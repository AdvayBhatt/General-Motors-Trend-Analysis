# Challenges

## Severe class imbalance

Only 286 of 2,937 households in the modeling set (9.7%) own a hybrid or electric vehicle. A model
that always predicts "no hybrid" already scores 90.3% accuracy, which is higher than three of the
four models this project trained. Logistic Regression (90.0%), KNN (90.1%), and the neural
network (90.8%) all landed close to that trivial baseline; only Random Forest (96.3%) clearly beat
it. Accuracy alone is close to meaningless here, precision and recall on the minority class are
the numbers that actually say something, and Random Forest's own numbers show the real tension:
99% precision but only 62% recall. It rarely cries wolf, but it also misses more than a third of
actual hybrid-owning households.

## The target ended up broader than the original brief

The brief GM gave Aggie Data Science asked specifically about GM-brand EV adoption (Buick,
Cadillac, Chevrolet, GMC) out of GM vehicle owners. The modeling target that was actually built,
`HAS_HYBRID`, flags a household if *any* vehicle it owns is a hybrid, regardless of make. The
`MAKE` column was loaded and checked for missing values but never used to filter down to
GM-brand owners. Narrowing to GM-brand hybrid owners specifically would have shrunk an already
small positive class (286 households) to something likely too small to model reliably, that's the
probable reason for the broader framing, but it means the finished model answers "does this
household own a hybrid" rather than the more specific GM-brand question it was scoped to answer.

## Merging data collected at three different levels

NHTS ships household-, vehicle-, and person-level files. A household can have multiple vehicles
and multiple people, so `HAS_HYBRID` is computed per household (true if any vehicle qualifies)
and the dataset is grouped down to one row per household before modeling. That grouping is
necessary to get a clean household-level target, but it also discards vehicle-level detail, model
year, ownership length, annual mileage, that could plausibly matter for predicting adoption.

## A coarse feature set

The final feature set is household income, household size, vehicle count, urban/rural
classification, and urban area size, five variables. Real-world EV/hybrid adoption research
usually points to purchase price, available incentives, and charging access as major factors,
none of which are in the NHTS public-use files used here. The 96.3% Random Forest number should be
read as "predictable from demographic and geographic household characteristics alone," not as a
ceiling on what's predictable given richer data.
