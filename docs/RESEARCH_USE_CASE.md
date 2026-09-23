# Kinematic conditioning research workflow

Intended user: a computer-vision researcher testing whether motion-conditioned
synthetic frames preserve information beyond an unconditioned scene baseline.
Potential use: controlled synthetic-data experiments for gesture recognition.
This is not a deployed surgical simulator or validated surgical training product.

The useful output is a reproducible real/generated comparison with checkpoint,
seed, held-out trial identity, matched and mismatched conditioning metrics. An
attractive image alone does not demonstrate gesture accuracy or useful augmentation.

The grid-metrics tool now uses sample metadata, refuses empty/single-pair evaluations
and count mismatches, and labels unknown sample identities explicitly. Grid-crop
metrics are labeled exploratory because plotting and resizing affect scores.

Next pilot (not completed): train a simple gesture classifier with real data only,
then with synthetic augmentation. Hold out entire participants/trials in both arms;
report per-gesture performance, seeds, failure examples and temporal consistency.
Continue only if the independent held-out task improves, not merely pixel similarity.
No new model training or clinical benefit is claimed by this engineering update.
