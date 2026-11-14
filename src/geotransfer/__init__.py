"""Analysis code for the GeoSpOT urban-transfer experiment.

Modules
-------
decomposition  variance components of a seeds x sources x targets transfer array,
               outcome definitions (relative drop, within-target score, ...),
               split-half reliability and noise ceilings
inference      Spearman over off-diagonal cells, QAP permutation tests, region bootstrap
selection      decision value of a distance: picking a source for each target
ot_distance    GeoSpOT-style OT distances between regions (debiased Sinkhorn divergence)
"""
