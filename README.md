[![Clojars Project](https://img.shields.io/clojars/v/org.scicloj/scicloj.ml.smile.svg)](https://clojars.org/org.scicloj/scicloj.ml.smile)
[![CI](https://github.com/scicloj/scicloj.ml.smile/actions/workflows/main.yml/badge.svg)](https://github.com/scicloj/scicloj.ml.smile/actions/workflows/main.yml)
[![cljdoc badge](https://cljdoc.org/badge/org.scicloj/scicloj.ml.smile)](https://cljdoc.org/d/org.scicloj/scicloj.ml.smile)


# scicloj.ml.smile

Smile models for [metamorph.ml](https://github.com/scicloj/metamorph.ml) 


This project depends on Smile 2.6.0  for licence reasons.
Smile 3.0.0 has changed the licence from LGPL to GPL and no decision was taken if to upgrade `scicloj.ml.smile` to Smile 3.0.0 and therefore change licence to GPL.

## Included models/algorithms

### Classificatiopn 

- :smile.classification/linear-discriminant-analysis
- :smile.classification/fld
- :smile.classification/random-forest
- :smile.classification/ada-boost
- :smile.classification/knn
- :smile.classification/decision-tree
- :smile.classification/gradient-tree-boost
- :smile.classification/regularized-discriminant-analysis
- :smile.classification/quadratic-discriminant-analysis
- :smile.classification/logistic-regression
- :smile.classification/svm
- :smile.classification/maxent-multinomial
- :smile.classification/maxent-binomial
- :smile.classification/mlp
- :smile.classification/discrete-naive-bayes
- :smile.classification/sparse-svm
- :smile.classification/sparse-logistic-regression

### Regression

-  :smile.regression/ordinary-least-square
-  :smile.regression/elastic-net
-  :smile.regression/lasso
-  :smile.regression/ridge
-  :smile.regression/gradient-tree-boost
-  :smile.regression/random-forest

### Clustering

- :fastmath.cluster/spectral
- :fastmath.cluster/dbscan
- :fastmath.cluster/k-means
- :fastmath.cluster/mec
- :fastmath.cluster/clarans
- :fastmath.cluster/g-means
- :fastmath.cluster/lloyd
- :fastmath.cluster/x-means
- :fastmath.cluster/deterministic-annealing
- :fastmath.cluster/denclue

### Projections
- :smile.projections/pca-cov
- :smile.projections/pca-cor
- :smile.projections/pca-prob
- :smile.projections/kpca
- :smile.projections/gha
- :smile.projections/random

### Manifolds

- :smile.manifold/isomap
- :smile.manifold/laplacian
- :smile.manifold/lle
- :smile.manifold/tsne
- :smile.manifold/umap

## License

Copyright © Scicloj

Distributed under the Eclipse Public License 2.0
