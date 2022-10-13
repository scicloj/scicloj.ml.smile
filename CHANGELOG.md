# ConstantChangeLog
## unrelease
  - fixed serialisation for  svm
  - added tfidf->dense
  - added :tf-map-handler-fn to support pruning of the terms for tfidf
  - metamorph'ed  bow->tfidf reuses tf-map from :fit in :transform
  - 2 weighting schees for tf calculation


## v6.2.585
- added LDA QDA RDA FLD]

## 6.1.578
- produce java 8 compatible java classes
- more docu

## 6.1.572
add 3 types of unsupervised models

- added clustering models
- added projection models
- added manifold models


## 6.00

- added Malli schemas to model options


## 5.07
- added clustering

## 5.06

- added projections
- added more model specific docu



## 3.00
 * Upgrade to smile 2.5.0.
 * Minimum workingtech.ml.dataset version is 4.00

## 2.0-beta-56

### **Breaking Change:** Smile 2.4.0 upgrade.
There are fewer smile regressors and classifiers supported.  XGBoost support is the
same.  This requires `[techascent/tech.ml.dataset "2.0-beta-56"]` or later.  If you
are using dataset, xgboost, or the set of supported smile regressors and classifiers
changes to your code should be zero or minimal.


## 2.0-beta-48
**Breaking Change:** This library expects tech.ml.dataset to be provided.  So your project
needs both this library and `[tech.ml.dataset "2.0-beta-49"]`.  This is to reduce the
number of spurious releases.
