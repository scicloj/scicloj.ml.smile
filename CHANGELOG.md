# ConstantChangeLog

# unreleased
- made predicion target datatype symetric to train
- fixes #450 unexpected "distinct" of float columns after categorical->number using :float64

# 7.5.0
- imported code for TMS<->smile dataframe conversions from tech.v3.libs.smile
- options as malli

# 7.4.4
- more complete hyperparameter for classification

# 7.4.3
updated to fastmath 3

# 7.4.2
- more OLS metrices in glance
- more tidy models functions

# 7.4.1
- added misisng files

# 7.4

- added function to easly retrieve datasets from Smile Github data folder
- upgraded deps
- added tidy,glance and augment for OLS
   
# v7.3

- allow classificaion of already numeric targets
- re-added some gridsearch options
- added loglikelihood calculatin for OLS
- added one-line linear regression

## v7.2
- use latest metamorph.ml

## v7.1.657
- use latest metamorph.ml

## v7.1.656
- adapted to tablecloth 7.x

## v7.0.650
- fixed pom.xml in release

## v7.0.642
- fixed arrity exception in `reduce-dimensions` transform
- allow configuration of ppmap grain size


## v7.0.632
  - fixed serialisation for  svm
  - added tfidf->dense
  - added :tf-map-handler-fn to support pruning of the terms for tfidf
  - metamorph'ed  bow->tfidf reuses tf-map from :fit in :transform
  - added 2 weighting schemes for tf calculation
  - added 3 weighting schemes for idf calculation
  - support :word-normalize-fn in count-vectorizer to configure tokenisation


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
