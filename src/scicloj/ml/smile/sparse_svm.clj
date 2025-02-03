(ns scicloj.ml.smile.sparse-svm
  (:require
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.gridsearch :as gs]
   [scicloj.ml.smile.malli :as malli]
   [scicloj.ml.smile.registration :refer [class->smile-url]]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.modelling :as ds-mod]
   [tech.v3.datatype :as dt]
   [tech.v3.datatype.errors :as errors])
  (:import
   (smile.classification SVM)
   (smile.util SparseArray)))

(defn train
  "Training function of sparse SVM model.
   The column of name `(options :sparse-column)` of `feature-ds` needs to contain the text as SparseArrays
   over the vocabulary."
  [feature-ds target-ds options]
  (let [train-array (into-array SparseArray (get feature-ds (options :sparse-column)))
        score (get target-ds (first (ds-mod/inference-target-column-names target-ds)))
        p (:p options)
        _ (errors/when-not-error (and (not (nil? p)) (pos? p)) "p needs to be specified in options and greater 0")]
        
    (SVM/fit train-array
             (dt/->int-array score)
             p
             ^double (get options :C 1.0)
             ^double (get options :tol 1e-4))))


(defn- predict
  "Predict function for sparse SVM model"
  [feature-ds thawed-model model]
  (let [sparse-arrays (into-array ^SparseArray  (get feature-ds (get-in model [:options :sparse-column])))
        target-colum (first (:target-columns model))
        predictions (.predict (:model-data model) sparse-arrays)]
    (ds/->dataset {target-colum predictions})))

(def ^:private hyperparameters
  {:C (gs/linear 1 10)
   :tol (gs/categorical [1e-4 1e-3 1e-2 0.1])})

(ml/define-model!
  :smile.classification/sparse-svm
  train
  predict
  {:options
   (malli/options->malli
   [{:name :C
     :type :float32
     :default 1.0
     :description "soft margin penalty parameter"}
    {:name :tol
     :type :float32
     :default 1e-4
     :description "tolerance of convergence test"}
    {:name :sparse-column
     :type :keyword} 
    {:name :p
     :type :int32}
    ])

   :hyperparameters hyperparameters
   :documentation {:javadoc (class->smile-url SVM)
                   :user-guide "https://haifengl.github.io/classification.html#svm"}})
