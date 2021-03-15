(ns tech.v3.libs.smile.svm
  (:require [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as ds-mod]
            [scicloj.metamorph.ml :as ml])
  (:import smile.classification.SVM))

(defn train [feature-ds target-ds options]
  "Training function of SVM model. "
  (let [train-data
        (into-array
         (map
          double-array
          (ds/value-reader feature-ds)))]
    (SVM/fit train-data
             (into-array Integer/TYPE (seq (get target-ds (first (ds-mod/inference-target-column-names target-ds)))))
             ^double (get options :C 1.0)
             ^double (get options :tol 1e-4))))

(defn predict [feature-ds
               thawed-model
               model]
  "Predict function for SVM model"
  (let [to-predict-data
        (into-array
         (map
          double-array
          (ds/value-reader feature-ds)))
        target-colum (first (:target-columns model))
        predictions (.predict (:model-data model) to-predict-data)]
    (ds/->dataset {target-colum predictions} )))


(ml/define-model!
  :smile.classification/svm
  train
  predict
  {})
