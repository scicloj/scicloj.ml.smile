(ns tech.v3.libs.smile.svm-test
  (:require [tech.v3.libs.smile.svm :as svm]
            [tech.v3.dataset.column-filters :as cf]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.dataset :as ds]
            [scicloj.metamorph.ml.loss :as loss]
            [scicloj.metamorph.ml :as ml]
            [clojure.test :refer [deftest is] :as t])
  (:import [smile.math MathEx]))



(def new-names
  ["mean radius"  "mean texture"
   "mean perimeter"  "mean area"
   "mean smoothness"  "mean compactness"
   "mean concavity"  "mean concave points"
   "mean symmetry"  "mean fractal dimension"
   "radius error"  "texture error"
   "perimeter error"  "area error"
   "smoothness error"  "compactness error"
   "concavity error"  "concave points error"
   "symmetry error"  "fractal dimension error"
   "worst radius"  "worst texture"
   "worst perimeter"  "worst area"
   "worst smoothness"  "worst compactness"
   "worst concavity"  "worst concave points"
   "worst symmetry"  "worst fractal dimension"
   "target"
   ])

(defn train-split [ds options-map]
  (let [
        _ (def ds ds)
        _ (def options-map options-map)
        split (ds-mod/train-test-split ds options-map)
        _ (def split split)
        target-colname (first (ds/column-names (cf/target (:test-ds split))))
        fitted-model (ml/train (:train-ds split) options-map)
        predictions (ml/predict (:test-ds split) fitted-model)]
    (loss/classification-loss ((:test-ds split) target-colname) (predictions target-colname))))

(def test-svn
  (let [src-ds (ds/->dataset "test/data/breast_cancer.csv.gz", {:header-row? false :n-initial-skip-rows 1 })
        ds (->  src-ds
                (ds/rename-columns
                 (zipmap
                  (ds/column-names src-ds)
                  new-names))
                (ds/add-or-update-column
                 (ds/new-column "target"
                                (map
                                 #(if  (= 0 %) 1 -1)
                                 (get src-ds "column-30"))
                                ))
                (ds-mod/set-inference-target "target"))

        _ (MathEx/setSeed 1234)
        loss
        (train-split ds {:model-type :smile.classification/svm
                         :randomize-dataset? false}
                     )]

    (is (= loss  0.13450292397660824 ))))
