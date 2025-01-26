(ns scicloj.ml.smile.svm-test
  (:require
   [tech.v3.dataset.math :as std-math]
   [tech.v3.dataset.column-filters :as cf]
   [tech.v3.dataset.modelling :as ds-mod]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.column :as ds-col]
   [scicloj.metamorph.ml :as ml]
   [scicloj.ml.smile.classification]
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
   "target"])
   

(defn train-split [ds options-map]
  (let [
        scaling (std-math/fit-std-scale (cf/feature ds))
        scaled-features (std-math/transform-std-scale (cf/feature ds) scaling)
        ds (ds/append-columns scaled-features (cf/target ds))
        split (ds-mod/train-test-split ds options-map)
        target-colname (first (ds/column-names (cf/target (:test-ds split))))
        fitted-model (ml/train (:train-ds split) options-map)
        predictions (ml/predict (:test-ds split) fitted-model)]
    predictions))

    
    

(deftest test-svm
  (let [src-ds (ds/->dataset "test/data/breast_cancer.csv.gz", {:header-row? false :n-initial-skip-rows 1})
        ds (->  src-ds
                (ds/rename-columns
                 (zipmap
                  (ds/column-names src-ds)
                  new-names))
                (ds/add-or-update-column
                 (ds-col/new-column "target"
                                    (map
                                     #(if  (= 0 %) 1 -1)
                                     (get src-ds "column-30"))))
                                
                (ds-mod/set-inference-target "target"))

        _ (MathEx/setSeed 1234)

        predictions-ds
        (train-split ds {:model-type :smile.classification/svm})
        predictions
        (get predictions-ds "target")

        pred-freqs
        (frequencies predictions)]

    (is (not (nil? (cf/prediction predictions-ds))))
    (is (= [-1 1] (keys pred-freqs)))))
