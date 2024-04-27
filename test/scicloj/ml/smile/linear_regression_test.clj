(ns scicloj.ml.smile.linear-regression-test
  (:require
   [clojure.test :as t :refer [deftest is]]
   [scicloj.metamorph.ml.toydata :as toydata]
   [scicloj.ml.smile.regression :as sut]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.modelling :as ds-mod]))

(deftest linear-regression

  (let [ds
        (->
         (toydata/mtcars-ds)
         (ds/drop-columns [:model])
         (ds-mod/set-inference-target :mpg))
        result (sut/linear-regression ds)]

    (is (= 181.29864126807638 (:bic result)))
    (is  (= 0.8690157644777647 (:r.squared result)))))
