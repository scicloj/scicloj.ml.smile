(ns scicloj.ml.smile.projections-test
  (:require [scicloj.ml.smile.projections :as projections]
            [clojure.test :refer [deftest is]]
            [scicloj.ml.smile.test-tools :as tt]
            [tablecloth.api :as tc]
            [scicloj.metamorph.core :as mm]
            [scicloj.metamorph.ml :as ml]
            [same.core :refer [ish?]]
            [scicloj.metamorph.ml.preprocessing :as preprocessing]))
            


(def data
  (tc/dataset {:f1 [1 5 1 5 8]
               :f2 [2 5 4 3 1]
               :f3 [3 6 2 2 2]
               :f4 [4 7 3 1 2]}))



(deftest train-pca-model
  (let [train-result
        (ml/train data  {:model-type :smile.projections
                         :target-dims 2
                         :algorithm :pca-cov
                         :cnames [:f1 :f2 :f3 :f4]})]
    (is  (every? pos?
                 (seq
                  (.getCumulativeVarianceProportion
                   (-> train-result :model-data :model)))))))

(deftest train-pca-model-2
  (let [train-result
        (ml/train data  {:model-type :smile.projections/pca-cov
                         :target-dims 2
                         :algorithm :pca-cov
                         :cnames [:f1 :f2 :f3 :f4]})]
    (is  (every? pos?
                 (seq
                  (.getCumulativeVarianceProportion
                   (-> train-result :model-data :model)))))))

(deftest reduce-dimensions-test
  (let [pipe-fn
        (mm/pipeline
         (preprocessing/std-scale [:f1 :f2 :f3 :f4] {})
         (projections/reduce-dimensions :pca-cov 2 [:f1 :f2 :f3 :f4 ] {}))
         
        fit-context
        (pipe-fn {:metamorph/data data
                  :metamorph/mode :fit})]
    (is (tt/seq-fuzzy=?  [ 0.01400330784019062
                          -2.556533994286818
                          -0.0514801918647123
                          1.014150018390943
                          1.5798608599203965]
                         (seq
                          (get-in fit-context [:metamorph/data "pca-cov-0"]))
                         0.0001))


    (is (tt/seq-fuzzy=?  [ 0.7559747649563955
                          -0.7804317748323724
                          1.2531347040524976
                          2.3880830993447353E-4
                          -1.228916502486455]
                         (seq
                          (get-in fit-context [:metamorph/data "pca-cov-1"]))
                         0.0001))

    (is (ish? [ 0.628948310202486
               0.8952704360795221
               0.9937421970301336
               1.0]
           (-> (vals fit-context)
               (nth 3)
               :fit-result :model
               (.getCumulativeVarianceProportion)
               seq)))))
