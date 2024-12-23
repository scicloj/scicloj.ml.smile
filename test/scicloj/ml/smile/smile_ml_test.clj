(ns scicloj.ml.smile.smile-ml-test
  (:require [scicloj.metamorph.ml.verify :as verify]
            [scicloj.metamorph.ml :as ml]
            [scicloj.ml.smile.regression]
            [scicloj.ml.smile.classification]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.dataset.utils :as ds-utils]
            [tech.v3.dataset.column-filters :as cf]
            [scicloj.ml.smile.malli :as malli]
            [clojure.test :refer [deftest is]]
            [scicloj.metamorph.ml.gridsearch :as ml-gs]))


;;shut that shit up.
(ds-utils/set-slf4j-log-level :warn)


(def smile-regression-models
  (->> (ml/model-definition-names)
       (filter #(= "smile.regression" (namespace %)))))


(deftest smile-regression-test
  (doseq [reg-model smile-regression-models]
    (verify/basic-regression {:model-type reg-model})))


(def smile-classification-models
  (->> (ml/model-definition-names)
       (filter #(= "smile.classification" (namespace %)))
       (remove #{:smile.classification/mlp
                 :smile.classification/maxent-binomial
                 :smile.classification/maxent-multinomial
                 :smile.classification/sparse-svm
                 :smile.classification/svm
                 :smile.classification/discrete-naive-bayes
                 :smile.classification/sparse-logistic-regression})))

(defn- one-gs-option [model-type]
  (let [options       
        (->>
         (ml/hyperparameters model-type)
         (ml-gs/sobol-gridsearch)
         (take 1)
         first)]
    (assoc options
           :model-type model-type)))



(deftest smile-classification-hyperparameters-test
  (doseq [classify-model smile-classification-models]
    ;(println :classify-model classify-model)
    (verify/basic-classification (one-gs-option classify-model))))


(deftest smile-classification-test
  (doseq [classify-model smile-classification-models]
    (verify/basic-classification {:model-type classify-model})))


(defn ->malli [[k def]]
  (let [
        option (:options def)]
    (malli/options->malli option)))

(deftest can-convert-options-to-malli
  (is (sequential?
       (mapv
        ->malli
        @ml/model-definitions*))))





(deftest test-fail-invalid-params
  (let [titanic (-> (ds/->dataset "test/data/titanic.csv")
                    (ds/categorical->number tech.v3.dataset.column-filters/categorical)
                    (ds-mod/set-inference-target "Survived"))]


    (is (thrown? IllegalArgumentException
                 (ml/train titanic {:model-type :smile.classification/random-forest
                                    :not-existing 123})))))



(deftest test-valid-params
  (let [titanic (-> (ds/->dataset "test/data/titanic.csv")
                    (ds/categorical->number tech.v3.dataset.column-filters/categorical)
                    (ds-mod/set-inference-target "Survived"))]

    (is map?
        (ml/train titanic {:model-type :smile.classification/random-forest
                           :trees 10}))))
