(ns scicloj.ml.smile.categorical-systematic-test
  (:require
   [clojure.test :refer [deftest is]]
   [cheshire.core :as json]
   [scicloj.ml.smile.classification]
   [scicloj.metamorph.ml :as ml]

   [tech.v3.dataset :as ds]
   [tech.v3.dataset.modelling :as ds-mod]
   [tech.v3.dataset.categorical :as ds-cat]))

(def raw-iris
  (->>
   (slurp "https://raw.githubusercontent.com/mljs/dataset-iris/refs/heads/master/src/data/iris.json")
   (json/decode)
   (map
    #(zipmap (range) %))
   ds/->dataset))

(def iris
  (-> raw-iris
      (ds/rename-columns {4 :species})
      (ds-mod/set-inference-target :species)))

(defn- validate-cat-maps-handling [model-type result-datatype]
  (let [model
        (ml/train
         (ds/categorical->number iris [:species] {} result-datatype)
         {:model-type model-type})
        prediction
        (->
         (ml/predict iris model)
         (ds-cat/reverse-map-categorical-xforms))]
    (is (= "setosa" (-> prediction :species first)))))

(def models 
  (->> 
   (ml/model-definition-names)
   (filter #(= "smile.classification" (namespace %)))
   (remove #{:smile.classification/mlp
             :smile.classification/maxent-binomial
             :smile.classification/maxent-multinomial
             :smile.classification/sparse-svm
             :smile.classification/svm
             :smile.classification/discrete-naive-bayes
             :smile.classification/sparse-logistic-regression})))

(def combinations
  (for [data-type [:int :float64 :float32]
        model-type models]
    [model-type data-type]))

(deftest assert-cat-maps-roundtrip
  (run!
   #(validate-cat-maps-handling (first %) (second %))
   combinations))


;; fails correctly, or not ?
(deftest numer-in-targt-throws 
  (is (thrown? Exception
       (ml/train iris {:model-type :smile.classification/logistic-regression})
       ))
  )



