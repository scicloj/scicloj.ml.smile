(ns scicloj.ml.smile.maxent-test
  (:require [clojure.test :as t :refer [deftest is]]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as ds-mod]
            [scicloj.ml.smile.maxent :as maxent]
            [scicloj.ml.smile.nlp :as nlp]
            [scicloj.metamorph.ml :as ml]))


(defn get-reviews []
  (->
   (ds/->dataset "test/data/reviews.csv.gz" {:key-fn keyword :parser-fn :string})
   (ds/select-columns [:Text :Score])
   (ds/categorical->number [:Score])
   (nlp/count-vectorize :Text :bow)
   (maxent/bow->sparse-array :bow :bow-sparse {:create-vocab-fn #(nlp/->vocabulary-top-n % 1000)})
   (ds-mod/set-inference-target :Score)))

(deftest test-maxent-multinomial []
  (let [reviews (get-reviews)
        trained-model (ml/train reviews {:model-type :smile.classification/maxent-multinomial
                                         :sparse-column :bow-sparse
                                         :p 1000})]

    (is (= 1 (get (first (:bow reviews)) "sweet")))
    (is (= [120 244 457] (take 3 (-> reviews
                                     :bow-sparse
                                     first))))
    (is (= 1001 (-> trained-model
                    :model-data
                    .coefficients
                    first
                    count)))))



(defn get-binom-reviews []
  (->
   (ds/->dataset "test/data/reviews.csv.gz" {:key-fn keyword :parser-fn :string})
   (ds/select-columns [:Text :Score])
   (ds/filter-column :Score
                        (fn [score]
                          (or (= score "1")
                              (= score "2"))))
   (ds/categorical->number [:Score])
   (nlp/count-vectorize :Text :bow)
   (maxent/bow->sparse-array :bow :bow-sparse {:create-vocab-fn #(nlp/->vocabulary-top-n % 1000)})
   (ds-mod/set-inference-target :Score)))


(deftest test-maxent-binomial []
  (let [binom-reviews (get-binom-reviews)
                           
        trained-model (ml/train binom-reviews {:model-type :smile.classification/maxent-binomial
                                               :sparse-column :bow-sparse
                                               :p 1000})
        _ (def trained-model trained-model)
        prediction (ml/predict (ds/head binom-reviews) trained-model)]


    (is (= [0.0 1.0]) (-> prediction :Score frequencies keys))
    (is (= 1001 (-> trained-model
                    :model-data
                    .coefficients
                    count)))))
