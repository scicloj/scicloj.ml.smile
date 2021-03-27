(ns scicloj.ml.smile.discrete-nb-test
  (:require [clojure.test :refer :all]
            [scicloj.metamorph.ml :as ml]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as ds-mod]
            [scicloj.ml.smile.discrete-nb :as nb]
            [scicloj.ml.smile.nlp :as nlp]))



(defn get-reviews []
  (->
   (ds/->dataset "test/data/reviews.csv.gz" {:key-fn keyword :parser-fn :string})
   (ds/select-columns [:Text :Score])
   (ds/categorical->number [:Score])
   (nlp/count-vectorize :Text :bow {:text->bow-fn nlp/default-text->bow})
   (nb/bow->SparseArray :bow :sparse {:create-vocab-fn #(nlp/->vocabulary-top-n % 100)})
   (ds-mod/set-inference-target :Score)))



(deftest test-discrete-nb-bernoulli
  (is (= [1.000, 1.000, 1.000, 1.000, 1.000, 2.000, 2.000, 1.000, 1.000, 1.000]
       (:Score
        (let [reviews (get-reviews)
              trained-model
              (ml/train reviews {:model-type :smile.classification/discrete-naive-bayes
                                 :discrete-naive-bayes-model :bernoulli
                                 :sparse-column :sparse
                                 :p 100
                                 :k 5})
              prediction (ml/predict (ds/head reviews 10) trained-model)]
          prediction))
       )))

(deftest test-discrete-nb-multinomial
  (is (= [2.000, 2.000, 1.000, 0.000, 1.000, 2.000, 2.000, 2.000, 2.000, 2.000]
       (:Score
        (let [reviews (get-reviews)
              trained-model
              (ml/train reviews {:model-type :smile.classification/discrete-naive-bayes
                                 :discrete-naive-bayes-model :multinomial
                                 :sparse-column :sparse
                                 :p 100
                                 :k 5})
              prediction (ml/predict (ds/head reviews 10) trained-model)]
          prediction))
       )))
