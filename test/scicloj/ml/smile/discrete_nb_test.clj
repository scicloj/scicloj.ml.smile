(ns scicloj.ml.smile.discrete-nb-test
  (:require [clojure.test :refer [deftest is]]
            [scicloj.metamorph.ml :as ml]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.dataset.categorical :as ds-cat]
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
  (is (= ["4" "4" "4" "4" "4" "5" "5" "4" "4" "4"]
       (:Score
        (let [reviews (get-reviews)
              trained-model
              (ml/train reviews {:model-type :smile.classification/discrete-naive-bayes
                                 :discrete-naive-bayes-model :bernoulli
                                 :sparse-column :sparse
                                 :p 100
                                 :k 5})
              prediction (-> (ml/predict (ds/head reviews 10) trained-model)
                             (ds-cat/reverse-map-categorical-xforms))]
          prediction)))))
       


(deftest test-discrete-nb-multinomial
  (is (= ["5" "5" "4" "3" "4" "5" "5" "5" "5" "5"]
       (:Score
        (let [reviews (get-reviews)
              trained-model
              (ml/train reviews {:model-type :smile.classification/discrete-naive-bayes
                                 :discrete-naive-bayes-model :multinomial
                                 :sparse-column :sparse
                                 :p 100
                                 :k 5})
              prediction 
              (->
               (ml/predict (ds/head reviews 10) trained-model)
               (ds-cat/reverse-map-categorical-xforms))]
          prediction)))))
       





(deftest defaults []
  (is (map?
       (let [reviews (get-reviews)]
         (ml/train reviews {:model-type :smile.classification/discrete-naive-bayes
                            :sparse-column :sparse
                            :p 100
                            :k 5})))))


  