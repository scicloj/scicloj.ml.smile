(ns scicloj.ml.smile.sparse-logreg-test
  (:require [clojure.test :refer [deftest is]]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.dataset.categorical :as ds-cat]
            [scicloj.ml.smile.discrete-nb :as nb]
            [scicloj.ml.smile.nlp :as nlp]
            [scicloj.metamorph.ml :as ml]
            [scicloj.metamorph.ml.loss :as loss]

            [scicloj.ml.smile.sparse-logreg]))

(defn get-reviews []
  (->
   (ds/->dataset "test/data/reviews.csv.gz" {:key-fn keyword :parser-fn :string})
   (ds/select-columns [:Text :Score])
   (ds/categorical->number [:Score])
   (nlp/count-vectorize :Text :bow {:text->bow-fn nlp/default-text->bow})
   (nb/bow->SparseArray :bow :sparse {:create-vocab-fn #(nlp/->vocabulary-top-n % 100)})
   (ds-mod/set-inference-target :Score)))



(deftest  accurate-sparse-logistic-regession
  (let [reviews (get-reviews)
        trained-model
        (ml/train reviews {:model-type :smile.classification/sparse-logistic-regression
                           :n-sparse-columns 100
                           :sparse-column :sparse})
        prediction (-> (ml/predict reviews trained-model)
                       (ds-cat/reverse-map-categorical-xforms))]


    (is (= 0.741
           (loss/classification-accuracy (:Score prediction)
                                         (:Score (ds-cat/reverse-map-categorical-xforms reviews)))))))


(deftest  accurate-disrete-naive-bayes
  (let [reviews (get-reviews)
        trained-model
        (ml/train reviews {:model-type :smile.classification/discrete-naive-bayes
                           :discrete-naive-bayes-model :multinomial
                           :p 100
                           :k 5
                           :sparse-column :sparse})
        prediction (-> (ml/predict reviews trained-model)
                       (ds-cat/reverse-map-categorical-xforms))]
        
    (is (= 0.629
           (loss/classification-accuracy (:Score prediction)
                                         (:Score (ds-cat/reverse-map-categorical-xforms reviews)))))))
