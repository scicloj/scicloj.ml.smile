(ns scicloj.ml.smile.toydata-test
  (:require [scicloj.ml.smile.toydata :as sut]
            [tablecloth.api :as tc]
            [clojure.test :as t]))


(t/deftest json
  (t/is (= [7 10]
           (->
            (sut/get-smile-data "json/books2.json")
            (tc/shape)))))

  ;;
(t/deftest csv
  (t/is (= [569 32]
           (->
            (sut/get-smile-data "classification/breastcancer.csv")
            (tc/shape)))))

(t/deftest arff
  (t/is (= [20640 9]
           (->
            (sut/get-smile-data "weka/regression/cal_housing.arff")
            (tc/shape)))))

(comment
  (sut/get-smile-data
   "weka/weather.arff"))
