(ns scicloj.ml.smile.ols-test
  (:require
   [clojure.test :refer (deftest is)]
   [same.compare :as compare]
   [same.core :refer [ish? with-comparator]]
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.toydata]
   [scicloj.ml.smile.regression]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.modelling :as ds-mod]))

(def interest-rate  [2.75 2.5 2.5 2.5 2.5 2.5 2.5 2.25 2.25 2.25 2 2 2 1.75 1.75 1.75 1.75 1.75 1.75 1.75 1.75 1.75 1.75 1.75])
(def unemployment-rate [5.3 5.3 5.3 5.3 5.4 5.6 5.5 5.5 5.5 5.6 5.7 5.9 6 5.9 5.8 6.1 6.2 6.1 6.1 6.1 5.9 6.2 6.2 6.1])
(def stock-index-price [1464 1394 1357 1293 1256 1254 1234 1195 1159 1167 1130 1075 1047 965 943 958 971 949 884 866 876 822 704 719])



(def ds (-> (ds/->dataset {:interest-rate interest-rate
                              :unemployment-rate unemployment-rate
                              :stoc-index-price stock-index-price})
            (ds-mod/set-inference-target :stoc-index-price)))

(deftest explain
  (let [ols
        (ml/train ds {:model-type :smile.regression/ordinary-least-square})

        prediction
        (ml/predict ds ols)

        loglik (ml/loglik ols (:stoc-index-price ds) (:stoc-index-price prediction))
        ;; =>
        ols-model
        (ml/thaw-model ols
                       (ml/options->model-def (:options ols)))

        weights (ml/explain ols)]

    (is (ish? [

               {:adj.r.squared 0.8878844074567379
                :mse 4356.611357123131
                :rss 104558.67257095512
                :p.value 4.042532223561903E-11
                :df 21, :aic 277.2158453335483
                :bic 281.9280606549401
                :rmse 66.00463133086292
                :r.squared 0.8976335894170215
                :log-lik -134.60792266677416}]

              (ds/rows
               (ml/glance ols))))


    (with-comparator (compare/compare-ulp 100.0 1000)
      (is (ish? -134.60792266677416 loglik))
      (is (ish? 1798.4039776258396  (:bias weights)))
      (is (ish?  345.54008701056785   (-> weights :coefficients first second)))
      (is (ish? -250.14657136937868  (-> weights :coefficients second second))))))

(deftest tidy-fns

  (let [ds (scicloj.metamorph.ml.toydata/iris-ds)
        ols
        (->
         ds
         (ds/drop-columns [:species])
         (ds-mod/set-inference-target :sepal-length)
         (ml/train {:model-type :smile.regression/ordinary-least-square}))]


    (is (= [5 4]
           (ds/shape
            (ml/tidy ols))))
    (is (= [10 1]
           (ds/shape
            (ml/glance ols))))
    (is (= [7 150]
           (ds/shape
            (ml/augment ols ds))))))


(deftest fail-on-wrong-params
  (is (thrown? Exception (ml/train ds {:model-type :smile.regression/ordinary-least-square
                                                      :blub false}))))
