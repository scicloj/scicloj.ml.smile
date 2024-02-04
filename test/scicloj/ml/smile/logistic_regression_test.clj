(ns scicloj.ml.smile.logistic-regression-test
  (:require  [clojure.test :as t]
             [scicloj.ml.smile.classification]
             [scicloj.metamorph.ml :as ml]
             [scicloj.metamorph.ml.loss :as loss]
             [tech.v3.dataset :as ds]
             [tech.v3.dataset.modelling :as dsmod]
             [tech.v3.dataset.column-filters :as cf]
             [tech.v3.dataset.categorical :as dscat]
             [scicloj.metamorph.ml.toydata :as datasets]))

(def iris (-> (datasets/iris-ds)
              (ds/update-column :species (fn [col]
                                           (map
                                            #(case %
                                               0 :a
                                               1 :b
                                               2 :c)

                                            col)))
              (ds/categorical->number [:species] [] :int)
              (dsmod/set-inference-target :species)))





(t/deftest train-predict
  (let [train-ds (-> iris (ds/shuffle) (ds/head 100))
        test-ds (-> iris (ds/shuffle) (ds/head 100))
        model (ml/train train-ds {:model-type :smile.classification/logistic-regression})
        prediction (ml/predict test-ds model)]

    ;; (ml/scorqe prediction test-ds :species loss/classification-accuracy [{:name :loss
    ;;                                                                       :metric-fn loss/classification-loss}])

    (t/is (= #{:a :b :c}
             (-> prediction
                 (cf/prediction)
                 (dscat/reverse-map-categorical-xforms)
                 :species
                 frequencies
                 keys
                 set)))))
