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


(t/deftest test-iris
  (let [plain-iris
        (-> (datasets/iris-ds)
            (ds/update-column :species (fn [col]
                                         (map
                                          #(case %
                                             0 :a
                                             1 :b
                                             2 :c)

                                          col)))

            (dsmod/set-inference-target :species))


        species-map
        (dscat/fit-categorical-map plain-iris :species {} :int64)

        train-ds (-> plain-iris (ds/shuffle {:seed 123}) (ds/head 100))
        test-ds  (-> plain-iris (ds/shuffle {:seed 123}) (ds/head 100))




        model
        (-> train-ds
            (dscat/transform-categorical-map species-map)
            (ml/train {:model-type :smile.classification/logistic-regression}))


        _
        (-> train-ds
            (dscat/transform-categorical-map species-map))

        prediction
        (-> test-ds
            (dscat/transform-categorical-map species-map)
            (ml/predict model)
            (dscat/invert-categorical-map species-map))]

    (t/is (= {:b 39 :c 29 :a 32}
             (-> prediction :species frequencies))))) ;; =>


(t/deftest allow-numeric-target
  (let [iris-ds-traget-is-non-categorical
        (-> (datasets/iris-ds)
            (ds/assoc-metadata [:species]
                               :categorical-map nil
                               :categorical? nil))]
    ;; should not crash
    (ml/train
     iris-ds-traget-is-non-categorical
     {:model-type :smile.classification/logistic-regression})))


(t/deftest fail-on-string-target
  (let [iris-ds-traget-is-string
        (-> (datasets/iris-ds)
            (dscat/reverse-map-categorical-xforms))]

    ;; should not crash
    (t/is
     (thrown-with-msg? Exception #"All values in target need to be numbers."
                       (ml/train
                        iris-ds-traget-is-string
                        {:model-type :smile.classification/logistic-regression})))))
