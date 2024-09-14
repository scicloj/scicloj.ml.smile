(ns scicloj.ml.smile.categorical-test
  (:require
   [clojure.test :as t]
   [scicloj.metamorph.ml :as ml]
   [scicloj.ml.smile.classification]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.modelling :as ds-mod]))

(def working-ds [(-> (ds/->dataset {:col-1 [:a :a :b :b
                                            :a :a :b :b]
                                    :y [1 2 1 2
                                        2 1 2 1]})
                     (ds-mod/set-inference-target :y)
                     (ds/categorical->number [:col-1]))

                 (-> (ds/->dataset {:col-1 [:a :a :b :b
                                            :a :a :b :b]
                                    :y [1 1 1 2
                                        2 1 2 1]})
                     (ds-mod/set-inference-target :y)
                     (ds/categorical->number [:col-1]))

                 (-> (ds/->dataset {:col-1 [:a :a :b :b
                                            :a :a :b :b]
                                    :y [:a :a :b :b
                                        :a :b :b :b]})
                     (ds-mod/set-inference-target :y)
                     (ds/categorical->number [:col-1 :y]))])




(t/deftest should-not-crash
  (t/is (= 3
           (->
            (map
             #(ml/train % {:model-type :smile.classification/decision-tree})
             working-ds)
            count))))



(t/deftest float-target-should-not-crash
  (t/is (map?
         (ml/train
          
          (-> (ds/->dataset {:col-1 [:a :a :b :b
                                     :a :a :b :b]
                             :y [1.1 1 1 0
                                 0 1 0 1]})
              (ds-mod/set-inference-target :y)
              (ds/categorical->number [:col-1]))
          {:model-type :smile.classification/decision-tree}))))

(t/deftest rounded-float-target-should-not-crash
  (t/is (map?
         (ml/train
          (-> (ds/->dataset {:col-1 [:a :a :b :b
                                     :a :a :b :b]
                             :y [1.0 1 1 0
                                 0 1 0 1]})
              (ds-mod/set-inference-target :y)
              (ds/categorical->number [:col-1]))
          {:model-type :smile.classification/decision-tree}))))

(t/deftest keyword-target-should-crash
  (t/is (thrown? Exception

                 (ml/train
                  (-> (ds/->dataset {:col-1 [:a :a :b :b
                                             :a :a :b :b]
                                     :y [:a :a :b :b
                                         :a :b :b :b]})
                      (ds-mod/set-inference-target :y)
                      (ds/categorical->number [:col-1]))
                  {:model-type :smile.classification/decision-tree}))))

(t/deftest cat-label

  (let [model
        (-> (ds/->dataset {:col-1 [:a :a :b :b :a :a :b :b]
                           :label     [:x :y :x :y :x :y :x :y]})

            (ds/categorical->number [:col-1] [:a :b])
            (ds/categorical->number [:label] [:x :y])
            (ds-mod/set-inference-target :label)
            (ml/train {:model-type :smile.classification/decision-tree}))


        prediction
        (ml/predict
         (->
          (ds/->dataset {:col-1 [:a :a :b :b :a :a :b :b]})
          (ds/categorical->number [:col-1] [:a :b]))
         model)]

    (t/is (= (repeat 8 0.0)
             (-> prediction :label)))))


(t/deftest cat-label-numerc

  (let [model
        (-> (ds/->dataset {:col-1 [:a :a :b :b :a :a :b :b]
                           :label     [0 1 0 1 0 1 0 1]})


            (ds/categorical->number [:col-1] [:a :b])

            (ds-mod/set-inference-target :label)
            (ml/train {:model-type :smile.classification/decision-tree}))


        prediction
        (ml/predict
         (->
          (ds/->dataset {:col-1 [:a :a :b :b :a :a :b :b]})
          (ds/categorical->number [:col-1] [:a :b]))
         model)]
    (t/is (= (repeat 8 0.0)
             (-> prediction :label)))))
