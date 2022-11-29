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



(t/deftest float-target-should-crash
  (try
    (ml/train
     (-> (ds/->dataset {:col-1 [:a :a :b :b
                                :a :a :b :b]
                        :y [1.1 1 1 0
                            0 1 0 1]})
         (ds-mod/set-inference-target :y)
         (ds/categorical->number [:col-1]))
     {:model-type :smile.classification/decision-tree})
    (throw "failed")
    (catch Exception e (t/is true))))

(t/deftest rounded-float-target-should-crash
  (try
    (ml/train
     (-> (ds/->dataset {:col-1 [:a :a :b :b
                                :a :a :b :b]
                        :y [1.0 1 1 0
                            0 1 0 1]})
         (ds-mod/set-inference-target :y)
         (ds/categorical->number [:col-1]))
     {:model-type :smile.classification/decision-tree})
    (throw "failed")
    (catch Exception e (t/is true))))

(t/deftest keyword-target-should-crash
  (try

    (ml/train
     (-> (ds/->dataset {:col-1 [:a :a :b :b
                                :a :a :b :b]
                        :y [:a :a :b :b
                            :a :b :b :b]})
        (ds-mod/set-inference-target :y)
        (ds/categorical->number [:col-1]))
     {:model-type :smile.classification/decision-tree})


    (throw "failed")
    (catch Exception e (t/is true))))
