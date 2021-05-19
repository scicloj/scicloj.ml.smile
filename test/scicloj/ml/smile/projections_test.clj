(ns scicloj.ml.smile.projections-test
  (:require [scicloj.ml.smile.projections :as projections]
            [clojure.test :refer [deftest is]]
            [tablecloth.api :refer [dataset]]
            [tech.v3.dataset.column-filters :as cf]
            [scicloj.metamorph.core :as mm]
            [tech.v3.dataset.math :as math]
            [scicloj.metamorph.ml.preprocessing :as preprocessing]
            ))

(def data
  (dataset {:f1 [1 5 1 5 8]
            :f2 [2 5 4 3 1]
            :f3 [3 6 2 2 2]
            :f4 [4 7 3 1 2]
            }))


(deftest reduce-dimensions-test
  (let [pipe-fn
        (mm/pipeline
         (preprocessing/std-scale [:f1 :f2 :f3 :f4] {})
         (projections/reduce-dimensions :pca-cov 2 [:f1 :f2 :f3 :f4 ] {})
         )
        fit-context
        (pipe-fn {:metamorph/data data
                  :metamorph/mode :fit})]
    (def fit-context fit-context)
    (is (=  [ 0.014003307840190701
             -2.5565339942868195
             -0.05148019186471198
             1.0141500183909435
             1.579860859920397]
            (get-in fit-context [:metamorph/data "pca-cov-0"])))

    (is (=  [ 0.7559747649563955
             -0.7804317748323735
             1.2531347040524983
             2.3880830993486978E-4
             -1.228916502486455 ]
            (get-in fit-context [:metamorph/data "pca-cov-1"])))
    (is (= [ 0.628948310202486
            0.8952704360795221
            0.9937421970301336
            1.0]
           (-> (vals fit-context)
               (nth 3)
               :fit-result :model
               (.getCumulativeVarianceProportion)
               seq
               )
           )
        ))
  )
