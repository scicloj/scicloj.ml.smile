(ns scicloj.ml.smile.mlp-test
  (:require
   [clojure.test :refer [deftest is]]
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.loss :as loss]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.categorical :as ds-cat]
   [tech.v3.dataset.column-filters :as cf]
   [tech.v3.dataset.modelling :as ds-mod])
  (:import
   (smile.base.mlp
    ActivationFunction
    Cost
    HiddenLayerBuilder
    OutputFunction
    OutputLayerBuilder))
  )

 (deftest mlp
   (let [ hidden-layer-builder
         (HiddenLayerBuilder. 1 (ActivationFunction/linear))

         output-layer-builder
         (OutputLayerBuilder. 3  OutputFunction/LINEAR  Cost/MEAN_SQUARED_ERROR)

         src-ds (ds/->dataset "test/data/iris.csv")
         ds (->  src-ds
                 (ds/categorical->number cf/categorical)
                 (ds-mod/set-inference-target "species"))
         split-data (ds-mod/train-test-split ds {:seed 1234})
         train-ds (:train-ds split-data)
         test-ds (:test-ds split-data)
         model (ml/train train-ds {:model-type :smile.classification/mlp
                                   :layer-builders [hidden-layer-builder output-layer-builder]})
         prediction
         (-> (ml/predict test-ds model)
             (ds-cat/reverse-map-categorical-xforms))]

     (is (< 0.2
            (loss/classification-accuracy
             (-> test-ds ds-cat/reverse-map-categorical-xforms (get  "species"))
             (get prediction "species"))))))
