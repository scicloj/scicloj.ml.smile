(ns scicloj.ml.smile.mlp
  (:require
   [scicloj.metamorph.ml :as ml]
   [scicloj.ml.smile.classification :as classification]
   [scicloj.ml.smile.malli :as malli]
   [scicloj.ml.smile.model :as model]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.categorical :as ds-cat]
   [tech.v3.dataset.modelling :as ds-mod]
   [tech.v3.datatype.errors :as errors]
   [tech.v3.tensor :as dtt])
  (:import
   (smile.base.mlp
    ActivationFunction
    Cost
    HiddenLayerBuilder
    LayerBuilder
    OutputFunction
    OutputLayerBuilder)
   (smile.classification MLP)))

(defn- train
  "Training function of MLP model. "
  [feature-ds target-ds options]
  (let [mlp (MLP. (ds/column-count feature-ds)
                  (into-array LayerBuilder (:layer-builders options)))
        ;;

        train-data (into-array
                    (map
                     double-array
                     (ds/value-reader feature-ds)))
        y (int-array (seq (get target-ds (first (ds-mod/inference-target-column-names target-ds)))))]
    (.update mlp train-data y)
    mlp))



(defn- predict
  "Predict function for MLP model"
  [feature-ds thawed-model {:keys [target-columns
                                   target-categorical-maps
                                   options]}]
  (errors/when-not-error target-categorical-maps "target-categorical-maps not found. Target column need to be categorical.")
  (let [
        target-colname (first target-columns)
        n-labels (-> (get target-categorical-maps target-colname)
                     :lookup-table
                     count)
        _ (errors/when-not-error (pos? n-labels) "n-labels equals 0. Something is wrong with the :lookup-table")

        predictions (classification/double-array-predict-posterior
                     thawed-model
                     ;; (:model-data thawed-model)
                     feature-ds {} n-labels)
        finalised-predictions
        (-> predictions
            (dtt/->tensor)
            (model/finalize-classification (ds/row-count feature-ds)
                                           target-colname
                                           n-labels
                                           target-categorical-maps))
        mapped-predictions
        (-> (ds-mod/probability-distributions->label-column finalised-predictions target-colname)
            (ds/update-column target-colname
                              #(vary-meta % assoc :column-type :prediction)))]

    mapped-predictions))


(ml/define-model!
  :smile.classification/mlp
  train
  predict
  {:options
   (malli/options->malli
   [{:name :layer-builders
     :type :seq
     :default []
     :description "Sequence of type smile.base.mlp.LayerBuilder describing the layers of the neural network "}])


   :documentation {:javadoc "https://haifengl.github.io/api/java/smile/classification/MLP.html"
                   :user-guide "https://haifengl.github.io/classification.html#neural-network"}})

                   


(comment
  (do
    (require '[tech.v3.dataset.column-filters :as cf])
    (require '[tech.v3.dataset.modelling :as ds-mod])
    (require '[scicloj.metamorph.ml.loss :as loss])

    (def hidden-layer-builder
     (HiddenLayerBuilder. 1 (ActivationFunction/linear)))

   (def output-layer-builder
     (OutputLayerBuilder. 3  OutputFunction/LINEAR  Cost/MEAN_SQUARED_ERROR))

   (def src-ds (ds/->dataset "test/data/iris.csv"))
   (def ds (->  src-ds
                (ds/categorical->number cf/categorical)
                (ds-mod/set-inference-target "species")))
   (def feature-ds (cf/feature ds))
   (def split-data (ds-mod/train-test-split ds))
   (def train-ds (:train-ds split-data))
   (def test-ds (:test-ds split-data))
   (def model (ml/train train-ds {:model-type :smile.classification/mlp
                                  :layer-builders [hidden-layer-builder output-layer-builder]}))
   (def prediction
     (-> (ml/predict test-ds model)
         (ds-cat/reverse-map-categorical-xforms))))

  :ok)
