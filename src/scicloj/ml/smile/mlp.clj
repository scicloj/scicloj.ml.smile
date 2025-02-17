(ns scicloj.ml.smile.mlp
  (:require
   [scicloj.metamorph.ml :as ml] ;[scicloj.ml.smile.classification :as classification]
   [scicloj.ml.smile.malli :as malli]
   [scicloj.ml.smile.model :as model]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.modelling :as ds-mod]
   [tech.v3.datatype.errors :as errors]
   [tech.v3.datatype.protocols :as dtype-proto]
   [tech.v3.tensor :as dtt])
  (:import
   (smile.base.mlp
    LayerBuilder)
   (smile.classification Classifier MLP)
   [tech.v3.datatype ObjectReader]))

(defn- train
  "Training function of MLP model. "
  [feature-ds target-ds options]
  (let [mlp (MLP. (ds/column-count feature-ds)
                  (into-array LayerBuilder (:layer-builders options)))
        ;;
        target-column-names  (ds/column-names target-ds)
        _ (errors/when-not-error (= 1 (count target-column-names)) "Only one target column is supported.")
        target-colname (first target-column-names)

        train-data (into-array
                    (map
                     double-array
                     (ds/value-reader feature-ds)))
        y (int-array (seq (get target-ds (first (ds-mod/inference-target-column-names target-ds)))))]
    (.update mlp train-data y)

    {:predictor mlp
     :n-labels (-> target-ds (get target-colname)
                   vec  ;; see https://github.com/techascent/tech.ml.dataset/issues/450
                   distinct
                   count)}))



(defn double-array-predict-posterior
  [^Classifier model ds options n-labels]
  (let [value-reader (ds/value-reader ds)
        n-rows (ds/row-count ds)]
    (reify
      dtype-proto/PShape
      (shape [rdr] [n-rows n-labels])
      ObjectReader
      (lsize [rdr] n-rows)
      (readObject [rdr idx]
        (let [posterior (double-array n-labels)]
          (.predict model (double-array (value-reader idx)) posterior)
          (errors/when-not-error (not (some #(Double/isNaN %) posterior)) (str "Model prediction returned NaN. Options: " options))
          posterior)))))


(defn- predict
  "Predict function for MLP model"
  [feature-ds thawed-model {:keys [target-columns
                                   target-categorical-maps
                                   target-datatypes
                                   options]}]
  (errors/when-not-error target-categorical-maps "target-categorical-maps not found. Target column need to be categorical.")
  (let [
        target-colname (first target-columns)
        n-labels (-> (get target-categorical-maps target-colname)
                     :lookup-table
                     count)
        _ (errors/when-not-error (pos? n-labels) "n-labels equals 0. Something is wrong with the :lookup-table")

        predictions (double-array-predict-posterior
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
        (-> (ds-mod/probability-distributions->label-column finalised-predictions target-colname
                                                            (get target-datatypes target-colname))
            (ds/update-column target-colname
                              #(vary-meta % assoc :column-type :prediction)))]

    mapped-predictions))


(ml/define-model!
  :smile.classification/mlp
  train
  predict
  {:thaw-fn (fn
              [model-data]
              (:predictor model-data)
              )

   :options
   (malli/options->malli
   [{:name :layer-builders
     :type :seq
     :default []
     :description "Sequence of type smile.base.mlp.LayerBuilder describing the layers of the neural network "}])


   :documentation {:javadoc "https://haifengl.github.io/api/java/smile/classification/MLP.html"
                   :user-guide "https://haifengl.github.io/classification.html#neural-network"}})

                   


