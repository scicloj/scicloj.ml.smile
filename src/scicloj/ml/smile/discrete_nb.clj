(ns scicloj.ml.smile.discrete-nb
  (:require [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as ds-mod]
            [scicloj.ml.smile.nlp :as nlp]
            [tech.v3.datatype.errors :as errors]
            [scicloj.metamorph.ml :as ml]
            [scicloj.metamorph.ml.model :as model]
            [tech.v3.tensor :as dtt]

            )
  (:import [smile.classification DiscreteNaiveBayes DiscreteNaiveBayes$Model]
           smile.util.SparseArray))



(defn bow->SparseArray [ds bow-col indices-col options]
  "Converts a bag-of-word column `bow-col` to sparse indices column `indices-col`,
   as needed by the discrete naive bayes model. `vocab size` is the size of vocabluary used, sorted by token frequency "
  (nlp/bow->something-sparse ds bow-col indices-col nlp/freqs->SparseArray options))






(defn train [feature-ds target-ds options]
  "Training function of discrete naive bayes model.
   The column of name `(options :sparse-colum)` of `feature-ds` needs to contain the text as SparseArrays
   over the vocabulary."
  (let [train-array (into-array SparseArray
                                (get feature-ds (:sparse-column options)))
        train-score-array (into-array Integer/TYPE
                                      (get target-ds (first (ds-mod/inference-target-column-names target-ds))))
        p (:p options)
        _ (errors/when-not-error (and (not (nil? p)) (pos? p)) "p needs to be specified in options and greater 0")
        nb-model
        (case (options :discrete-naive-bayes-model)
          :polyaurn DiscreteNaiveBayes$Model/POLYAURN
          :wcnb DiscreteNaiveBayes$Model/WCNB
          :cnb DiscreteNaiveBayes$Model/CNB
          :twcnb DiscreteNaiveBayes$Model/TWCNB
          :bernoulli  DiscreteNaiveBayes$Model/BERNOULLI
          :multinomial DiscreteNaiveBayes$Model/MULTINOMIAL)
        nb (DiscreteNaiveBayes. nb-model (int (:k options)) (int  p))]
    (.update nb
             train-array
             train-score-array)
    nb))

(defn predict [feature-ds
               thawed-model
               model]
  "Predict function for discrete naive bayes"
  (def model model)
  (def feature-ds feature-ds)
  (def thawed-model thawed-model)

  (let [
        sparse-col (get-in model [:options :sparse-column])
        sparse-arrays (get feature-ds  sparse-col)
        _ (errors/when-not-error sparse-arrays (str "Sparse arrays not found in column " sparse-col))
        _ (def sparse-arrays sparse-arrays)
        target-colum (first (:target-columns model))
        n-labels (-> model :target-categorical-maps target-colum :lookup-table count)
        _ (errors/when-not-error (pos-int? n-labels) (str  "No labels found for target column" target-colum ))
        _ (def n-labels n-labels)
        posteriori (double-array n-labels )

        predictions (map
                     #(let [posteriori (double-array n-labels )
                            _ (.predict thawed-model % posteriori)

                            ]
                        posteriori
                        )

                     sparse-arrays)]
    (def predictions predictions)
    (def target-colum target-colum)
    (model/finalize-classification
     (dtt/->tensor predictions)
     (ds/row-count feature-ds)
     target-colum
     (-> model :target-categorical-maps)
     )

    ))








(ml/define-model!
  :smile.classification/discrete-naive-bayes
  train
  predict
  {})
