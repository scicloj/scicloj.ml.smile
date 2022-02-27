(ns scicloj.ml.smile.sparse-logreg
  (:require
   [tech.v3.datatype :as dt]
   [scicloj.metamorph.ml :as ml]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.modelling :as ds-mod]
   [scicloj.ml.smile.discrete-nb :as nb]
   [scicloj.ml.smile.nlp :as nlp]
   [tech.v3.datatype.errors :as errors])
   

  (:import [smile.classification SparseLogisticRegression]
           [smile.data SparseDataset]
           [smile.util SparseArray]))




(defn train [feature-ds target-ds options]
  ;; (def feature-ds feature-ds)
  ;; (def target-ds target-ds)
  ;; (def options options)
  "Training function of sparse logistic regression model.
   The column of name `(options :sparse-column)` of `feature-ds` needs to contain the text as SparseArrays
   over the vocabulary.
   Options:

   * `:sparse-column` : column name with contains the sparse data as seq of SparseArrays
   * `:n-sparse-columns`: Number of columns / dimensions of the sparse vectors

"
  (errors/when-not-error
   (ds-mod/inference-target-label-map target-ds)
   "In classification, the target column needs to be categorical and having been transformed to numeric.
See tech.v3.dataset/categorical->number.")
  (errors/when-not-error (:sparse-column options) ":sparse-column need to be given")
  (errors/when-not-error (:n-sparse-columns options) ":n-sparse-columns need to be given")

  (let [sparse-column (get feature-ds (:sparse-column options))
        _ (errors/when-not-error sparse-column (str  "Column not found: " (:sparse-column options)))
        train-array (into-array SparseArray sparse-column)
        train-dataset (SparseDataset/of (seq train-array) (options :n-sparse-columns))
        score (get target-ds (first (ds-mod/inference-target-column-names target-ds)))]
    (SparseLogisticRegression/fit train-dataset
                                  (dt/->int-array score)
                                  (get options :lambda 0.1)
                                  (get options :tolerance 1e-5)
                                  (get options :max-iterations 500))))
                                  
(defn predict [feature-ds
               thawed-model
               model]
  "Predict function for sparse logistic regression model."
  (nb/predict feature-ds thawed-model model))


(ml/define-model!
  :smile.classification/sparse-logistic-regression
  train
  predict
  {:options [{:name :lambda
               :type :float32
               :default 0.1}
             {:name :tolerance
              :type :float32
              :default 1e-5}
             {:name :max-iterations
              :type :int32
              :default 500}]})
              


(comment

  (defn get-reviews []
    (->
     (ds/->dataset "test/data/reviews.csv.gz" {:key-fn keyword :parser-fn {:Score :string}})
     (ds/select-columns [:Text :Score])
     (ds/categorical->number [:Score])
    ;; (ds/update-column :Score #(map dec %))
     (nlp/count-vectorize :Text :bow nlp/default-text->bow)
     (nb/bow->SparseArray :bow :bow-sparse 100)
     (ds-mod/set-inference-target :Score)))


  (def reviews (get-reviews))

  (def trained-model
    (ml/train reviews {:model-type :smile.classification/sparse-logistic-regression
                       :sparse-column :bow-sparse}))

  (ml/predict reviews trained-model))
