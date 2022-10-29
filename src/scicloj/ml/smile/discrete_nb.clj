(ns scicloj.ml.smile.discrete-nb
  (:require
   [scicloj.metamorph.ml :as ml]
   [scicloj.ml.smile.model :as model]
   [scicloj.ml.smile.nlp :as nlp]
   [scicloj.ml.smile.registration :refer [class->smile-url]]
   [scicloj.ml.smile.utils :refer :all]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.modelling :as ds-mod]
   [tech.v3.datatype.errors :as errors]
   [tech.v3.tensor :as dtt])
  (:import
   (smile.classification DiscreteNaiveBayes DiscreteNaiveBayes$Model)
   (smile.util SparseArray)))

(defn bow->SparseArray
  "Converts a bag-of-word column `bow-col` to sparse indices column `indices-col`,
   as needed by the discrete naive bayes model. `vocab size` is the size of vocabluary used, sorted by token frequency "
  [ds bow-col indices-col options]
  (nlp/bow->something-sparse ds bow-col indices-col nlp/freqs->SparseArray options))


(def nb-lookup-table
  {
   :polyaurn DiscreteNaiveBayes$Model/POLYAURN
   :wcnb DiscreteNaiveBayes$Model/WCNB
   :cnb DiscreteNaiveBayes$Model/CNB
   :twcnb DiscreteNaiveBayes$Model/TWCNB
   :bernoulli DiscreteNaiveBayes$Model/BERNOULLI
   :multinomial DiscreteNaiveBayes$Model/MULTINOMIAL})

(defn train
  "Training function of discrete naive bayes model.
   The column of name `(options :sparse-colum)` of `feature-ds` needs to contain the text as SparseArrays
   over the vocabulary."
  [feature-ds target-ds options]
  (errors/when-not-error
   (ds-mod/inference-target-label-map target-ds)
   "In classification, the target column needs to be categorical and having been transformed to numeric.
See tech.v3.dataset/categorical->number.")

  (let [sparse-data-ds (get feature-ds (:sparse-column options))
        _ (errors/when-not-error sparse-data-ds "Column with sparse data need to be defined in option :sparse-column")
        train-array (into-array SparseArray sparse-data-ds)
        train-score-array (into-array Integer/TYPE
                                      (get target-ds (first (ds-mod/inference-target-column-names target-ds))))
        p (:p options)
        _ (when-not-pos-error p "p needs to be specified in options and greater 0")
        k (:k options)
        - (when-not-pos-error k "k needs to be specified in options and greater 0")
        nb-model
        (get nb-lookup-table (get options :discrete-naive-bayes-model :multinomial))
        _ (errors/when-not-error nb-model ":discrete-naive-bayes-model contains invalid model")
        nb (DiscreteNaiveBayes. nb-model (int (:k options)) (int  p))]
    (.update nb
             train-array
             train-score-array)
    nb))

(defn predict
  "Predict function for discrete naive bayes"
  [feature-ds thawed-model model]
  (let [
        sparse-col (get-in model [:options :sparse-column])

        sparse-arrays (get feature-ds  sparse-col)
        _ (errors/when-not-error sparse-arrays (str "Sparse arrays not found in column " sparse-col))
        target-colum (first (:target-columns model))
        n-labels (-> model :target-categorical-maps target-colum :lookup-table count)
        _ (errors/when-not-error (pos-int? n-labels) (str  "No labels found for target column" target-colum))
        posteriori (double-array n-labels)

        predictions (map
                     #(let [posteriori (double-array n-labels)
                            _ (.predict thawed-model % posteriori)]

                        posteriori)
                     sparse-arrays)
        finalised-predictions
        (model/finalize-classification
         (dtt/->tensor predictions)
         (ds/row-count feature-ds)
         target-colum
         (-> model :target-categorical-maps))

        mapped-predictions
        (-> (ds-mod/probability-distributions->label-column finalised-predictions target-colum)
            (ds/update-column target-colum
                              #(vary-meta % assoc :column-type :prediction)))]
        
    mapped-predictions))



(ml/define-model!
  :smile.classification/discrete-naive-bayes
  train
  predict
  {:options [{:name :p :type :int32 :default nil}
             {:name :k :type :int32 :default nil}
             {:name :discrete-naive-bayes-model
              :type :keyword
              :default nil
              :lookup-table nb-lookup-table}]
              
             
   :documentation {:javadoc (class->smile-url DiscreteNaiveBayes)
                   :user-guide "https://haifengl.github.io/nlp.html#naive-bayes"}})

   
