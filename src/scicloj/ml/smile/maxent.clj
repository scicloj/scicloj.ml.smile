(ns scicloj.ml.smile.maxent
  (:require
   [scicloj.metamorph.ml :as ml]
   [scicloj.ml.smile.model :as model]
   [scicloj.ml.smile.nlp :as nlp]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.modelling :as ds-mod]
   [tech.v3.datatype.errors :as errors]
   [tech.v3.tensor :as dtt])
  (:import
   (smile.classification Maxent)))

(def maxent-default-parameters
  {
   :lambda 0.1
   :tol 1e-5
   :max-iter 500})
   

(defn bow->sparse-array
  "Converts a bag-of-word column `bow-col` to sparse indices column `indices-col`,
   as needed by the Maxent model.
   `vocab size` is the size of vocabluary used, sorted by token frequency "
  [ds bow-col indices-col options]
  (nlp/bow->something-sparse ds bow-col indices-col  nlp/bow->sparse-indices options))





(defn- maxent-train
  "Training function of Maxent model
   The column of name `(options :sparse-colum)` of `feature-ds` needs to contain the text as a sparce vector
   agains the vocabulary."
  [feature-ds target-ds options maxent-type]
  (errors/when-not-error
          (ds-mod/inference-target-label-map target-ds)
          "In classification, the target column needs to be categorical and having been transformed to numeric.
See tech.v3.dataset/categorical->number.
")
           

  (let [train-array (into-array ^"[[Ljava.lang.Integer"
                                (get feature-ds (:sparse-column options)))
        train-score-array (into-array Integer/TYPE
                                      (get target-ds (first (ds-mod/inference-target-column-names target-ds))))
        p (int  (or  (:p options) 0))
        _ (errors/when-not-error (pos? p) "p needs to be specified in options and greater 0")
        options (merge maxent-default-parameters options)]

    (case maxent-type
      :multinomial
      (Maxent/multinomial
       p
       train-array
       train-score-array
       (:lambda options)
       (:tol options)
       (:max-iter options))
      :binomial
      (Maxent/binomial
       p
       train-array
       train-score-array
       (:lambda options)
       (:tol options)
       (:max-iter options)))))

(defn- maxent-train-multinomial
  "Training function of Maxent/multinomial model
   The column of name `(options :sparse-colum)` of `feature-ds` needs to contain the text as a sparse vector
   agains the vocabulary."
  [feature-ds target-ds options]
  (maxent-train feature-ds target-ds options :multinomial))


(defn- maxent-train-binomial
  "Training function of Maxent/binomial model
   The column of name `(options :sparse-colum)` of `feature-ds` needs to contain the text as a sparse vector
   agains the vocabulary."
  [feature-ds target-ds options]
  (maxent-train feature-ds target-ds options :binomial))


(defn- maxent-predict

  "Predict function for Maxent"
  [feature-ds thawed-model model]
  (let [predict-array
        (into-array ^"[[Ljava.lang.Integer"
                    (get feature-ds (get-in model [:options :sparse-column])))
        target-colum (first (:target-columns model))
        n-labels (-> model :target-categorical-maps target-colum :lookup-table count)
        _ (errors/when-not-error (pos-int? n-labels) (str  "No labels found for target column" target-colum))

        posteriori
        (into-array
         (repeatedly (ds/row-count feature-ds)
                     #(double-array n-labels)))
        _ (.predict (:model-data model) predict-array posteriori)

        finalised-predictions
        (model/finalize-classification
         (dtt/->tensor posteriori)
         (ds/row-count feature-ds)
         target-colum
         (-> model :target-categorical-maps))

        mapped-prediction
        (-> (ds-mod/probability-distributions->label-column finalised-predictions target-colum)
            (ds/update-column target-colum
                              #(vary-meta % assoc :column-type :prediction)))]
    mapped-prediction))

    
     
    



(ml/define-model!
  :smile.classification/maxent-multinomial
  maxent-train-multinomial
  maxent-predict
  {})

(ml/define-model!
  :smile.classification/maxent-binomial
  maxent-train-binomial
  maxent-predict
  {})
