(ns tech.v3.libs.smile.maxent
  (:require [pppmap.core :as ppp]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.libs.smile.nlp :as nlp]
            [scicloj.metamorph.ml :as ml]
            [tech.v3.datatype.errors :as errors]
            [scicloj.metamorph.ml.model :as model]

            [tech.v3.tensor :as dtt]
            )
  (:import smile.classification.Maxent))

(def maxent-default-parameters
  {
   :lambda 0.1
   :tol 1e-5
   :max-iter 500
   })




(defn bow->sparse-array [ds bow-col indices-col options]
  "Converts a bag-of-word column `bow-col` to sparse indices column `indices-col`,
   as needed by the Maxent model.
   `vocab size` is the size of vocabluary used, sorted by token frequency "
  (nlp/bow->something-sparse ds bow-col indices-col  nlp/bow->sparse-indices options))



(defn maxent-train [feature-ds target-ds options maxent-type]
    "Training function of Maxent model
   The column of name `(options :sparse-colum)` of `feature-ds` needs to contain the text as a sparce vector
   agains the vocabulary."
  (let [train-array (into-array ^"[[Ljava.lang.Integer"
                                (get feature-ds (:sparse-column options)))
        _ (def train-array train-array)
        train-score-array (into-array Integer/TYPE
                                      (get target-ds (first (ds-mod/inference-target-column-names target-ds))))
        p (int  (or  (:p options) 0) )
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

(defn maxent-train-multinomial [feature-ds target-ds options]
  "Training function of Maxent/multinomial model
   The column of name `(options :sparse-colum)` of `feature-ds` needs to contain the text as a sparse vector
   agains the vocabulary."
  (maxent-train feature-ds target-ds options :multinomial))


(defn maxent-train-binomial [feature-ds target-ds options]
  "Training function of Maxent/binomial model
   The column of name `(options :sparse-colum)` of `feature-ds` needs to contain the text as a sparse vector
   agains the vocabulary."
  (maxent-train feature-ds target-ds options :binomial))


(defn maxent-predict [feature-ds
                      thawed-model
                      model]

  "Predict function for Maxent"
  ;; (def feature-ds feature-ds)
  ;; (def model model)
  ;; (def thawed-model thawed-model)
  (let [predict-array
        (into-array ^"[[Ljava.lang.Integer"
                    (get feature-ds (get-in model [:options :sparse-column])))
        target-colum (first (:target-columns model))
        ;; _ (def predictions predictions)
        ;; _ (def predict-array predict-array)
        ;; _ (def target-colum target-colum)
        n-labels (-> model :target-categorical-maps target-colum :lookup-table count)
        _ (errors/when-not-error (pos-int? n-labels) (str  "No labels found for target column" target-colum ))

        posteriori
        (into-array
         (repeatedly (ds/row-count feature-ds)
                     #(double-array n-labels)))
        predictions (seq  (.predict (:model-data model) predict-array posteriori))
        ;; _ (def predictions predictions)
        ]
    (model/finalize-classification
     (dtt/->tensor posteriori)
     (ds/row-count feature-ds)
     target-colum
     (-> model :target-categorical-maps)
     )
    ))



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
