(ns scicloj.ml.smile.classification
  "Namespace to require to enable a set of smile classification models."
  (:require [tech.v3.datatype :as dtype]
            [tech.v3.datatype.protocols :as dtype-proto]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.dataset.column-filters :as ds-cf]

            [tech.v3.dataset.utils :as ds-utils]
            [tech.v3.tensor :as dtt]
            [scicloj.metamorph.ml.model :as model]
            [scicloj.metamorph.ml.gridsearch :as ml-gs]
            [scicloj.metamorph.ml :as ml]
            [scicloj.ml.smile.protocols :as smile-proto]
            [tech.v3.libs.smile.data :as smile-data]
            [tech.v3.datatype.errors :as errors]
            [scicloj.ml.smile.discrete-nb]
            [scicloj.ml.smile.maxent]
            [scicloj.ml.smile.sparse-logreg]
            [scicloj.ml.smile.sparse-svm]
            [scicloj.ml.smile.svm]
            [clojure.string :as str]
            [scicloj.ml.smile.malli :as malli]
            [scicloj.ml.smile.registration :refer [class->smile-url]]
            [scicloj.ml.smile.model-examples :as examples]



            [malli.util :as mu])
  (:import [smile.classification SoftClassifier AdaBoost LogisticRegression
            DecisionTree RandomForest KNN GradientTreeBoost]
           [smile.base.cart SplitRule]
           [smile.data.formula Formula]
           [smile.data DataFrame]
           [java.util Properties List]
           [tech.v3.datatype ObjectReader]))


(set! *warn-on-reflection* true)

(defn- tuple-predict-posterior
  [^SoftClassifier model ds options n-labels]
  (let [df (smile-data/dataset->smile-dataframe ds)
        n-rows (ds/row-count ds)]
    (smile-proto/initialize-model-formula! model ds)
    (reify
      dtype-proto/PShape
      (shape [rdr] [n-rows n-labels])
      ObjectReader
      (lsize [rdr] n-rows)
      (readObject [rdr idx]
        (let [posterior (double-array n-labels)]
          (.predict model (.get df idx) posterior)
          posterior)))))


(defn- double-array-predict-posterior
  [^SoftClassifier model ds options n-labels]
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
          posterior)))))


(defn construct-knn [^Formula formula ^DataFrame data-frame ^Properties props]
  (KNN/fit (.toArray (.matrix  formula data-frame false))
           (.toIntArray  (.y formula data-frame))
           (Integer/parseInt (.getProperty props "smile.knn.k" "3"))))
           


(def split-rule-lookup-table
  {:gini SplitRule/GINI
   :entropy SplitRule/ENTROPY
   :classification-error  SplitRule/CLASSIFICATION_ERROR})

(def ^:private classifier-metadata
  {:ada-boost
   {:class AdaBoost
    :name :ada-boost
    :documentation {:user-guide "https://haifengl.github.io/classification.html#adaboost"}
    :options [{:name :trees
               :description "Number of trees"
               :type :int32
               :default 500}
              {:name :max-depth
               :description "Maximum depth of the tree"
               :type :int32
               :default 200}
              {:name :max-nodes
               :description "Maximum number of leaf nodes in the tree"
               :type :int32
               :default 6}
              {:name :node-size
               :description "Number of instances in a node below which the tree will not split, setting nodeSize = 5 generally gives good results"
               :type :int32
               :default 1}]
    :gridsearch-options {:trees (ml-gs/linear 2 50 10 :int64)
                         :max-nodes (ml-gs/linear 4 1000 20 :int64)}
    :property-name-stem "smile.adaboost"
    :constructor #(AdaBoost/fit ^Formula %1 ^DataFrame %2 ^Properties %3)
    :predictor tuple-predict-posterior}

   :logistic-regression
   {:class LogisticRegression
    :documentation {:user-guide "https://haifengl.github.io/classification.html#logit"}
    :name :logistic-regression
    :options [{:name :lambda
               :type :float64
               :default 0.1
               :description "lambda > 0 gives a regularized estimate of linear weights which often has superior generalization performance, especially when the dimensionality is high"}
              {:name :tolerance
               :type :float64
               :default 1e-5
               :description "tolerance for stopping iterations"}
              {:name :max-iterations
               :type :int32
               :default 500
               :description "maximum number of iterations"}]
    :gridsearch-options {:lambda (ml-gs/linear 1e-3 1e2 30)
                         :tolerance (ml-gs/linear 1e-9 1e-1 20)
                         :max-iterations (ml-gs/linear 1e2 1e4 20 :int64)}
    :property-name-stem "smile.logistic"
    :constructor #(LogisticRegression/fit ^Formula %1 ^DataFrame %2 ^Properties %3)
    :predictor double-array-predict-posterior}

   :decision-tree
   {:class DecisionTree
    :documentation {:user-guide "https://haifengl.github.io/classification.html#cart"}
    :name :decision-tree
    :options [{:name :max-nodes
               :type :int32
               :default 100
               :description "maximum number of leaf nodes in the tree"}
              {:name :node-size
               :type :int32
               :default 1
               :description "minimum size of leaf nodes"}
              {:name :max-depth
               :type :int32 
               :default 20
               :description "maximum depth of the tree"}
              {:name :split-rule
               :type :keyword
               :lookup-table split-rule-lookup-table
               :default :gini
               :description "the splitting rule"}]
    :gridsearch-options {:max-nodes (ml-gs/linear 10 1000 30)
                         :node-size (ml-gs/linear 1 20 20)
                         :max-depth (ml-gs/linear 1 50 20)
                         :split-rule (ml-gs/categorical [:gini :entropy :classification-error])}

                         
    :property-name-stem "smile.cart"
    :constructor #(DecisionTree/fit ^Formula %1 ^DataFrame %2  ^Properties %3)
    :predictor tuple-predict-posterior}

    

   ;; :fld {:attributes #{:projection}
   ;;       :class-name "FLD"
   ;;       :datatypes #{:float64-array}
   ;;       :name :fld
   ;;       :options [{:name :L
   ;;                  :type :int32
   ;;                  :default -1}
   ;;                 {:name :tolerance
   ;;                  :type :float64
   ;;                  :default 1e-4}]}
   :gradient-tree-boost
   {:class GradientTreeBoost
    :class-name "GradientTreeBoost"
    :documentation {:user-guide "https://haifengl.github.io/classification.html#gbm"}
    :name :gradient-tree-boost
    :options [{:name :ntrees
               :type :int32
               :default 500
               :description "number of iterations (trees)"}
              {:name :max-depth
               :type :int32
               :default 20
               :description "maximum depth of the tree"}
              {:name :max-nodes
               :type :int32
               :default 6
               :description "maximum number of leaf nodes in the tree"}
              {:name :node-size
               :type :int32
               :default 5
               :description "number of instances in a node below which the tree will not split, setting nodeSize = 5 generally gives good results"}
              {:name :shrinkage
               :type :float64
               :default 0.05
               :description "the shrinkage parameter in (0, 1] controls the learning rate of procedure"}
              {:name :sampling-rate
               :type :float64
               :default 0.7
               :description "the sampling fraction for stochastic tree boosting"}]
    :property-name-stem "smile.gbt"
    :constructor #(GradientTreeBoost/fit ^Formula %1 ^DataFrame %2  ^Properties %3)
    :predictor tuple-predict-posterior}

   :knn {:class KNN
         :name :knn
         :documentation {
                         :user-guide "https://haifengl.github.io/classification.html#knn"
                         :code-example (:knn examples/model-examples)}

                         
         :options [{:name :k
                    :type :int32
                    :default 3
                    :description "number of neighbors for decision"}]
                   
         :constructor #(construct-knn ^Formula %1 ^DataFrame %2  ^Properties %3)
         :predictor double-array-predict-posterior
         :property-name-stem "smile.knn"
         :gridsearch-options {:k (ml-gs/categorical [2 100])}}

   ;; :naive-bayes {:attributes #{:online :probabilities}
   ;;               :class-name "NaiveBayes"
   ;;               :datatypes #{:float64-array :sparse}
   ;;               :name :naive-bayes
   ;;               :options [{:name :model
   ;;                          :type :enumeration
   ;;                          :class-type NaiveBayes$Model
   ;;                          :lookup-table {
   ;;                                         ;; Users have to provide probabilities for this to work.
   ;;                                         ;; :general NaiveBayes$Model/GENERAL

   ;;                                         :multinomial NaiveBayes$Model/MULTINOMIAL
   ;;                                         :bernoulli NaiveBayes$Model/BERNOULLI
   ;;                                         :polyaurn NaiveBayes$Model/POLYAURN}
   ;;                          :default :multinomial}
   ;;                         {:name :num-classes
   ;;                          :type :int32
   ;;                          :default utils/options->num-classes}
   ;;                         {:name :input-dimensionality
   ;;                          :type :int32
   ;;                          :default utils/options->feature-ecount}
   ;;                         {:name :sigma
   ;;                          :type :float64
   ;;                          :default 1.0}]
   ;;               :gridsearch-options {:model (ml-gs/nominative [:multinomial :bernoulli :polyaurn])
   ;;                                    :sigma (ml-gs/exp [1e-4 0.2])}}
   ;; :neural-network {:attributes #{:online :probabilities}
   ;;                  :class-name "NeuralNetwork"
   ;;                  :datatypes #{:float64-array}
   ;;                  :name :neural-network}
   ;; :platt-scaling {:attributes #{}
   ;;                 :class-name "PlattScaling"
   ;;                 :datatypes #{:double}
   ;;                 :name :platt-scaling}


   ;; ;;Lots of discriminant analysis
   ;; :linear-discriminant-analysis
   ;; {:attributes #{:probabilities}
   ;;  :class-name "LDA"
   ;;  :datatypes #{:float64-array}
   ;;  :name :lda
   ;;  :options [{:name :prioiri
   ;;             :type :float64-array
   ;;             :default nil}
   ;;            {:name :tolerance
   ;;             :default 1e-4
   ;;             :type :float64}]
   ;;  :gridsearch-options {:tolerance (ml-gs/linear [1e-9 1e-2])}}


   ;; :quadratic-discriminant-analysis
   ;; {:attributes #{:probabilities}
   ;;  :class-name "QDA"
   ;;  :datatypes #{:float64-array}
   ;;  :name :qda
   ;;  :options [{:name :prioiri
   ;;             :type :float64-array
   ;;             :default nil}
   ;;            {:name :tolerance
   ;;             :default 1e-4
   ;;             :type :float64}]
   ;;  :gridsearch-options {:tolerance (ml-gs/linear [1e-9 1e-2])}}


   ;; :regularized-discriminant-analysis
   ;; {:attributes #{:probabilities}
   ;;  :class-name "RDA"
   ;;  :datatypes #{:float64-array}
   ;;  :name :rda
   ;;  :options [{:name :prioiri
   ;;             :type :float64-array
   ;;             :default nil}
   ;;            {:name :alpha
   ;;             :type :float64
   ;;             :default 0.0 }
   ;;            {:name :tolerance
   ;;             :default 1e-4
   ;;             :type :float64}]
   ;;  :gridsearch-options {:tolerance (ml-gs/linear [1e-9 1e-2])
   ;;                       :alpha (ml-gs/linear [0.0 1.0])}}


   :random-forest {:class RandomForest
                   :name :random-forest
                   :documentation {:user-guide "https://haifengl.github.io/classification.html#random-forest"}
                   :constructor #(RandomForest/fit ^Formula %1 ^DataFrame %2  ^Properties %3)
                   :predictor tuple-predict-posterior
                   :options [{:name :trees :type :int32 :default 500
                              :description "Number of trees"}
                             {:name :mtry :type :int32 :default 0
                              :description "number of input variables to be used to determine the decision at a node of the tree. floor(sqrt(p)) generally gives good performance, where p is the number of variables"}
                             {:name :split-rule
                              :type :string
                              :lookup-table split-rule-lookup-table
                              :default :gini
                              :description "Decision tree split rule"}
                             {:name :max-depth :type :int32 :default 20
                              :description "Maximum depth of tree"}
                             {:name :max-nodes :type :int32 :default (fn [dataset props] (unchecked-int (max 5 (/ (ds/row-count dataset) 5))))
                              :description "Maximum number of leaf nodes in the tree"}
                             {:name :node-size :type :int32 :default 5
                              :description "number of instances in a node below which the tree will not split, nodeSize = 5 generally gives good results"}
                             {:name :sample-rate :type :float32 :default 1.0 :description "the sampling rate for training tree. 1.0 means sampling with replacement. < 1.0 means sampling without replacement."}
                             {:name :class-weight :type :string :default nil
                              :description "Priors of the classes. The weight of each class is roughly the ratio of samples in each class. For example, if there are 400 positive samples and 100 negative samples, the classWeight should be [1, 4] (assuming label 0 is of negative, label 1 is of positive)"}]
                             
                   :property-name-stem "smile.random.forest"}})
   ;; :rbf-network {:attributes #{}
   ;;               :class-name "RBFNetwork"
   ;;               :datatypes #{}
   ;;               :name :rbf-network}


   


(defmulti ^:private model-type->classification-model
  (fn [model-type] model-type))


(defmethod model-type->classification-model :default
  [model-type]
  (if-let [retval (get classifier-metadata model-type)]
    retval
    (throw (ex-info "Failed to find classification model"
                    {:model-type model-type
                     :available-types (keys classifier-metadata)}))))


(defn- all-int? [v]
  (def v v)
  (== 0
     (tech.v3.datatype.functional/sum
      (tech.v3.datatype.functional/rem v 1))))



(defn- train
  [feature-ds label-ds options]
  (let [entry-metadata (model-type->classification-model
                        (model/options->model-type options))
        _ (malli/check-schema (:options entry-metadata) options)
        _ (errors/when-not-error
           (every? all-int?  (ds/columns label-ds))
           "All values in target need to be castable to int.")

        target-colname (first (ds/column-names label-ds))
        feature-colnames (ds/column-names feature-ds)
        formula (smile-proto/make-formula (ds-utils/column-safe-name target-colname)
                                          (map ds-utils/column-safe-name
                                               feature-colnames))

        ;;  this does eventualy the wrong thing, but we check for int
        dataset (merge feature-ds
                       (ds/update-columnwise
                        label-ds :all
                        dtype/elemwise-cast :int32))


        data (smile-data/dataset->smile-dataframe dataset)
        properties (smile-proto/options->properties entry-metadata dataset options)
        ctor (:constructor entry-metadata)
        model (ctor formula data properties)]
    {:smile-df-used data
     :smile-props-used properties
     :smile-formula-used formula
     :model-as-bytes
     (model/model->byte-array model)}))


(defn- thaw
  [model-data]
  (model/byte-array->model (:model-as-bytes model-data)))


(defn- predict
  [feature-ds thawed-model {:keys [target-columns
                                   target-categorical-maps
                                   options]}]
  (let [entry-metadata (model-type->classification-model
                        (model/options->model-type options))
        target-colname (first target-columns)
        n-labels (-> (get target-categorical-maps target-colname)
                     :lookup-table
                     count)
        _ (errors/when-not-error (pos? n-labels) "n-labels equals 0. Something is wrong with the :lookup-table")
        predictor (:predictor entry-metadata)
        predictions (predictor thawed-model feature-ds options n-labels)]
    (-> predictions
        (dtt/->tensor)
        (model/finalize-classification (ds/row-count feature-ds)
                                       target-colname
                                       target-categorical-maps))))



(doseq [[reg-kwd reg-def] classifier-metadata]
  (ml/define-model! (keyword "smile.classification" (name reg-kwd))
    train predict {:thaw-fn thaw
                   :hyperparameters (:gridsearch-options reg-def)
                   :options (:options reg-def)
                   :documentation {:javadoc (class->smile-url (:class reg-def))
                                   :user-guide (-> reg-def :documentation :user-guide)
                                   :code-example (-> reg-def :documentation :code-example)}}))
                                   









(comment
  (do
    (require '[tech.v3.dataset.column-filters :as cf])
    (require '[tech.v3.dataset.modelling :as ds-mod])
    (require '[scicloj.metamorph.ml.loss :as loss])
    (def src-ds (ds/->dataset "test/data/iris.csv"))
    (def ds (->  src-ds
                 (ds/categorical->number cf/categorical)
                 (ds-mod/set-inference-target "species")))
    (def feature-ds (cf/feature ds))
    (def split-data (ds-mod/train-test-split ds))
    (def train-ds (:train-ds split-data))
    (def test-ds (:test-ds split-data))
    (def model (ml/train train-ds {:model-type :smile.classification/gradient-tree-boost}))
    (def prediction (ml/predict test-ds model))))

  

(comment
  (require '[tech.v3.dataset.metamorph :as ds-mm])
  (require '[scicloj.metamorph.ml  :as mm-ml])
  (require '[tech.v3.dataset.column-filters :as cf])

  ;; (require '[scicloj.metamorph.ml  :as mm-ml-mm])

  (def src-ds (ds/->dataset "test/data/iris.csv"))

  (def predicted-ctx
    (-> {:metamorph/data src-ds
         :metamorph/mode :fit
         :metamorph/id :the-model}
         
        ((ds-mm/categorical->number cf/categorical))
        ((ds-mm/set-inference-target "species"))
        ((mm-ml/model {:model-type :smile.classification/random-forest :treesa 50})))))
