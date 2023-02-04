(ns scicloj.ml.smile.svm
  (:require
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.gridsearch :as gs]
   [scicloj.ml.smile.model :as model]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.modelling :as ds-mod])
  (:import
   (smile.classification SVM)))

(defn train
  "Training function of SVM model. "
  [feature-ds target-ds options]
  (let [train-data
        (into-array
         (map
          double-array
          (ds/value-reader feature-ds)))
        trained-model (SVM/fit train-data
                               (into-array Integer/TYPE (seq (get target-ds (first (ds-mod/inference-target-column-names target-ds)))))
                               ^double (get options :C 1.0)
                               ^double (get options :tol 1e-4))]

    {:model-as-bytes (model/model->byte-array trained-model)}))
     

(defn predict
 "Predict function for SVM model"
  [feature-ds thawed-model model]
  (let [to-predict-data
        (into-array
         (map
          double-array
          (ds/value-reader feature-ds)))
        target-colum (first (:target-columns model))
        predictions (.predict thawed-model to-predict-data)]

    (ds/new-dataset [(ds/new-column target-colum predictions {:column-type :prediction})])))

(defn- thaw
  [model-data]
  (model/byte-array->model (:model-as-bytes model-data)))


(def ^:private hyperparameters
  {:C (gs/linear 1 10)
   :tol (gs/categorical [1e-4 1e-3 1e-2 0.1])})


(ml/define-model!
  :smile.classification/svm
  train
  predict
  {
   :thaw-fn thaw
   :options [{:name :C
              :type :float32
              :default 1.0
              :description "soft margin penalty parameter"}
             {:name :tol
              :type :float32
              :default 1e-4
              :description "tolerance of convergence test"}]

   :hyperparameters hyperparameters
             
   :documentation {:javadoc "http://haifengl.github.io/api/java/smile/classification/SVM.html"
                   :user-guide "https://haifengl.github.io/classification.html#svm"}})

                   
  
