(ns scicloj.ml.smile.svm
  (:require [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as ds-mod]
            [scicloj.metamorph.ml :as ml]
            [scicloj.ml.smile.model :as model])
  (:import smile.classification.SVM))

(defn train [feature-ds target-ds options]
  "Training function of SVM model. "
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
     

(defn predict [feature-ds
               thawed-model
               model]
  "Predict function for SVM model"
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

             
   :documentation {:javadoc "http://haifengl.github.io/api/java/smile/classification/SVM.html"
                   :user-guide "https://haifengl.github.io/classification.html#svm"}})

                   
  
