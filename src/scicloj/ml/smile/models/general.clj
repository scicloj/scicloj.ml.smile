(ns scicloj.ml.smile.models.general
  (:require
   [tech.v3.dataset :as ds]
   [tech.v3.datatype :as dtype])
  (:import
   (java.util List)
   (smile.regression LinearModel)))

(defn predict-linear-model
  [^LinearModel thawed-model ds]
  (let [^List val-rdr (ds/value-reader ds)]
    (->> (dtype/make-reader
          :float64
          (ds/row-count ds)
          (.predict thawed-model
                    ^doubles (dtype/->double-array (val-rdr idx))))
         (dtype/make-container :java-array :float64))))
