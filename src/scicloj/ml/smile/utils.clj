(ns scicloj.ml.smile.utils
  (:require

   [tech.v3.datatype.errors :as errors]))

(defn when-not-pos-error [x error]
  (errors/when-not-error (and (not (nil? x)) (pos? x)) error))
