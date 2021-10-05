(ns scicloj.ml.smile.malli
  (:require
   [malli.util :as mu]
   [malli.core :as m]
   [malli.error :as me]))


(defn type->malli [type]
  (case type
    :int32 'integer?
    :string 'string?
    :float32 'float?
    :float64 'double?
    :boolean 'boolean?
    :keyword 'keyword?
    :enumeration 'any?))


(defn options->malli [options]
 (->> options
      (mapv (fn [option]
              (vector (:name option)
                      (type->malli (:type option)))))))


(defn check-schema [defined-options options]
  (def options options)
  (let [;; entry-metadata (model-type->classification-model
        ;;                 (model/options->model-type options))
        malli-schema (apply vector :map (options->malli defined-options))

        model-options (dissoc options :model-type)
        final-schema (-> malli-schema
                         mu/optional-keys
                         mu/closed-schema)
        explanation (m/explain final-schema model-options)]
    (when (some? explanation)
      (throw (IllegalArgumentException. (str "invalid options:" (me/humanize explanation)))))))
