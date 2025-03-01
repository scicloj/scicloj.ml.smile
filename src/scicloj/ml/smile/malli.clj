(ns scicloj.ml.smile.malli
  (:require
   [malli.core :as m]
   [malli.dev.pretty :as pretty]
   [malli.error :as me]
   [malli.instrument :as mi]
   [malli.util :as mu]
   [tech.v3.dataset.impl.dataset :refer [dataset?]]))

(defn type->malli [type]
  (case type
    :float64-array [:sequential float?]
    :int32 'integer?
    :string 'string?
    :float32 'float?
    :float64 'double?
    :boolean 'boolean?
    :keyword 'keyword?
    :enumeration 'any?
    :key-string-symbol [:or :keyword :string :symbol]
    :seq sequential?))


(defn options->malli [options]
  (vec
   (concat [:map {:closed true}]
           (->> options
                (mapv (fn [option]
                        (vector
                         (:name option)
                         {:optional true
                          :description (:description option)
                          :default (:default option)
                          :lookup-table (:lookup-table option)
                          :type (:type option)
                          }
                         (type->malli (:type option)))))))))


(defn check-schema [defined-options options]
  (def options options)
  (def defined-options defined-options)
  (let [
        malli-schema (apply vector :map (options->malli defined-options))

        model-options (dissoc options :model-type)
        final-schema (-> malli-schema
                         mu/optional-keys
                         mu/closed-schema)
        explanation (m/explain final-schema model-options)]
    (when (some? explanation)
      (throw (IllegalArgumentException. (str "invalid options:" (me/humanize explanation)))))))


(defn instrument-mm [fn]
  (m/-instrument
     {:report (pretty/thrower) :scope #{:input}
      :schema [:=> [:cat [:map
                          [:metamorph/id any?]
                          [:metamorph/data [:fn dataset?]]
                          [:metamorph/mode [:enum :fit :transform]]]]

               map?]}
     fn))

(defn instrument-ns [ns]
  (mi/collect! {:ns ns})
  (mi/instrument! {:report (pretty/thrower) :scope #{:input}}))

;; (mi/unstrument!)
