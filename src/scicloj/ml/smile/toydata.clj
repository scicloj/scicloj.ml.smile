(ns scicloj.ml.smile.toydata
  (:require [clojure.java.io :as io]
            [clojure.string :as str]
            [tablecloth.api :as tc]
            [camel-snake-kebab.core :as csf]
            [tech.v3.libs.smile.data :as smile-data])
  (:import
   [smile.io Arff]))





(defn- arff->ds [data-url]
  (->  (Arff. (io/reader data-url))
       .read
       (smile-data/smile-dataframe->dataset {:key-fn csf/->kebab-case-keyword})))

(defn get-smile-data
  "Returns a data set from the Smile github repo data folder.
  The passed `data-file-name` is added as suffix to
  'https://raw.githubusercontent.com/haifengl/smile/v2.6.0/shell/src/universal/data/'

  The data file is returned as a tech.ml dataset.

  It support currently '.csv' and '.arff' files.
  "

  [data-file-name]
  (let [data-url
        (format
         "https://raw.githubusercontent.com/haifengl/smile/v2.6.0/shell/src/universal/data/%s"
         data-file-name)]


    (cond (str/ends-with? data-file-name ".arff") (arff->ds data-url)
          (str/ends-with? data-file-name ".csv") (tc/dataset data-url {:key-fn csf/->kebab-case-keyword})
          true (Exception. (format "Not supported file name extension: %s" data-file-name)))))



(comment
  (get-smile-data "classification/breastcancer.csv")
  (get-smile-data "weka/regression/cal_housing.arff")
  (get-smile-data
   "weka/weather.arff"))
