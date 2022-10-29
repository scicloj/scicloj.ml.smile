(ns scicloj.ml.smile.registration
  (:require
   [clojure.string :as str]))

(defn class->smile-url [class]
  (if (nil? class)
    ""
    (str "http://haifengl.github.io/api/java/"
         (str/replace (.getName class)
                      "." "/")
         ".html")))
         
