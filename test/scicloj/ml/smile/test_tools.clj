(ns scicloj.ml.smile.test-tools
  (:require  [clojure.test :as t]))





(defn fuzzy=? [tolerance x y]
  (let [diff (Math/abs (- x y))]
    ;; (println "fuzzy=? :" x y diff tolerance)
    (< diff tolerance)))

(defn seq-fuzzy=? [s1 s2 tolerance]
  ;; (println "s1: " s1)
  ;; (println "s2: " s2)
  (every?
   true?
   (map #(fuzzy=? tolerance %1 %2 ) s1 s2)))
