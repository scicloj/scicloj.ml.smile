(ns scicloj.ml.smile.test-tools)

(defn fuzzy=? [tolerance x y]
  (let [diff (Math/abs (- x y))]
    (< diff tolerance)))

(defn seq-fuzzy=? [s1 s2 tolerance]
  (every?
   true?
   (map #(fuzzy=? tolerance %1 %2 ) s1 s2)))
