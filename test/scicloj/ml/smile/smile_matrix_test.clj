(ns scicloj.ml.smile.smile-matrix-test
  (:require
   [clojure.test :refer [is deftest]]
   [scicloj.ml.smile.smile-matrix]
   [tech.v3.libs.smile.matrix])
  (:import
   [smile.math.matrix Matrix]))



(deftest round-trip
  (let 
   [ m (Matrix/rand 3 4 0.1 1)]
    (is (= m
           (-> m
               tech.v3.libs.smile.matrix/smile-matrix-as-tensor
               tech.v3.libs.smile.matrix/tensor->smile-matrix)))))


(comment
  (println
   (-> (Matrix. 3 4)
       scicloj.ml.smile.smile-matrix/smile-matrix-as-tensor))
;;=> Execution error (ArityException) at tech.v3.datatype.protocols/eval12166$fn$G (protocols.clj:43).
;;   Wrong number of args (2) passed to: tech.v3.datatype.nio-buffer/buf->buffer
;;   
  
    

  (def t
    (-> 
     (Matrix. 3 4)
     tech.v3.libs.smile.matrix/smile-matrix-as-tensor
     ))
  


  (println
   (tech.v3.tensor/new-tensor [3 4]))
  )

