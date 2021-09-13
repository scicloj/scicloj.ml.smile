(ns scicloj.ml.smile.clustering-test
  (:require  [clojure.test :as t]
             [fastmath.clustering :as clustering]
             [tablecloth.api :as tc]
             [scicloj.ml.smile.clustering :refer [cluster]]
             [scicloj.metamorph.core :as morph]
             [scicloj.metamorph.ml :as morphml]))



(def data
  (tc/dataset {:f1 [1 5 1 5 8]
               :f2 [2 5 4 3 1]
               :f3 [3 6 2 2 2]
               :f4 [4 7 3 1 2]}))

(def split
  (first
   (tc/split->seq data :holdout)))


(t/deftest cluster-test
  (let [pipeline (morph/pipeline
                  {:metamorph/id :cluster} (cluster :k-means [3]))

        fittex-ctx
        (pipeline
         {:metamorph/mode :fit
          :metamorph/data (:train split)})]
    (t/is (= 3
           (-> fittex-ctx :cluster :clustering count)))))
