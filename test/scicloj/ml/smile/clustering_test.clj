(ns scicloj.ml.smile.clustering-test
  (:require  [clojure.test :refer [deftest is] :as t]
             [fastmath.clustering :as clustering]
             [tablecloth.api :as tc]
             [tablecloth.pipeline :as tc-mm]
             [scicloj.ml.smile.clustering :refer [cluster]]
             [scicloj.metamorph.core :as mm]
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
  (let [pipeline (mm/pipeline
                  {:metamorph/id :cluster} (cluster :k-means [3] :cluster-row))
        fittex-ctx
        (pipeline
         {:metamorph/mode :fit
          :metamorph/data (:train split)})

        _ (def fittex-ctx fittex-ctx)]
    (t/is (= 3
             (-> fittex-ctx :cluster :clustering count)))
    (t/is (= 3 (-> fittex-ctx :metamorph/data :cluster-row count)))))


(def iris
  (->
   (tc/dataset
    "https://raw.githubusercontent.com/scicloj/metamorph.ml/main/test/data/iris.csv" {:key-fn keyword})))

(deftest cluster-test
  (let [
        pipe-fn
        (mm/pipeline
         (tc-mm/drop-columns [:species])
         {:metamorph/id :cluster} (cluster :k-means [3] :cluster))

        fitted-ctx
        (mm/fit-pipe iris pipe-fn)]
    (is (= 3
           (-> fitted-ctx :cluster :clusters)))))


(deftest cluster-model-test
  (is (= :g-means
         (get-in
          (mm/fit iris
                  (tc-mm/drop-columns [:species])
                  {:metamorph/id :cluster}
                  (morphml/model {:model-type :fastmath/cluster
                                  :clustering-method :g-means
                                  :clustering-method-args [5]}))
          [:cluster :model-data :type]))))

(deftest cluster-model-test
  (is (= :g-means
         (get-in
          (mm/fit iris
                  (tc-mm/drop-columns [:species])
                  {:metamorph/id :cluster}
                  (morphml/model {:model-type :fastmath.cluster/g-means
                                  :clustering-method-args [5]}))
          [:cluster :model-data :type]))))
