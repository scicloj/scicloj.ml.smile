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



  




  















  





((var clustering/clustering) :k-means (tc/rows data) 3 0 0)

(fastmath.clustering/clustering :kmeans
                                some-data
                                {:clusters 3
                                 :max-iter 10
                                 :tolerance 1.0e-4})
(def create-clusters
  (var  clustering/create-clusters))

(defn centroid-data
  [^CentroidClustering in]
  {:representatives (fastmath.core/double-double-array->seq (.centroids in))
   :distortion (.distortion in)})

(defmacro clustering
  "Analyze clustering method and pack into the structure."
  [clustering-method data & params]
  (let [[clss data-fn fit missing-predict?] ((var  clustering/clustering-classes) clustering-method)
        obj (with-meta (gensym "obj") {:tag clss})]
    `(let [~obj (. ~clss ~(or fit 'fit) (fastmath.core/seq->double-double-array ~data) ~@params)]
       (create-clusters ~clustering-method ~data ~obj ~data-fn
            ~(if missing-predict?
               `nil
               `(fn [in#] (.predict ~obj (fastmath.core/seq->double-array in#))))))))

(macroexpand-1 '(clustering :k-means data clusters))

(apply
 clustering :lloyd (tc/rows data) [10 1 1.0e-4])

(apply
 clustering/lloyd (tc/rows data) [10 1 1.0e-4])

(apply
 (resolve (symbol  "fastmath.clustering" (name :lloyd)))
 (tc/rows data)
 [10 1 1.0e-4])











(import [smile.clustering CentroidClustering PartitionClustering KMeans GMeans XMeans DeterministicAnnealing
           DENCLUE CLARANS DBSCAN MEC SpectralClustering]
        [smile.math.distance Distance]
        [clojure.lang IFn])

((var  clustering/clustering-classes) :k-means)
