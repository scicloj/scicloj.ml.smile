(ns scicloj.ml.smile.clustering
  (:require
   [fastmath.clustering :as clustering]
   [scicloj.metamorph.ml :as ml]
   [scicloj.ml.smile.malli :as malli]
   [scicloj.ml.smile.registration :refer [class->smile-url]]
   [tablecloth.api :as tc])
  (:import
   (smile.clustering CLARANS DBSCAN DENCLUE DeterministicAnnealing GMeans KMeans MEC SpectralClustering XMeans)))

(def model-keywords
  {
   :spectral {:class SpectralClustering
              :documentation {:user-guide "https://haifengl.github.io/clustering.html#spectral-clustering"}}
   :dbscan {:class DBSCAN
            :documentation {:user-guide "https://haifengl.github.io/clustering.html#dbscan"}}
   :k-means {:class KMeans
             :documentation {:user-guide "https://haifengl.github.io/clustering.html#k-means"}}
   :mec {:class MEC
         :documentation {:user-guide "https://haifengl.github.io/clustering.html#mec"}}
   :clarans {:class CLARANS
             :documentation {:user-guide "https://haifengl.github.io/clustering.html#clarans"}}
   :g-means {:class GMeans
             :documentation {:user-guide "https://haifengl.github.io/clustering.html#g-means"}}
   :lloyd {:class KMeans
           :documentation {:user-guide "https://haifengl.github.io/clustering.html#k-means"}}
   :x-means {:class XMeans
             :documentation {:user-guide "https://haifengl.github.io/clustering.html#x-means"}}
   :deterministic-annealing {:class DeterministicAnnealing
                             :documentation {:user-guide "https://haifengl.github.io/clustering.html#deterministic-annealing"}}
   :denclue {:class DENCLUE
             :documentation {:user-guide "https://haifengl.github.io/clustering.html#denclue"}}})


(defn fit-cluster [data clustering-method clustering-method-args]
  (let [
        fun (resolve (symbol  "fastmath.clustering" (name clustering-method)))
        data-rows (tc/rows data)]
    (apply fun data-rows clustering-method-args)))

(defn cluster
  "Metamorph transformer, which clusters the data and creates a new column with the cluster id.

  `clustering-method` can be any of:

* :spectral
* :dbscan
* :k-means
* :mec
* :clarans
* :g-means
* :lloyd
* :x-means
* :deterministic-annealing
* :denclue

The `clustering-args` is a vector with the positional arguments for each cluster function,
as documented here:
https://cljdoc.org/d/generateme/fastmath/2.1.5/api/fastmath.clustering
(but minus the `data` argument, which will be passed in automatically)

The cluster id of each row gets written to the column in `target-column`

  metamorph                    | .
  -----------------------------|----------------------------------------------------------------------------
  Behaviour in mode :fit       | Calculates cluster centers of the rows dataset at key `:metamorph/data` and stores them in ctx under key at `:metamorph/id`. Adds as wll column in `target-column` with cluster centers into the dataset.
  Behaviour in mode :transform | Reads cluster centers from ctx and applies it to data in `:metamorph/data`
  Reads keys from ctx          | In mode `:transform` : Reads cluster centers to use from ctx at key in `:metamorph/id`.
  Writes keys to ctx           | In mode `:fit` : Stores cluster centers in ctx under key in `:metamorph/id`.

  "
  {:malli/schema [:=> [:cat
                       [:enum :spectral :dbscan :k-means :mec :clarans :g-means :lloyd :x-means :deterministic-annealing :denclue]
                       sequential?
                       [:or string? keyword?]]
                  [fn?]]}
  [clustering-method clustering-method-args target-column]
  (malli/instrument-mm
   (fn [ctx]
     (let [mode (:metamorph/mode ctx)
           id (:metamorph/id ctx)
           data (:metamorph/data ctx)
           fun (resolve (symbol  "fastmath.clustering" (name clustering-method)))
           data-rows (tc/rows data)
           clusterresult-and-clusters
           (case mode
             :fit (let [fit-result (fit-cluster data clustering-method clustering-method-args)]
                    {:clusterresult  fit-result
                     :clusters (:clustering fit-result)})
                    
             :transform {:clusterresult  (ctx id)
                         :clusters (map (partial clustering/predict (ctx id))
                                        data-rows)})]

       (cond-> ctx
         (= mode :fit) (assoc id (clusterresult-and-clusters :clusterresult))
         true          (update :metamorph/data
                               tc/add-column target-column (clusterresult-and-clusters :clusters)))))))


(defn train-fn [feature-ds label-ds options]
  (fit-cluster feature-ds
                          (options :clustering-method)
                          (options :clustering-method-args)))

(defn train-fn-method [clustering-method feature-ds label-ds options]
  (fit-cluster feature-ds
                          clustering-method
                          (options :clustering-method-args)))


(ml/define-model! 
  :fastmath/cluster 
  train-fn
  (fn [_] (throw (Exception. "prediction not supported"))) 
  {:unsupervised? true})

(run!
 (fn [[kwf reg-def]]
   (ml/define-model!
     (keyword (str "fastmath.cluster/" (name  kwf)))
     (partial train-fn-method kwf)
     (fn [_] (throw (Exception. "prediction not supported"))) 
     {:documentation
      {:javadoc (class->smile-url (:class reg-def))
       :user-guide (-> reg-def :documentation :user-guide)
       :code-example nil} ;; (-> reg-def :documentation :code-example)
       
      :unsupervised? true}))
 model-keywords)



(malli/instrument-ns 'scicloj.ml.smile.clustering)
