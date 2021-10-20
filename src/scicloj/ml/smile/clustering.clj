(ns scicloj.ml.smile.clustering
  (:require
   [scicloj.metamorph.ml :as ml]
   [fastmath.clustering :as clustering]
   [tablecloth.api :as tc]
   [scicloj.ml.smile.malli :as malli]))



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

The cluster id of each row gets written to the column in `target-column`

  metamorph                            | .
  -------------------------------------|----------------------------------------------------------------------------
  Behaviour in mode :fit               | Calculates cluster centers of the rows dataset at key `:metamorph/data` and stores them in ctx under key at `:metamorph/id`. Adds as wll column in `target-column` with cluster centers into the dataset.
  Behaviour in mode :transform         | Reads cluster centers from ctx and applies it to data in `:metamorph/data`
  Reads keys from ctx                  | In mode `:transform` : Reads cluster centers to use from ctx at key in `:metamorph/id`.
  Writes keys to ctx                   | In mode `:fit` : Stores cluster centers in ctx under key in `:metamorph/id`.

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
           clustering
           (case mode
             :fit (fit-cluster data clustering-method clustering-method-args)
             :transform (ctx id))
           clusters (map (partial clustering/predict clustering)
                         data-rows)]

       (cond-> ctx
         (= mode :fit) (assoc id clustering)
         true          (update :metamorph/data
                               tc/add-column target-column clusters))))))


(defn train-fn [feature-ds label-ds options] {}
  (fit-cluster feature-ds
                          (options :clustering-method)
                          (options :clustering-method-args)))




(ml/define-model! :fastmath/cluster train-fn nil {:unsupervised? true})


(malli/instrument-ns 'scicloj.ml.smile.clustering)
