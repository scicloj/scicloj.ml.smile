(ns scicloj.ml.smile.projections
  (:require
   [fastmath.kernel :as k]
   [scicloj.metamorph.ml :as ml]
   [scicloj.ml.smile.malli :as malli]
   [scicloj.ml.smile.registration :refer [class->smile-url]]
   [tablecloth.api :as tc])
  (:import
   (smile.math.kernel MercerKernel)
   (smile.projection GHA KPCA PCA ProbabilisticPCA Projection RandomProjection)))

(require '[malli.instrument :as mi])
(require '[malli.dev.pretty :as pretty])

(def model-keywords
  {
   :pca-cov
   {:class PCA
    :documentation {:user-guide "https://haifengl.github.io/feature.html#pca"}}

   :pca-cor
   {:class PCA
    :documentation {:user-guide "https://haifengl.github.io/feature.html#pca"}}

   :pca-prob
   {:class ProbabilisticPCA
    :documentation {:user-guide "https://haifengl.github.io/feature.html#ppca"}}

   :kpca
   {:class KPCA
    :documentation {:user-guid "https://haifengl.github.io/feature.html#kpca"}}

   :gha
   {:class GHA
    :documentation {:user-guide "https://haifengl.github.io/feature.html#gha"}}

   :random
   {:class RandomProjection
    :documentation {:user-guide "https://haifengl.github.io/feature.html"}}})


;; (set! *warn-on-reflection* true)

(defn- pca
  ([rows target-dims] (pca rows target-dims false))
  ([rows ^long target-dims cor?]
   (let [^PCA model (if cor? (PCA/cor rows) (PCA/fit rows))]
     (.setProjection model target-dims)
     model)))

(defn- pca-prob
  [rows target-dims]
  (ProbabilisticPCA/fit rows target-dims))

(defn- build-smile-kernel
  [kernel kernel-params]
  (cond
    (instance? MercerKernel kernel) kernel
    (fn? kernel) (k/smile-mercer kernel)
    :else (k/smile-mercer (apply k/kernel kernel kernel-params))))

(defn- kpca
  [rows target-dims kernel kernel-params threshold]
  (KPCA/fit rows (build-smile-kernel kernel kernel-params) target-dims threshold))

(defn- gha
  [rows target-dims learning-rate decay]
  (let [^GHA model (GHA. (count (first rows)) target-dims learning-rate)]
    (doseq [row rows]
      (.setLearningRate model (* decay (.getLearningRate model)))
      (.update model row))
    model))

(defn- random
  [rows target-dims]
  (let [cnt (count (first rows))]
    (RandomProjection/of cnt target-dims)))

(defn- build-model
  [rows algorithm target-dims {:keys [kernel kernel-params
                                      threshold learning-rate decay]
                               :or {kernel (k/kernel :gaussian)
                                    threshold 0.0001
                                    learning-rate 0.0001
                                    decay 0.995}}]
  (case algorithm
    :pca-cov (pca rows target-dims)
    :pca-cor (pca rows target-dims true)
    :pca-prob (pca-prob rows target-dims)
    :kpca (kpca rows target-dims kernel kernel-params threshold)
    :gha (gha rows target-dims learning-rate decay)
    :random (random rows target-dims)
    (pca rows target-dims)))

(defn- rows->array
  [ds names]
  (-> ds
      (tc/select-columns names)
      (tc/rows :as-double-arrays)))

(defn- array->ds
  [arr target-columns]
  (->> arr
       (map (partial zipmap target-columns))
       (tc/dataset)))

(defn process-reduction-fit
  [ds algorithm target-dims cnames opts]
  (let [target-columns (map  #(str  (name algorithm) "-" %) (range target-dims))
        rows (rows->array ds cnames)
        ^Projection model (build-model rows algorithm target-dims opts)
        ds-res (array->ds (.project model #^"[[D" rows) target-columns)]
    {:dataset
     (-> ds
         (tc/add-or-replace-columns (tc/columns ds-res :as-map)))
     :model model
     :cnames cnames
     :target-columns target-columns}))


(defn process-reduction-transform
  [ds model cnames target-columns]
  (let [rows (rows->array ds cnames)
        ds-res (array->ds (.project model #^"[[D" rows) target-columns)]
    (-> ds
        (tc/add-or-replace-columns (tc/columns ds-res :as-map)))))
    



(defn reduce-dimensions
  "Metamorph transformer, which reduces the dimensions of a given dataset.

  `algorithm` can be any of:
    * :pca-cov
    * :pca-cor
    * :pca-prob
    * :kpca
    * :gha
    * :random

  `target-dims` is number of dimensions to reduce to.

  `cnames` is a sequence of column names on which the reduction get performed

  `opts` are the options of the algorithm

  metamorph                            | .
  -------------------------------------|----------------------------------------------------------------------------
  Behaviour in mode :fit               | Reduces dimensions of the dataset at key `:metamorph/data` and stores the trained model in ctx under key at `:metamorph/id`
  Behaviour in mode :transform         | Reads trained reduction model from ctx and applies it to data in `:metamorph/data`
  Reads keys from ctx                  | In mode `:transform` : Reads trained model to use from ctx at key in `:metamorph/id`.
  Writes keys to ctx                   | In mode `:fit` : Stores trained model in ctx under key in `:metamorph/id`.

  "
  {:malli/schema [:=> [:cat
                       [:enum :pca-cov :pca-cor :pca-prob :kpca :gha :random]
                       int?
                       [:sequential [:or string? keyword?]]
                       any?]
                  [fn?]]}
  [algorithm target-dims cnames opts]

  (malli/instrument-mm
   (fn [{:metamorph/keys [data id mode] :as ctx}]
     (case mode
       :fit
       (let [fit-result (process-reduction-fit data algorithm target-dims cnames opts)]

         (assoc ctx
                id {:fit-result (dissoc fit-result :dataset)}
                :metamorph/data (:dataset fit-result)))
       :transform
       (let [fit-result (get-in ctx [id :fit-result])]
         (assoc ctx :metamorph/data (process-reduction-transform
                                     (:metamorph/data ctx)
                                     (:model fit-result)
                                     (:cnames fit-result)
                                     (:target-columns fit-result))))))))

(defn- train-algorithm
  [algorithm feature-ds label-ds options]
  (let [{:keys [target-dims cnames opts]} options]
    (select-keys
     (process-reduction-fit feature-ds
                            algorithm
                            target-dims
                            cnames
                            opts)
     [:dataset :model])))

(defn- train
  [feature-ds label-ds options]
  (train-algorithm (options :algorithm) feature-ds label-ds options))
  



(ml/define-model! :smile.projections
  train nil {:unsupervised? true})

(doseq [[reg-kwd reg-def] model-keywords]
  (ml/define-model!
    (keyword (str "smile.projections/" (name  reg-kwd)))
    (partial train-algorithm reg-kwd)
    nil
    {:unsupervised? true
     :documentation {:javadoc (class->smile-url (:class reg-def))
                     :user-guide (-> reg-def :documentation :user-guide)
                     :code-example nil ;; (-> reg-def :documentation :code-example)
                     :description nil}}))



(malli/instrument-ns *ns*)
;; (mi/collect! {:ns 'scicloj.ml.smile.projections})
;; (mi/instrument! {:report (pretty/thrower) :scope #{:input}})
 
