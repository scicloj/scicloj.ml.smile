(ns scicloj.ml.smile.manifold
  (:require [smile.manifold :as smile-mf]
            [scicloj.metamorph.ml :as ml]
            [tablecloth.api :as tc]
            [scicloj.metamorph.ml.model :as model]))


(defn manifold [data manifold-method manifold-method-args]
  (let [fun (resolve (symbol (str "smile.manifold/" (name manifold-method))))
        data-rows (tc/rows data :as-double-arrays)]
    (apply fun data-rows manifold-method-args)))

(defn- train [manifold-method]
  (fn  [feature-ds label-ds options]

    (let [model (manifold feature-ds manifold-method (options :args))]
      {:coordinates
       (tc/dataset (.coordinates model))
       :model model})))



(doseq [reg-kwd [:isomap :laplacian :lle :tsne :umap]]
  (ml/define-model! (keyword "smile.manifold" (name reg-kwd))
    (train reg-kwd) nil {:unsupervised? true}))
