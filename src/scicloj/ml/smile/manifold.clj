(ns scicloj.ml.smile.manifold
  (:require [smile.manifold :as smile-mf]
            [tablecloth.api :as tc]))


(defn manifold [data manifold-method manifold-method-args]
  (def data data)
  (def manifold-method manifold-method)
  (let [fun (resolve (symbol (str "smile.manifold/" (name manifold-method))))
        data-rows (tc/rows data :as-double-arrays)]
    (def fun fun)
    (def data-rows data-rows)
    (def manifold-method-args manifold-method-args)
    (->  (apply fun data-rows manifold-method-args)
         (#(.coordinates %))
         tc/dataset)))


(defn- train [manifold-method]
  (fn  [feature-ds label-ds options]
    (manifold feature-ds manifold-method (options :args))))



(doseq [reg-kwd [:isomap :laplacian :lle :tsne :umap]]
  (ml/define-model! (keyword "smile.manifold" (name reg-kwd))
    (train reg-kwd) nil {:unsupervised? true}))
