(ns scicloj.ml.smile.manifold
  (:require
   [scicloj.metamorph.ml :as ml]
   [scicloj.ml.smile.registration :refer [class->smile-url]]
   [smile.manifold :as smile-mf]
   [tablecloth.api :as tc])
  (:import
   (smile.manifold IsoMap LaplacianEigenmap LLE TSNE UMAP)))

(def definitions
  {
   :isomap {:class IsoMap
            :documentation {:user-guide "https://haifengl.github.io/manifold.html#isomap"}}
   :laplacian {:class LaplacianEigenmap
               :documentation {:user-guide "https://haifengl.github.io/manifold.html#lle"}}
   :lle {:class LLE
         :documentation {:user-guide "https://haifengl.github.io/manifold.html#laplacia"}}
   :tsne {:class TSNE
          :documentation {:user-guide "https://haifengl.github.io/manifold.html#t-sne"}}
   :umap {:class UMAP
          :documentation {:user-guide "https://haifengl.github.io/manifold.html#umap"}}})

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



(defn make-options [reg-kwd reg-def]
  (let [fun (resolve (symbol (str "smile.manifold/" (name reg-kwd))))]
    {:unsupervised? true
     :documentation {:javadoc (class->smile-url (:class reg-def))
                     :user-guide (-> reg-def :documentation :user-guide)
                     :code-example nil;; (-> reg-def :documentation :code-example)
                     :description (-> fun meta :doc)}}))


(doseq [[reg-kwd reg-def] definitions]
  (ml/define-model! (keyword "smile.manifold" (name reg-kwd))
    (train reg-kwd) 
    (fn [_] (throw (Exception. "prediction not supported"))) 
    (make-options reg-kwd reg-def)))
