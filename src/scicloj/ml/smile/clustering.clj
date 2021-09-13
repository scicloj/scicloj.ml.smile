(ns scicloj.ml.smile.clustering
  (:require
             [fastmath.clustering :as clustering]
             [tablecloth.api :as tc]))



(defn cluster  [clustering-method clustering-method-args]
  (fn [ctx]
    (let [mode (:metamorph/mode ctx)
          id (:metamorph/id ctx)
          data (:metamorph/data ctx)
          fun (resolve (symbol  "fastmath.clustering" (name :lloyd)))
          data-rows (tc/rows data)
          clustering
          (case mode
            :fit (apply fun data-rows clustering-method-args)
            :transform (ctx id))
          clusters (map (partial clustering/predict clustering)
                    data-rows)]

      (cond-> ctx
        (= mode :fit) (assoc id clustering)
        true          (update :metamorph/data
                              tc/add-column :cluster clusters)))))
