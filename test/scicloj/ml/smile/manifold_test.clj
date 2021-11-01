(ns scicloj.ml.smile.manifold-test
  (:require  [clojure.test :refer [deftest is] :as t]
             [scicloj.metamorph.core :as mm]
             [scicloj.metamorph.ml :as ml]
             [camel-snake-kebab.core :as csk]
             [tablecloth.pipeline :as tc-mm]
             [tablecloth.api :as tc]
             [scicloj.metamorph.ml.preprocessing]))


(deftest manifold
  (let [pinguins
        (->
         (tc/dataset
          "https://github.com/allisonhorst/palmerpenguins/raw/5b5891f01b52ae26ad8cb9755ec93672f49328a8/data/penguins_size.csv"
          {:key-fn csk/->kebab-case-keyword}))
        pipe
        (mm/pipeline
         (tc-mm/drop-missing)
         (tc-mm/select-columns [:culmen-length-mm :culmen-depth-mm :flipper-length-mm :body-mass-g])
         (scicloj.metamorph.ml.preprocessing/std-scale :type/numerical {})
         {:metamorph/id :model}
         (ml/model {:model-type :smile.manifold/isomap
                    :args [4 5 false]}))
        fit-ctx
        (mm/fit-pipe pinguins pipe)]
    (is (= [334 4]
           (-> fit-ctx :metamorph/data tc/shape)))))













  


  
  ;; (sklearn-mm/fit-transform :umap "UMAP")
