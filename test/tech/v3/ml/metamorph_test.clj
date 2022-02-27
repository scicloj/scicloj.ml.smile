(ns tech.v3.ml.metamorph-test
  (:require [scicloj.metamorph.ml :as ml]
            [clojure.test :refer [deftest is]]
            [tech.v3.dataset.column-filters :as cf]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.dataset.metamorph :as ds-mm]

            [tech.v3.dataset :as ds]
            [scicloj.metamorph.ml.loss :as loss]
            [scicloj.metamorph.ml.classification]
            [scicloj.ml.smile.classification]
            [scicloj.ml.smile.metamorph :as smile-mm]
            [scicloj.metamorph.core :as morph]))
            
(comment
  (def train-ds
    (ds/->dataset {:text ["a" "b" "c"]
                   :score ["x" "y" "z"]}))


  (def test-ds-1
    (ds/->dataset {:text ["a" "b" "c"]
                   :score ["x" "y" "z"]}))
  (def test-ds-2
    (ds/->dataset {:text ["a" "b" "c"]
                   :score ["y" "x" "z"]}))

  (def pipe
    (morph/pipeline
     (ds-mm/categorical->number [:score])
     (smile-mm/count-vectorize :text :text)
     (smile-mm/bow->SparseArray :text :text)
     (ds-mm/set-inference-target :score)
     {:metamorph/id :model}
     (ml/model {:model-type :smile.classification/sparse-logistic-regression
                :sparse-column :text
                :n-sparse-columns 3})))
     
  ;;  { :lookup-table { "z" 0, "x" 1, "y" 2 }, :src-column :score, :result-datatype :f
  (def evals
    (ml/evaluate-pipelines
     [pipe]
     [{:train train-ds :test test-ds-1}
      {:train train-ds :test test-ds-2}]
      
     loss/classification-accuracy
     :accuracy))
     

  (map :metric (first evals))

  (def fitted-ctx
    (pipe {:metamorph/data train-ds
           :metamorph/mode :fit}))


  (def predicted-ctx
    (pipe
     (merge fitted-ctx
            {:metamorph/data test-ds
             :metamorph/mode :transform}))))


(deftest test-model
  (let [
        src-ds (ds/->dataset "test/data/iris.csv")
        ds (->  src-ds
                (ds/categorical->number cf/categorical)
                (ds-mod/set-inference-target "species")
                (ds/shuffle {:seed 1234}))
        feature-ds (cf/feature ds)
        split-data (ds-mod/train-test-split ds {:randomize-dataset? false})
        train-ds (:train-ds split-data)
        test-ds  (:test-ds split-data)

        pipeline (fn  [ctx]
                   ((ml/model {:model-type :smile.classification/random-forest})
                    ctx))


        fitted
        (pipeline
         {:metamorph/id "1"
          :metamorph/mode :fit
          :metamorph/data train-ds})


        prediction
        (pipeline (merge fitted
                         {:metamorph/mode :transform
                          :metamorph/data test-ds}))

        predicted-species (ds-mod/column-values->categorical (:metamorph/data prediction)
                                                             "species")]
                                                             

    (is (= ["setosa" "setosa" "virginica"]
           (take 3 predicted-species)))))

    
