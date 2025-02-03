(ns scicloj.ml.smile.smile-data-test
  (:require [tech.v3.dataset :as ds]
            [tech.v3.datatype.functional :as dfn]
            [tech.v3.libs.smile.data :as smile-data]
            [scicloj.metamorph.ml.toydata :as data]
            [clojure.test :refer [deftest is]])
  (:import [smile.data DataFrame]))


(deftest stocks-test
  (let [stocks (ds/->dataset "test/data/stocks.csv")
        df-stocks (smile-data/dataset->smile-dataframe stocks)
        new-val (smile-data/smile-dataframe->dataset df-stocks)]
    (is (instance? DataFrame df-stocks))
    ;;Datetime types included
    (is (= (vec ((ds/ensure-array-backed stocks) "date"))
           (vec (new-val "date"))))
    (is (= (vec ((ds/ensure-array-backed stocks) "symbol"))
           (vec (new-val "symbol"))))
    (is (dfn/equals (stocks "price")
                    (new-val "price")))))


(deftest ames-test
  (let [ames-src (-> (ds/->dataset "test/data/ames-house-prices/train.csv")
                     (ds/select-rows (range 10)))
        ames-ary (ds/ensure-array-backed ames-src)
        df-ames (smile-data/dataset->smile-dataframe ames-ary)
        new-val (smile-data/smile-dataframe->dataset df-ames)]
    (is (every? = (map vector
                       (map (comp :datatype meta) (vals ames-src))
                       (map (comp :datatype meta) (vals ames-ary))
                       (map (comp :datatype meta) (vals new-val)))))
    (is (java.util.Objects/equals (ds/missing ames-src)
                                  (ds/missing ames-ary)))
    ;;Missing for booleans gets lost in the translation with inference turned on.
    #_(is (java.util.Objects/equals (ds/missing ames-src)
                                    (ds/missing new-val)))
    (is (instance? DataFrame df-ames))
    ;;Datetime types included
    (is (= (vec (ames-src "SalePrice"))
           (vec (new-val "SalePrice"))))
    ;;Missing for booleans gets lost in the translation with inference turned on.
    #_(is (= (vec (ames-src "PoolQC"))
             (vec (new-val "PoolQC"))))))

(defn- validate-round-trip [ds]
  (is (= ds
         (-> ds

             (smile-data/dataset->smile-dataframe)
             (smile-data/smile-dataframe->dataset)))))




(deftest test-validate-round-trip

  (validate-round-trip
   (->
    (data/iris-ds)
    (ds/rename-columns ["sepal_length" "sepal_width" "petal_length" "petal_width" "species"])))

  (validate-round-trip
   (->
    (data/breast-cancer-ds)
    (ds/rename-columns
     ["mean-radius"
      "mean-texture"
      "mean-perimeter"
      "mean-area"
      "mean-smoothness"
      "mean-compactness"
      "mean-concavity"
      "mean-concave-points"
      "mean-symmetry"
      "mean-fractal-dimension"
      "radius-error"
      "texture-error"
      "perimeter-error"
      "area-error"
      "smoothness-error"
      "compactness-error"
      "concavity-error"
      "concave-points-error"
      "symmetry-error"
      "fractal-dimension-error"
      "worst-radius"
      "worst-texture"
      "worst-perimeter"
      "worst-area"
      "worst-smoothness"
      "worst-compactness"
      "worst-concavity"
      "worst-concave-points"
      "worst-symmetry"
      "worst-fractal-dimension"
      "class"])))
      
      (validate-round-trip 
       (ds/->dataset {"a" [true false true] 
                      "b" ["x" nil "z"]
                      "c" [1 2 3]
                      "d" [1.0 nil 2.0]
                      "e" [0.1 0.2 0.3]
                      "f" [nil 0.2 nil]
                      "g" [nil nil nil]
                      "h" [1 "x" 1.0]
                      "i" [ [1 2] [3 4] [5 6]]
                          })
       )
      )

