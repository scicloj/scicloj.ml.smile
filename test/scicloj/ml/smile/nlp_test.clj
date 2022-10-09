(ns scicloj.ml.smile.nlp-test
  (:require [scicloj.ml.smile.nlp :as nlp]
            [clojure.string :as str]
            [tech.v3.dataset :as ds]
            [clojure.test :refer [deftest is] :as t]))

(deftest calculate-tfidf

  (let [bows [{:this 1 :is 1 :a 2 :sample 1}
              {:this 1 :is 1 :another 2 :example 3}]
        tf-map (nlp/tf-map bows)
        bow-1 (first bows)
        bow-2 (second bows)]
    (is (= 0.0  (nlp/tfidf tf-map :example bow-1  bows)))
    (is (= 0.12901285528456338  (nlp/tfidf tf-map :example bow-2  bows)))))


(deftest bow->tfidf->dense
  (let [ds (->
            (ds/->dataset {:text ["This is a a sample"  "this is another another example example example"]})
            (nlp/count-vectorize :text :bow  {:stopwords [""]})
            (nlp/bow->tfidf :bow :tfidf)
            (nlp/tfidf->dense-array :tfidf :dense))

        tfidf-1 (first (:tfidf ds))
        tfidf-2 (second (:tfidf ds))
        tfidf-v (-> ds :dense first)]

    (is (= 6 (count tfidf-v)))
    (is (= 0.12041199826559248 (nth tfidf-v 2)))
    (is (= {"thi" 2, "is" 2, "a" 1, "sampl" 1, "anoth" 1, "exampl" 1}
         (-> ds :dense meta :tf-map)))

    (is (= 0.12901285528456338 (get tfidf-2 "exampl")))
    (is (= 0.12041199826559248 (get tfidf-1 "a")))
    (is (= 0.0 (get tfidf-1 "thi")))))


(deftest freqs->SparseArray
  (is (=
       (map
        #(vector (.i %) (.x %))
        (iterator-seq
         (.iterator (nlp/freqs->SparseArray {"a" 10.0  "b" 70.0 "c" 50.0} {"a" 0 "b" 1}))))
       [[0 10.0] [1 70.0]])))



(deftest custom-text->bow-fn
  (let [test-text->bow
        (fn [text options]
          (let [tokens (str/split text #" ")
                freqs (frequencies tokens)]
            freqs))
        ds
        (->
         (ds/->dataset {:text ["This is a a sample"  "this is another another example example example"]})
         (nlp/count-vectorize :text :bow {:text->bow-fn test-text->bow}))]

    (is (= {"This" 1, "is" 1, "a" 2, "sample" 1}
           (-> ds :bow first)))))
    
