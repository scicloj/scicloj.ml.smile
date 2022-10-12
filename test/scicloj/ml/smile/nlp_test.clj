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
    (is (= 0.0  (nlp/tfidf :example bow-1  bows)))
    (is (= 4.216395324324493  (nlp/tfidf :example bow-2  bows)))))


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
    (is (= 2.8109302162163288 (nth tfidf-v 2)))
    (is (= {"thi" 2, "is" 2, "a" 1, "sampl" 1, "anoth" 1, "exampl" 1}
           (-> ds :dense meta :tf-map)))

    (is (= 4.216395324324493 (get tfidf-2 "exampl")))
    (is (= 2.8109302162163288 (get tfidf-1 "a")))
    (is (= 1.0 (get tfidf-1 "thi")))))

(deftest bow->tfidf->dense-handler
  (let [ds (->
            (ds/->dataset {:text ["This is a a sample"  "this is another another example example example"]})
            (nlp/count-vectorize :text :bow  {:stopwords [""]})
            (nlp/bow->tfidf :bow :tfidf {:tf-map-handler-fn (partial nlp/tf-map-handler-top-n 3)}))]

    (is  (= ["is" "thi" "exampl"]
            (-> ds :tfidf second keys)))))


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


(deftest test-tf
  (is (= 1.0 (float (nlp/tf :a {:a 1 :b 1}))))
  (is (= 1.0 (nlp/tf :a {:a 1 :b 2})))
  (is (= 0.0 (nlp/tf :c {:a 1 :b 2}))))

(deftest test-idf
  (is (= 1.0 (nlp/idf :a [{:a 1} {:a 2 :b 1 :c 3}])))
  (is (=  1.4054651081081644   (nlp/idf :a [{:c 1} {:a 2 :b 1 :c 3}])))
  (is (= 2.09861228866811    (nlp/idf :d [{:c 1} {:a 2 :b 1 :c 3}]))))


(deftest test-idf-2
  ;; TfidfTransformer(norm=None).fit_transform([[1,0,0],[2,1,3]])
  ;; array([1.        , 1.40546511, 1.40546511])
  (is (= [1.0 1.4054651081081644 1.4054651081081644]
         (map
          #(nlp/idf % [{:a 1 :b 0 :c 0} {:a 2 :b 1 :c 3}])
          [:a :b :c]))))


(deftest tfidf-from-count
;;   The same result as sklearn:)

;; counts = [[3, 0, 1],
;;           [2, 0, 0],
;;           [3, 0, 0],
;;           [4, 0, 0],
;;           [3, 2, 0],
;;           [3, 0, 2]]
;; TfidfTransformer(norm=None).fit_transform(counts).toarray()

  (let [counts [[3 0 1]
                [2 0 0]
                [3 0 0]
                [4 0 0]
                [3 2 0]
                [3 0 2]]
        bows
        (map
         #(zipmap
           [:a :b :c]
           %)
         counts)]

    (is (=
         [[3.0       0.0               1.8472978603872037]
          [2.0       0.0               0.0]
          [3.0       0.0               0.0]
          [4.0       0.0               0.0]
          [3.0       4.505525936990736 0.0]
          [3.0       0.0               3.6945957207744073]]
         (for [bow bows]
           (for [term [:a :b :c]]
             (nlp/tfidf term bow bows)))))))




