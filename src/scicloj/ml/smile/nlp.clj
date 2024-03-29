(ns scicloj.ml.smile.nlp
  (:require
   [clojure.set :as set]
   [clojure.string :as str]
   [pppmap.core :as ppp]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.column :as ds-col]
   [tech.v3.datatype.errors :as errors])
  (:import
   (smile.nlp.dictionary EnglishStopWords)
   (smile.nlp.normalizer SimpleNormalizer)
   (smile.nlp.stemmer PorterStemmer Stemmer)
   (smile.nlp.tokenizer BreakIteratorSentenceSplitter SimpleTokenizer)
   (smile.util SparseArray)))

(defn resolve-stopwords [stopwords-option]
  (if (keyword? stopwords-option)
    (iterator-seq (.iterator (EnglishStopWords/valueOf (str/upper-case (name stopwords-option)))))
    stopwords-option))

(def simple-normalizer (SimpleNormalizer/getInstance))

(defn default-word-normalize [word]
  (-> word
      str/lower-case
      (#(.normalize ^SimpleNormalizer simple-normalizer  ^String %))))

(defn word-process [^Stemmer stemmer word-normalizer-fn ^String word options]
  (let [word (word-normalizer-fn word)
        word (if (nil? stemmer)
               word
               (.stem stemmer word))]
    word))

(defn resolve-stemmer [options]
  (let [stemmer-type (get options :stemmer :porter)]
       (case stemmer-type
                :none nil
                :porter (PorterStemmer.))))
                  
(defn default-tokenize
  "Tokenizes text.
  The usage of a stemmer can be configured by options :stemmer "
  ([text options]
   (let [word-normalizer-fn (get options :word-normalizer-fn  default-word-normalize)
         stemmer (resolve-stemmer options)
         tokenizer (SimpleTokenizer.)
         sentence-splitter (BreakIteratorSentenceSplitter.)

         tokens
         (->> text
              (#(.normalize ^SimpleNormalizer simple-normalizer ^String %))
              (.split sentence-splitter)
              (map #(.split tokenizer %))
              (map seq)
              flatten
              (remove nil?)
              (map #(word-process stemmer word-normalizer-fn % options))
              (remove empty?))]

             
     tokens))
  ([text] (default-tokenize text {})))



(defn default-text->bow
  "Converts text to token counts (a map token -> count).
   Takes options:
   `stopwords` being either a keyword naming a
   default Smile dictionary (:default :google :comprehensive :mysql)
   or a seq of stop words. The stopwords get normalized in the same way
   as the text itself, so it should contain `full words` (non stemmed)
   As default, no stopwords are used.
   `stemmer` being either :none or :porter for selecting the porter stemmer.
   `freq-handler-fn` A function taking a term-frequency map, and can further manipulate it.
     Defaults to `identity`
"
  ([text options]
   (let [normalize-fn (get options :word-normalizer-fn  default-word-normalize)
         stemmer (resolve-stemmer options)
         stopwords-option (:stopwords options)
         stopwords  (resolve-stopwords stopwords-option)
         processed-stop-words (map #(word-process stemmer normalize-fn % options) stopwords)
         tokens (default-tokenize text options)
         freqs (-> tokens frequencies ((get options :freq-handler-fn identity)))]

     (apply dissoc freqs processed-stop-words)))
  ([text] (default-text->bow text {})))






(defn count-vectorize
  "Converts text column `text-col` to bag-of-words representation
   in the form of a frequency-count map.
  The default text->bow function is `default-text-bow`.
  All `options` are passed to it.
  "
  ([ds text-col bow-col {:keys [text->bow-fn]
                         :or {text->bow-fn default-text->bow}
                         :as options}]
                         
   (ds/add-or-update-column
    ds
    (ds-col/new-column
     bow-col
     (ppp/ppmap-with-progress
      "text->bow"
      (get options :ppmap-grain-size 1000)
      #(text->bow-fn % options)
      (get ds text-col)))))
  ([ds text-col bow-col]
   (count-vectorize ds text-col bow-col {:text->bow-fn  default-text->bow})))
   
  

(defn ->vocabulary-top-n
  "Takes top-n most frequent tokens as vocabulary"
  [bows n]
  (let [vocabulary
        (->>
         (apply merge-with + bows)
         (sort-by second)
         reverse
         (take n)
         keys)]
    vocabulary))

(defn create-vocab-all
  "Uses all tokens as the vocabulary"
  [bow]
  (keys
   (apply merge bow)))
  



(defn bow->sparse-and-vocab
  "Converts a bag-of-word column `bow-col` to a sparse data column `indices-col`.
   The exact transformation to the sparse representtaion is given by `bow->sparse-fn`"
  [ds bow-col indices-col bow->sparse-fn {:keys [create-vocab-fn] :or {create-vocab-fn create-vocab-all}}]
  (let [bow (get ds bow-col)
        _ (errors/when-not-error bow (str "bow column not found: " bow-col))
        vocabulary-list (create-vocab-fn bow)
        vocab->index-map (zipmap vocabulary-list  (range))
        vocabulary {:vocab vocabulary-list
                    :vocab->index-map vocab->index-map
                    :index->vocab-map (set/map-invert vocab->index-map)}
                    
        vocab->index-map (:vocab->index-map vocabulary)
        ds
        (ds/add-or-update-column
         ds
         (ds-col/new-column
          indices-col
          (ppp/ppmap-with-progress
           "bow->sparse"
           1000
           #(bow->sparse-fn % vocab->index-map)
           (get ds bow-col))
          {:categorical? false}))]
    {:ds ds
     :vocab vocabulary}))
    

(defn bow->sparse
  "Generic function to convert a colmn to something sparse,
  using the given vocabulary."
  [ds bow-col indices-col bow->sparse-fn vocabulary]
  (let [
        vocab->index-map (:vocab->index-map vocabulary)
        ds
        (ds/add-or-update-column
         ds
         (ds-col/new-column
          indices-col
          (ppp/ppmap-with-progress
           "bow->sparse"
           1000
           #(bow->sparse-fn % vocab->index-map)
           (get ds bow-col))))]
    {:ds ds
     :vocab vocabulary}))
    

(defn bow->something-sparse
  "Converts a bag-of-word column `bow-col` to a sparse data column `indices-col`.
   The exact transformation to the sparse representtaion is given by `bow->sparse-fn`"
  [ds bow-col indices-col bow->sparse-fn options]
  (let [{:keys [ds vocabulary]}
        (bow->sparse-and-vocab ds bow-col indices-col bow->sparse-fn options)]
    ds))

(defn tf-map [bows]
  (loop [m {} bows bows]
    (let [bow (first bows)
          token-present (zipmap (keys bow) (repeat 1))]

      (if (empty? bows)
        m
        (recur
         (merge-with + m token-present)
         (rest bows))))))

;; https://en.wikipedia.org/wiki/Tf%E2%80%93idf#Definition
;; variant "term freqency"
(defn- tf-term-frequency [term bow]
  (if (empty? bow)
    0
    (float (/
            (get bow term 0)
            (apply + (vals bow))))))


;; https://en.wikipedia.org/wiki/Tf%E2%80%93idf#Definition
;; variant "raw count"
(defn- tf-raw [term bow]
  (float (get bow term 0)))


(defn tf
  ([term bow options]
   (case (or (:tf-weighting-scheme options) :raw-count)
     :raw-count (tf-raw term bow)
     :term-frequency (tf-term-frequency term bow)))
  ([term bow] (tf term bow nil)))


;;  num docs containing term
(defn- n_t [term bows options]
  (apply + (map #(Math/signum ^float (tf term % options))
                bows)))

;;  this is as skleran does it when smooth_idf=True
;;  idf(t) = log [ (1 + n) / (1 + df(t)) ] + 1.
;;  does not match precisely any of
;; https://en.wikipedia.org/wiki/Tf%E2%80%93idf#Inverse_document_frequency_2
(defn- idf-smooth-sklearn
  ([term bows options]
   (let [N (count bows)
         n_t (n_t term bows options)]
     (+ 1
        (Math/log (/
                   (+ 1 N)
                   (+ 1 n_t)))))))


;; "inverse document frequeny smooth" from
;; https://en.wikipedia.org/wiki/Tf%E2%80%93idf#Inverse_document_frequency_2

(defn- idf-smooth
  ([term bows options]
   (let [N (count bows)
         n_t (n_t term bows options)]
     (+ 1
        (Math/log (/
                   N
                   (+ 1 n_t)))))))


;; "inverse document frequeny" from
;; https://en.wikipedia.org/wiki/Tf%E2%80%93idf#Inverse_document_frequency_2
(defn- idf-idf [term bows options]
  (let [N (count bows)
        n_t (n_t term bows options)]
    (Math/log (/ N n_t))))


(defn idf
  ([term bows options]
   (case (or (:idf-weighting-scheme options) :smooth-sklearn)
     :smooth-sklearn (idf-smooth-sklearn term bows options)
     :idf (idf-idf term bows options)
     :smooth (idf-smooth term bows options)))
  ([terms bows] (idf terms bows nil)))

(defn tfidf
  "Calculates tfidf.
  `term` : The term for which to calculate the tfidf value
  `bow`  : bag-of-words representation of the document (= term-frequency map)
  `bows` : list of bag-of-words representing the corpus (= list of term-frequency maps)

  `options` supported :
      - `:tf-weighting-scheme` The term-frequency weighting scheme with supported values `:raw-count` , `:term-frequency`
         Default is: `:raw-count`
         see here: https://en.wikipedia.org/wiki/Tf%E2%80%93idf#Term_frequency_2
      - `:idf-weighting-scheme` The inverse term-frequency weighting scheme with supported values `:smooth-sklearn`, `:idf`, `:smooth`
         Default is: `smooth-sklearn`
         https://en.wikipedia.org/wiki/Tf%E2%80%93idf#Inverse_document_frequency_2
         https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn.feature_extraction.text.TfidfTransformer

  
  "
  ([term bow bows options]
   (* (tf term bow options)  (idf term bows options)))
  ([term bow bows] (tfidf term bow bows nil)))


(defn tf-map-handler-top-n
  "Keeps the n most frequent terms in teh term-frequency table"
  [n freqs]
  (->> freqs
       (sort-by second)
       reverse
       (take n)
       (into {})))

(defn bow->tfidf
  "Calculates the tfidf score from bag-of-words (as token frequency maps)
   in column `bow-column` and stores them in a new column `tfid-column` as maps of token->tfidf-score.
  Possible `options`:
  - `:tf-map-handler-fn` : If present, it gets applied to the global term-frequency map after creating it.
     Fn need to take map of terms to frequencies and return such map. Typical use is to prune less frequent terms.
     Defaults to `identity`, so all terms are retained.
  - `:tf-weighting-scheme` See function [[tf-idf]]
  - `:idf-weighting-scheme` See function [[tf-idf]]
  "

  ([ds bow-column tfidf-column options]
   (let [tf-map-handler-fn (get options :tf-map-handler-fn identity)
         full-bows (get ds bow-column)
         global-tf-map (or (:reuse-tf-map options)
                           (tf-map full-bows))
         used-tf-map (->> global-tf-map tf-map-handler-fn (into {}))
         bows (map #(select-keys % (keys used-tf-map)) full-bows)
         tfidf-column (ds-col/new-column tfidf-column
                                         (ppp/ppmap-with-progress
                                          "tfidf"
                                          (get options :ppmap-grain-size 1000)
                                          (fn [bow]
                                            (let [
                                                  terms (keys bow)
                                                  tfidfs
                                                  (map
                                                   #(tfidf % bow bows options)
                                                   terms)]
                                              (zipmap terms tfidfs)))
                                          bows)
                                         {:tf-map used-tf-map})]
     (ds/add-or-update-column ds tfidf-column)))
  ([ds bow-column tfidf-column]
   (bow->tfidf ds bow-column tfidf-column {})))



(defn tfidf->dense-array
  "Converts the sparse tfidf map based representation into
  dense double arrays"
  [ds tfidf-column array-column]
  (let [tf-map (-> (get  ds tfidf-column) meta :tf-map)
        all-zeros (zipmap (keys tf-map) (repeat 0))
        tfidf-arrays (ppp/ppmap-with-progress
                      "->dense-array" 100
                      #(->  (merge all-zeros %) vals double-array)
                      (get ds tfidf-column))
        tfidf-arrays-col (ds/new-column array-column
                                        tfidf-arrays
                                        (select-keys (meta (get ds tfidf-column)) [:tf-map]))]
    (ds/add-or-update-column ds  tfidf-arrays-col)))


(defn freqs->SparseArray
  "Converts the token-frequency map to s smile SparseArray"
  [freq-map vocab->index-map]
  (let [sparse-array (SparseArray.)]
    (run!
     (fn [[token freq]]
       (when (contains? vocab->index-map token)
         (.append sparse-array ^int (get vocab->index-map token) ^double freq)))
     freq-map)
    sparse-array))


(defn bow->sparse-indices
  "Converts the token-frequencies to the sparse vectors
   needed by Maxent"
  [bow vocab->index-map]
  (->>
   (merge-with
    (fn [index count]
      [index count])
    vocab->index-map
    bow)
   vals
   (filter vector?)
   (map first)
   (into-array Integer/TYPE)))
