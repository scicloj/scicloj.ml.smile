(ns scicloj.ml.smile.nlp
  (:require [clojure.string :as str]
            [pppmap.core :as ppp]
            [tech.v3.dataset :as ds]
            [tech.v3.datatype.errors :as errors])
            
  (:import smile.nlp.normalizer.SimpleNormalizer
           smile.nlp.stemmer.PorterStemmer
           [smile.nlp.tokenizer SimpleTokenizer BreakIteratorSentenceSplitter]
           [smile.nlp.dictionary EnglishStopWords]
           [smile.classification DiscreteNaiveBayes DiscreteNaiveBayes$Model]
           smile.util.SparseArray))


(defn resolve-stopwords [stopwords-option]
  (if (keyword? stopwords-option)
    (iterator-seq (.iterator (EnglishStopWords/valueOf (str/upper-case (name stopwords-option)))))
    stopwords-option))

(defn word-process [stemmer ^SimpleNormalizer normalizer ^String word]
  (let [word
        (-> word
            (str/lower-case)
            (#(.normalize normalizer %)))
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
  [text options]
  (let [normalizer (SimpleNormalizer/getInstance)
        stemmer (resolve-stemmer options)
        tokenizer (SimpleTokenizer.)
        sentence-splitter (BreakIteratorSentenceSplitter.)

        tokens
        (->> text
             (.normalize normalizer)
             (.split sentence-splitter)
             (map #(.split tokenizer %))
             (map seq)
             flatten
             (remove nil?)
             (map #(word-process stemmer normalizer %)))]

             
    tokens))



(defn default-text->bow
  "Converts text to token counts (a map token -> count).
   Takes options:
   `stopwords` being either a keyword naming a
   default Smile dictionary (:default :google :comprehensive :mysql)
   or a seq of stop words.
   `stemmer` being either :none or :porter for selecting the porter stemmer.
"
  [text options]
  (let [normalizer (SimpleNormalizer/getInstance)
        stemmer (resolve-stemmer options)
        stopwords-option (:stopwords options)
        stopwords  (resolve-stopwords stopwords-option)
        processed-stop-words (map #(word-process stemmer normalizer %)  stopwords)
        tokens (default-tokenize text options)
        freqs (frequencies tokens)]
    (apply dissoc freqs processed-stop-words)))

(defn- remove-punctuation [sentence]
  (->>
    sentence
    (filter #(or (Character/isLetter %)
                 (Character/isSpace %)
                 (Character/isDigit %)))
    (apply str)))


(defn count-vectorize
  "Converts text column `text-col` to bag-of-words representation
   in the form of a frequency-count map"
  ([ds text-col bow-col {:keys [text->bow-fn]
                         :or {text->bow-fn default-text->bow}
                         :as options}]
                         
   ;; (def ds ds)
   ;; (def text-col text-col)
   ;; (def bow-col bow-col)
   ;; (def options options)
   (ds/add-or-update-column
    ds
    (ds/new-column
     bow-col
     (ppp/ppmap-with-progress
       "text->bow"
       1000
      #(text->bow-fn % options)
      (get ds text-col)))))
  ([ds text-col bow-col]
   (count-vectorize ds text-col bow-col {:text->bow-fn  default-text->bow})))
   
  

(defn ->vocabulary-top-n [bows n]
  "Takes top-n most frequent tokens as vocabulary"
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
                    :index->vocab-map (clojure.set/map-invert vocab->index-map)}
                    
        vocab->index-map (:vocab->index-map vocabulary)
        ds
        (ds/add-or-update-column
         ds
         (ds/new-column
          indices-col
          (ppp/ppmap-with-progress
           "bow->sparse"
           1000
           #(bow->sparse-fn % vocab->index-map)
           (get ds bow-col))))]
    {:ds ds
     :vocab vocabulary}))
    

(defn bow->sparse [ds bow-col indices-col bow->sparse-fn vocabulary]
  (let [
        vocab->index-map (:vocab->index-map vocabulary)
        ds
        (ds/add-or-update-column
         ds
         (ds/new-column
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


(defn idf [tf-map term bows]
  (let [n-t (count bows)
        n-d (get tf-map term)]
    (Math/log10 (/ n-t n-d))))


(defn tf [term bow]
  (/
   (get bow term 0)
   (apply + (vals bow))))


(defn tfidf [tf-map term bow bows]
  (* (tf term bow)  (idf tf-map term bows)))


(defn bow->tfidf
  "Calculates the tfidf score from bag-of-words (as token frequency maps)
   in column `bow-column` and stores them in a new column `tfid-column` as maps of token->tfidf-score."
  [ds bow-column tfidf-column]
  (let [bows (get ds bow-column)
        tf-map (tf-map bows)
        tfidf-column (ds/new-column tfidf-column
                                    (ppp/ppmap-with-progress
                                     "tfidf" 1000
                                     (fn [bow]
                                       (let [terms (keys bow)
                                             tfidfs
                                             (map
                                              #(tfidf tf-map % bow bows)
                                              terms)]
                                         (zipmap terms tfidfs)))
                                     bows))]
    (ds/add-or-update-column ds tfidf-column)))


(defn freqs->SparseArray [freq-map vocab->index-map]
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
