(ns scicloj.ml.smile.metamorph

  (:require
   [scicloj.ml.smile.nlp :as nlp]
   [tech.v3.dataset :as ds]
   [pppmap.core :as ppp]
   [scicloj.ml.smile.malli :as malli]))


   


(defn count-vectorize
  "Transforms the text column `text-col` into a map of token frequencies in column
  `bow-col`

  `options` can be any of

  `text->bow-fn` A functions which takes as input a

  metamorph                            |.
  -------------------------------------|---------
  Behaviour in mode :fit               |normal
  Behaviour in mode :transform         |normal
  Reads keys from ctx                  |none
  Writes keys to ctx                   |none
  "

  ([text-col bow-col options]
   (malli/instrument-mm
    (fn [ctx]
      (assoc ctx :metamorph/data
             (nlp/count-vectorize (:metamorph/data ctx) text-col bow-col options)))))
  ([text-col bow-col]
   (count-vectorize text-col bow-col {})))


(defn bow->something-sparse
  "Converts a bag-of-word column `bow-col` to a sparse data column `indices-col`.
   The exact transformation to the sparse representtaion is given by `bow->sparse-fn`

  metamorph                            |.
  -------------------------------------|---------
  Behaviour in mode :fit               |normal
  Behaviour in mode :transform         |normal
  Reads keys from ctx                  |none
  Writes keys to ctx                   |:scicloj.ml.smile.metamorph/bow->sparse-vocabulary

  "


  [bow-col indices-col bow->sparse-fn options]
  (malli/instrument-mm
   (fn [{:metamorph/keys [mode data] :as ctx}]
     (case mode
       :fit
       (let [
             {:keys [ds vocab]}
             (nlp/bow->sparse-and-vocab data
                                        bow-col indices-col
                                        bow->sparse-fn
                                        options)]

         (assoc ctx :metamorph/data ds
                ::bow->sparse-vocabulary vocab))
               
       :transform
       (let [{:keys [ds vocab]}
             (nlp/bow->sparse data bow-col indices-col bow->sparse-fn (::bow->sparse-vocabulary ctx))]
         (assoc ctx :metamorph/data ds))))))
        


      

(defn bow->sparse-array
  "Converts a bag-of-word column `bow-col` to sparse indices column
  `indices-col`,   as needed by the Maxent model.
  `Options` can be of:

  `create-vocab-fn` A function which converts the bow map to a list of tokens.
                    Defaults to scicloj.ml.smile.nlp/create-vocab-all


  The sparse data is represented as `primitive int arrays`,
  of which entries are the indices against the vocabulary
  of the present tokens.

  metamorph                            |.
  -------------------------------------|---------
  Behaviour in mode :fit               |normal
  Behaviour in mode :transform         |normal
  Reads keys from ctx                  |none
  Writes keys to ctx                   |:scicloj.ml.smile.metamorph/bow->sparse-vocabulary

  "
  ([bow-col indices-col options]
   (bow->something-sparse bow-col indices-col nlp/bow->sparse-indices options))

   
  ([bow-col indices-col]
   (bow->something-sparse bow-col indices-col nlp/bow->sparse-indices {})))
   


(defn bow->SparseArray
  "Converts a bag-of-word column `bow-col` to sparse indices column `indices-col`,
   as needed by the discrete naive bayes model.

  `Options` can be of:

  `create-vocab-fn` A function which converts the bow map to a list of tokens.
                    Defaults to scicloj.ml.smile.nlp/create-vocab-all

  The sparse data is represented as `smile.util.SparseArray`.

  metamorph                            |.
  -------------------------------------|---------
  Behaviour in mode :fit               |normal
  Behaviour in mode :transform         |normal
  Reads keys from ctx                  |none
  Writes keys to ctx                   |:scicloj.ml.smile.metamorph/bow->sparse-vocabulary

  "
  ([bow-col indices-col options]
   (bow->something-sparse bow-col indices-col nlp/freqs->SparseArray options))
  ([bow-col indices-col]
   (bow->something-sparse bow-col indices-col nlp/freqs->SparseArray {})))

(defn bow->tfidf
  "Calculates the tfidf score from bag-of-words (as token frequency maps)
   in column `bow-column` and stores them in a new column `tfid-column` as maps of token->tfidf-score.


  metamorph                            |.
  -------------------------------------|---------
  Behaviour in mode :fit               |normal
  Behaviour in mode :transform         |normal
  Reads keys from ctx                  |none
  Writes keys to ctx                   |none
  "
  ([bow-column tfidf-column]
   (malli/instrument-mm
    (fn [ctx]
      (assoc ctx :metamorph/data
             (nlp/bow->tfidf
              (:metamorph/data ctx)
              bow-column
              tfidf-column)))))
  ([bow-column tfidf-column options]
   (bow->tfidf bow-column tfidf-column {})))

(defn tfidf->dense-array
   "Converts the sparse tfidf map based representation into
  dense double arrays


  metamorph                            |.
  -------------------------------------|---------
  Behaviour in mode :fit               |normal
  Behaviour in mode :transform         |normal
  Reads keys from ctx                  |none
  Writes keys to ctx                   |none
  "
  [tfidf-column dense-column]
  (malli/instrument-mm
   (fn [ctx]
     (assoc ctx :metamorph/data
            (nlp/tfidf->dense-array
             (:metamorph/data ctx)
             tfidf-column
             dense-column)))))
