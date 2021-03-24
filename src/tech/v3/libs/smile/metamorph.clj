(ns tech.v3.libs.smile.metamorph

  (:require
   [tech.v3.libs.smile.nlp :as nlp]
   [tech.v3.dataset :as ds]
   [pppmap.core :as ppp]
   ))


(defn count-vectorize
  "Transforms the text column `text-col` into a map of token frequencies in column
  `bow-col`

  metamorph                            |.
  -------------------------------------|---------
  Behaviour in mode :fit               |normal
  Behaviour in mode :transform         |normal
  Reads keys from ctx                  |none
  Writes keys to ctx                   |none
  "

  ([text-col bow-col options]
   (fn [ctx]
     (assoc ctx :metamorph/data
            (nlp/count-vectorize (:metamorph/data ctx) text-col bow-col options))))
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
  Writes keys to ctx                   |:tech.v3.libs.smile.metamorph/count-vectorize-vocabulary

  "


  [bow-col indices-col bow->sparse-fn options]
  (def bow-col bow-col)
  (def indices-col indices-col)
  (def bow->sparse-fn bow->sparse-fn)
  (def options options)

  (fn [{:metamorph/keys [mode data] :as ctx}]
    (def data data)
    (def mode mode)
    (case mode
      :fit
      (let [
            {:keys [ds vocab]}
            (nlp/bow->sparse-and-vocab data
                                       bow-col indices-col
                                       bow->sparse-fn
                                       options)
             _ (def ds ds)
             _ (def vocab vocab)
            ]
        (assoc ctx :metamorph/data ds
               ::count-vectorize-vocabulary vocab
               ))
      :transform
      (do
        (def ctx ctx)
        (def data data)
        (let [{:keys [ds vocab]}
              (nlp/bow->sparse data bow-col indices-col bow->sparse-fn (::count-vectorize-vocabulary ctx))]
          (assoc ctx :metamorph/data ds))
        )


      )))

(defn bow->sparse-array
  "Converts a bag-of-word column `bow-col` to sparse indices column `indices-col`,
   as needed by the Maxent model.
   `vocab size` is the size of vocabluary used, sorted by token frequency

  metamorph                            |.
  -------------------------------------|---------
  Behaviour in mode :fit               |normal
  Behaviour in mode :transform         |normal
  Reads keys from ctx                  |none
  Writes keys to ctx                   |:tech.v3.libs.smile.metamorph/count-vectorize-vocabulary

  "
  ([bow-col indices-col options]
   (bow->something-sparse bow-col indices-col nlp/bow->sparse-indices options)

   )
  ([bow-col indices-col]
   (bow->something-sparse bow-col indices-col nlp/bow->sparse-indices {})
   ))


(defn bow->SparseArray
  "Converts a bag-of-word column `bow-col` to sparse indices column `indices-col`,
   as needed by the discrete naive bayes model. `vocab size` is the size of vocabluary used, sorted by token frequency

  metamorph                            |.
  -------------------------------------|---------
  Behaviour in mode :fit               |normal
  Behaviour in mode :transform         |normal
  Reads keys from ctx                  |none
  Writes keys to ctx                   |:tech.v3.libs.smile.metamorph/count-vectorize-vocabulary

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
  [bow-column tfidf-column]
  (fn [ctx]
    (assoc ctx :metamorph/data
           (nlp/bow->tfidf
            (:metamorph/data ctx)
            bow-column
            tfidf-column))))
