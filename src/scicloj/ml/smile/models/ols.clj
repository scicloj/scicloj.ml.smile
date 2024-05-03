(ns scicloj.ml.smile.models.ols
  (:require
   [fastmath.core :as m]
   [fastmath.random :as r]
   [fastmath.stats :as stats]
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.metrics :as metrics]
   [scicloj.ml.smile.models.general]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.column :as ds-col]
   [tech.v3.dataset.modelling :as ds-mod]
   [tech.v3.dataset.tensor :as ds-tens]
   [tech.v3.tensor :as dtt])
  (:import
   (smile.regression LinearModel)))


(def method-table
  {
   :qr "qr"
   :svd "svd"})




(defn standard-metric-maps [model]
  (let [
        ols (ml/thaw-model model)
        y (-> model :model-data :label-ds ds/columns first)
        sample-size (-> model :model-data :sample-size)
        y_hat (seq (.fittedValues ols))]
    {:r.squared (.RSquared ols)
     :adj.r.squared (.adjustedRSquared ols)
     :df (.df ols)
     :log-lik (ml/loglik model y y_hat)
     :bic (metrics/BIC model y y_hat sample-size (count  (:feature-columns model)))
     :aic (metrics/AIC model y y_hat (count  (:feature-columns model)))
     :p.value (.pvalue ols)
     :rss (.RSS ols)
     :rmse (stats/rmse y y_hat)
     :mse  (stats/mse y y_hat)}))

(defn predict [thawed-model ds]
  (let [ds-with-bias
        (ds/append-columns
         (ds/new-dataset
          [(ds-col/new-column :intercept (repeat (ds/row-count ds) 1))])
         (ds/columns ds))]
    (scicloj.ml.smile.models.general/predict-linear-model thawed-model ds-with-bias)))

(defn explain
  [thawed-model {:keys [feature-columns]} _options]
  (let [^LinearModel model thawed-model
        weights (seq (.coefficients model))]
    {:bias (first weights)
     :coefficients (->> (map vector
                             feature-columns
                             (rest weights))
                        (sort-by (comp second) >))}))

(defn log-likelihood
  [y yhat]
  (let [sigma (-> (stats/rss y yhat)
                  (/ (count y))
                  (m/sqrt))
        ldnorm (map (fn [vy vyhat]
                      (let [d (r/distribution :normal {:mu vyhat :sd sigma})]
                        (r/lpdf d vy))) y yhat)]
    (stats/sum ldnorm)))

(defn augment [model dataset]

    (let [fitted
          (->
           (ml/thaw-model model)
           (.fittedValues))

          residuals
          (->
           (ml/thaw-model model)
           (.residuals))]

      (-> dataset
          (ds/add-column (ds/new-column :.fitted fitted))
          (ds/add-column (ds/new-column :.resid residuals)))))

(defn tidy [model]
  (let [
        ttest
        (->
         (ml/thaw-model model)
         .ttest
         tech.v3.tensor/->tensor
         ds-tens/tensor->dataset)]

    (->
     (ds/->dataset
      {
       :term (concat [:intercept] (:feature-columns model))})
     (ds/append-columns ttest)
     (ds/rename-columns {0 :estimate
                         1 :std.error
                         2 :statistic
                         3 :p.value}))))

(defn glance [model]
    (ds/->dataset
     (standard-metric-maps model)))



(defn linear-regression
  "Does a linear regression with the given tech.ml.dataset.

  It should have the inference-target column marked as such.
  It uses model :smile.regression/ordinary-least-square

  It returns a result map with
     - various model diagnostics
     - fitted-values
     - the model as map
    ...
    ...


"
  ([ds options]

   (let [
         inference-target (first  (ds-mod/inference-target-column-names ds))
         m
         (ml/train ds (assoc options :model-type :smile.regression/ordinary-least-square))

         ols (ml/thaw-model m)

         y (seq (get ds inference-target))
         y_hat (seq (.fittedValues ols))

         standard-metrics-map (standard-metric-maps m)]
     (assoc standard-metrics-map


            :coefficients (seq (.coefficients ols))
            :intercept (.intercept ols)
            :resid (seq (.residuals ols))
            :fitted-values y_hat

            :f-test (.ftest ols)
            :error (.error ols)
            :t-test (.ttest ols)


            :model m)))


  ([ds] (linear-regression ds {})))
