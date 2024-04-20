(ns scicloj.ml.smile.loglikelyhood-test
  (:require [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as ds-mod]
            [scicloj.ml.smile.regression]
            [scicloj.metamorph.ml :as ml]
            [scicloj.metamorph.ml.metrics :as metric]
            [scicloj.metamorph.ml.toydata :as toydata]
            [fastmath.core :as fm]
            [clojure.test :refer (deftest is)]))

;; R
;; https://jmsallan.netlify.app/blog/summary-staitstics-in-linear-regression/
;; mod4 <- lm(mpg ~ ., mtcars)
;; mod	r.squared	adj.r.squared	sigma	statistic	p.value	df	logLik	            AIC	BIC	   deviance	df.residual	nobs
;; mod4	0.869	0.807	         2.650	13.932	     0	         10	-69.855	163.710	181.299	147.494	21	             32

(deftest bic_aic

  (let [mtcars
        (->
         (toydata/mtcars-ds)
         (ds/drop-columns [:model])
         (ds-mod/set-inference-target :mpg))

        m
        (ml/train mtcars {:model-type :smile.regression/ordinary-least-square})
        prediction-ds (ml/predict mtcars m)]

    (is (= 181.29864126807638
           (metric/BIC m mtcars prediction-ds)))
    (is (= 163.70981043447966
           (metric/AIC m mtcars prediction-ds)))))





;; =>

;; =>
