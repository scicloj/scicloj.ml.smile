(ns build
  (:refer-clojure :exclude [test])
  (:require [clojure.tools.build.api :as b] ; for b/git-count-revs
            [org.corfield.build :as bb]))

(def lib 'org.scicloj/scicloj.ml.smile)
; alternatively, use MAJOR.MINOR.COMMITS:
(def version (format "7.4.3"))
(def class-dir "target/classes")
(def basis (b/create-basis {:project "deps.edn"}))
(def jar-file (format "target/%s-%s.jar" (name lib) version))



(defn compile [_]
  (b/javac {:src-dirs ["java"]
            :class-dir class-dir
            :basis basis
            :javac-opts ["-source" "8" "-target" "8"]}))



(defn test "Run the tests." [opts]
  (-> opts
      (assoc :lib lib :version version
             :aliases [:run-tests])
      (bb/run-tests)))


  

(defn jar [_]
  (compile nil)
  (b/write-pom {:class-dir class-dir
                :lib lib
                :version version
                :basis basis
                :src-pom "template/pom.xml"
                :scm {:connection "scm:git:https://github.com/scicloj/scicloj.ml.smile.git"
                      :url "https://github.com/scicloj/scicloj.ml.smile"}
                :src-dirs ["src"]})
  (b/copy-dir {:src-dirs ["src" "resources"]
               :target-dir class-dir})
  (b/jar {:class-dir class-dir
          :jar-file jar-file}))

(defn ci "Run the CI pipeline of tests (and build the JAR)." [opts]
  (compile nil)
  (-> opts
      (assoc :lib lib :version version
             :aliases [:run-tests])
      (bb/run-tests)
      (bb/clean)
      (jar)))


(defn install "Install the JAR locally." [opts]
  (-> opts
      (assoc :lib lib :version version)
      (bb/install)))

(defn deploy "Deploy the JAR to Clojars." [opts]
  (-> opts
      (assoc :lib lib :version version)
      (bb/deploy)))
