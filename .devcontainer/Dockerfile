FROM mcr.microsoft.com/devcontainers/java:1.1.7-21-jdk-bookworm
RUN rm  -f /etc/apt/sources.list.d/yarn.list 

RUN bash -c "bash < <(curl -s https://raw.githubusercontent.com/babashka/babashka/master/install)"
RUN bash -c "bash < <(curl -s https://raw.githubusercontent.com/clojure-lsp/clojure-lsp/master/install)"


