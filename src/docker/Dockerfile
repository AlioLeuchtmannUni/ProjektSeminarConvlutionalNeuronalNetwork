#FROM nvidia/cuda:12.0.0-runtime-ubuntu22.04
FROM nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu20.04
# https://hub.docker.com/r/nvidia/cuda for more info

# Install OpenJDK-18
# --allow-unauthenticated --allow-insecure-repositories to solve cuda repo not signed
RUN apt-get update --allow-unauthenticated --allow-insecure-repositories && \
    apt-get install -y openjdk-11-jdk && \
    apt-get install -y ant && \
    apt-get clean;

# Fix certificate issues
RUN apt-get update --allow-unauthenticated --allow-insecure-repositories && \
    apt-get install ca-certificates-java && \
    apt-get clean && \
    update-ca-certificates -f;

# Setup JAVA_HOME -- useful for docker commandline
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64/
RUN export JAVA_HOME