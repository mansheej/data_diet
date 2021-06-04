FROM nvcr.io/nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04

RUN apt-get update && \
    DEBIAN_FRONTEND=DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential \
        ca-certificates \
        cuda-command-line-tools-11-0 \
        curl \
        git \
        locales \
        rsync \
        software-properties-common \
        tree \
        unzip \
        vim \
        wget \
        && \
    sed -i "s/# en_US.UTF-8/en_US.UTF-8/" /etc/locale.gen && \
    locale-gen

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
ENV LC_ALL en_US.UTF-8

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=DEBIAN_FRONTEND=noninteractive apt-get install -y \
        python3.7 \
        python3.7-dev \
        && \
    rm -rf /var/lib/apt/lists/* && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.7 get-pip.py && \
    ln -s /usr/bin/python3.7 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.7 /usr/local/bin/python && \
    pip install --upgrade pip && \
    pip install --upgrade \
        numpy \
        scipy \
        scikit-learn \
        matplotlib \
        seaborn \
        tqdm \
        Cython \
        jupyter \
        tensorflow-gpu \
        tensorflow-datasets \
        jax jaxlib==0.1.65+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html \
        neural-tangents \
        flax \
        && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY . /data_diet
WORKDIR /data_diet

EXPOSE 2222
