#FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04
FROM ubuntu:16.04

# ========== Anaconda for Python3 ==========
# https://hub.docker.com/r/continuumio/anaconda3/~/dockerfile/

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion dbus

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh


ENV PATH /opt/conda/bin:$PATH

# Install basic dependencies
RUN apt-get install -y cmake build-essential pkg-config libpython3-dev \
    libboost-python-dev libboost-dev python3 zlib1g-dev


RUN apt-get install qt4-dev-tools -y

RUN apt-get install -y libboost-filesystem-dev libboost-system-dev flex

WORKDIR /root/code/
RUN echo 'y'
RUN git clone -b ec2 https://github.com/nosyndicate/pytorchrl.git
RUN git clone https://github.com/nosyndicate/platform.git


WORKDIR /root/code/pytorchrl/

RUN conda create -y -n pytorchrl
ENV PYTHONPATH /root/code/pytorchrl:$PYTHONPATH

# Need this to use the python executable in conda env, not the
# default one. Otherwise, the pip packages won't be found.
ENV PATH /opt/conda/envs/pytorchrl/bin:$PATH


RUN echo "source activate pytorchrl" >> /root/.bashrc
RUN conda env update -f environment.yml

WORKDIR /root/code/platform/
RUN /opt/conda/envs/pytorchrl/bin/pip install .
WORKDIR /root/code/pytorchrl

# Need this to pass the last check in config.py.
# Otherwise, the process is exiting.
ENV CIRCLECI=true

ENV BASH_ENV /root/.bashrc