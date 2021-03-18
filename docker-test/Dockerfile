FROM nvidia/cuda:11.2.2-devel-ubuntu18.04

RUN apt-get update
RUN apt-get install -y python3 python3-pip
RUN pip3 install torch torchvision

WORKDIR /work

COPY train.py /work/

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs