FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime
RUN apt-get update && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y vim-tiny
RUN pip install docrep scipy 
ENV PYTHONPATH "${PYTHONPATH}:/brevitas"
