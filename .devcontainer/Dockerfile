
FROM ubuntu:18.04
# System packages 
RUN apt-get update && apt-get install -y curl
# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda
# Python packages from conda
RUN conda install -c anaconda -y python=3.7
RUN conda install -c anaconda -y pip 
RUN conda init bash

#Choose your version of azcli
RUN echo "pip install azure-cli" | bash
#Choose your version of ml cli
RUN echo "az extension add -n ml" | bash 
RUN echo "pip install azure-ml==0.0.139 --extra-index-url https://azuremlsdktestpypi.azureedge.net/sdk-cli-v2" | bash


RUN pip install jupyterlab && jupyter notebook --generate-config
WORKDIR /root/.jupyter
COPY jupyter_server_config.py jupyter_server_config.py
WORKDIR /root
COPY  .start.sh .start_jupyter.sh

# This is to support updating conda environment with user supplied packages with environment.yml
COPY environment.yml* no_op.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /miniconda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp
