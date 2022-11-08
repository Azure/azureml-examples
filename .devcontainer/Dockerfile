
FROM continuumio/miniconda3

RUN pip install jupyterlab && jupyter notebook --generate-config
WORKDIR /root/.jupyter
COPY jupyter_server_config.py jupyter_server_config.py
WORKDIR /root
COPY  .start.sh .start_jupyter.sh

# This is to support updating conda environment with user supplied packages with environment.yml
COPY environment.yml* no_op.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /miniconda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp
