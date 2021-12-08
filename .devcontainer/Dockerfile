FROM continuumio/miniconda3
RUN echo "pip install azure-cli" | bash
RUN echo "az extension add -n ml" | bash
RUN echo "pip install azure-ml==0.0.139 --extra-index-url https://azuremlsdktestpypi.azureedge.net/sdk-cli-v2" | bash
RUN pip install jupyterlab && jupyter notebook --generate-config
WORKDIR /root/.jupyter
COPY jupyter_server_config.py jupyter_server_config.py
WORKDIR /root
COPY  start.sh .start.sh

ENTRYPOINT ["/bin/bash",".start.sh"]
