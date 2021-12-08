FROM continuumio/miniconda3
RUN echo "pip install azure-cli" | bash
RUN echo "az extension add -n ml" | bash
RUN echo "pip install azure-ml==0.0.139 --extra-index-url https://azuremlsdktestpypi.azureedge.net/sdk-cli-v2" | bash
RUN pip install jupyterlab && jupyter notebook --generate-config
EXPOSE 8888
WORKDIR /root/.jupyter
COPY jupyter_notebook_config.py jupyter_notebook_config.py
WORKDIR /root
COPY  wrapper.sh wrapper.sh

CMD  ["/bin/bash","wrapper.sh"]
