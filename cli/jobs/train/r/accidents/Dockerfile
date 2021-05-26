FROM rocker/tidyverse:4.0.0-ubuntu18.04
 
# Install python (this is still required by AzureML, but this requirement will soon be removed )
RUN apt-get update -qq && \
 apt-get install -y python3-pip
RUN ln -f /usr/bin/python3 /usr/bin/python
RUN ln -f /usr/bin/pip3 /usr/bin/pip
RUN pip install -U pip

# install azureml-base and libfuse (this is required for data access in local execution -- this requirement will also be removed soon)
RUN sudo apt-get install -y libfuse-dev
RUN pip install azureml-dataprep azureml-core

# Install additional R packages 
RUN R -e "install.packages(c('optparse'), repos = 'https://cloud.r-project.org/')"

