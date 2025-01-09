FROM rocker/tidyverse:latest

# Install python
RUN apt-get update -qq && \
 apt-get install -y python3-pip tcl tk libz-dev libpng-dev && \
 apt-get install -y python3-venv

RUN /usr/bin/python3 -m venv /usr/venv/

RUN ln -f /usr/venv/python3 /usr/bin/python
RUN ln -f /usr/venv/pip3 /usr/bin/pip

# Install azureml-mlflow

RUN pip install azureml-mlflow
RUN pip install mlflow

# Install additional R packages
RUN R -e "install.packages(c('mlflow'), repos = 'https://cloud.r-project.org/')"
RUN R -e "install.packages(c('carrier'), repos = 'https://cloud.r-project.org/')"
RUN R -e "install.packages(c('optparse'), repos = 'https://cloud.r-project.org/')"
RUN R -e "install.packages(c('tcltk2'), repos = 'https://cloud.r-project.org/')"
