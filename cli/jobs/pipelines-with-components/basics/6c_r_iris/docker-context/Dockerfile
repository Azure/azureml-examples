FROM rocker/tidyverse:4.0.0-ubuntu18.04

# Install python
RUN apt-get update -qq && \
 apt-get install -y python3-pip tcl tk libz-dev libpng-dev curl

RUN ln -f /usr/bin/python3 /usr/bin/python
RUN ln -f /usr/bin/pip3 /usr/bin/pip
RUN pip install -U pip

# Install azureml-mlflow
# Change back when azureml-mlflow fix the No module named 'dateutil' bug
RUN pip install azureml-mlflow==1.42.0

# Create link for python
RUN ln -f /usr/bin/python3 /usr/bin/python

# Use a fixed snapshot so package updates do not drop support for R 4.0.0.
RUN Rscript -e "install.packages(c('mlflow', 'carrier', 'optparse', 'tcltk2'), repos = 'https://packagemanager.posit.co/cran/2022-06-01')"

# install.packages() can return success even when a package fails to install.
RUN Rscript -e "invisible(lapply(c('optparse', 'carrier', 'rpart', 'mlflow', 'httr', 'later', 'tcltk2'), library, character.only = TRUE))"
