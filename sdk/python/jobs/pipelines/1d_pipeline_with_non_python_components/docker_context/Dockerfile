FROM rocker/tidyverse:4.3.3

# Install python
RUN apt-get update -qq && \
 apt-get install -y python3-pip tcl tk libz-dev libpng-dev curl openssh-server

RUN ln -f /usr/bin/python3 /usr/bin/python
RUN ln -f /usr/bin/pip3 /usr/bin/pip
RUN pip install -U pip

# Install azureml-mlflow
# Change back when azureml-mlflow fix the No module named 'dateutil' bug
RUN pip install azureml-mlflow==1.42.0 "ipykernel~=6.0"

# Create link for python
RUN ln -f /usr/bin/python3 /usr/bin/python

# Loading each package makes a failed installation fail the image build.
RUN Rscript -e "packages <- c('mlflow', 'carrier', 'optparse', 'tcltk2', 'httr', 'later', 'rpart'); install.packages(packages, repos = 'https://cloud.r-project.org/'); invisible(lapply(packages, library, character.only = TRUE))"