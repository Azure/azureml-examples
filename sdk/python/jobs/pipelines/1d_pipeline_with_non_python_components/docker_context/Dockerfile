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

# Install additional R packages
RUN R -e "install.packages(c('mlflow'), repos = 'https://cloud.r-project.org/')"
RUN R -e "install.packages(c('carrier'), repos = 'https://cloud.r-project.org/')"
RUN R -e "install.packages(c('optparse'), repos = 'https://cloud.r-project.org/')"
# R 4.0.0 compatible version of tcltk2
RUN R -e "install.packages('https://cran.r-project.org/src/contrib/Archive/tcltk2/tcltk2_1.2-11.tar.gz', repos=NULL, type='source');"