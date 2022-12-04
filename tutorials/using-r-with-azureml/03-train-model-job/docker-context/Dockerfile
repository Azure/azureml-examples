FROM rocker/tidyverse:latest

# Install python
RUN apt-get update -qq && \
 apt-get install -y python3-pip tcl tk libz-dev libpng-dev

RUN ln -f /usr/bin/python3 /usr/bin/python
RUN ln -f /usr/bin/pip3 /usr/bin/pip
RUN pip install -U pip

# Install azureml-mlflow
RUN pip install azureml-mlflow
RUN pip install mlflow

# Create link for python
RUN ln -f /usr/bin/python3 /usr/bin/python

# Install R packages required for logging with mlflow (these are necessary)
RUN R -e "install.packages(c('mlflow'), repos = 'https://cloud.r-project.org/')"
RUN R -e "install.packages(c('carrier'), repos = 'https://cloud.r-project.org/')"
RUN R -e "install.packages(c('optparse'), repos = 'https://cloud.r-project.org/')"
RUN R -e "install.packages(c('tcltk2'), repos = 'https://cloud.r-project.org/')"

# Install any additional R packages you need for your script
RUN R -e "install.packages(c('tsibble'), repos = 'https://cloud.r-project.org/')"
RUN R -e "install.packages(c('fable'), repos = 'https://cloud.r-project.org/')"
RUN R -e "install.packages(c('janitor'), repos = 'https://cloud.r-project.org/')"
RUN R -e "install.packages(c('qs'), repos = 'https://cloud.r-project.org/')"
