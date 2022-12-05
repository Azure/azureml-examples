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
RUN R -e "install.packages('mlflow', dependencies = TRUE, repos = 'https://cloud.r-project.org/')"
RUN R -e "install.packages('carrier', dependencies = TRUE, repos = 'https://cloud.r-project.org/')"
RUN R -e "install.packages('optparse', dependencies = TRUE, repos = 'https://cloud.r-project.org/')"
RUN R -e "install.packages('tcltk2', dependencies = TRUE, repos = 'https://cloud.r-project.org/')"

# Install any additional R packages you need for your script
RUN R -e "install.packages('tsibble', dependencies = TRUE, repos = 'https://cloud.r-project.org/')"
RUN R -e "install.packages('fable', dependencies = TRUE, repos = 'https://cloud.r-project.org/')"
RUN R -e "install.packages('janitor', dependencies = TRUE, repos = 'https://cloud.r-project.org/')"
RUN R -e "install.packages('qs', dependencies = TRUE, repos = 'https://cloud.r-project.org/')"
#RUN R -e "install.packages(c('feasts'), repos = 'https://cloud.r-project.org/')"
#RUN R -e "install.packages(c('urca'), repos = 'https://cloud.r-project.org/')"
