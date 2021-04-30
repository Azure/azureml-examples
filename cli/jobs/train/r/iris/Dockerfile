FROM rocker/tidyverse:4.0.0-ubuntu18.04
 
# Install python - temporarily still required to be in your image
RUN add-apt-repository ppa:deadsnakes/ppa \
  apt-get update -qq && \
  apt-get install -y python3.8
 
# Create link for python
RUN ln -f /usr/bin/python3 /usr/bin/python

# Install additional R packages
RUN R -e "install.packages(c('optparse'), repos = 'https://cloud.r-project.org/')"
