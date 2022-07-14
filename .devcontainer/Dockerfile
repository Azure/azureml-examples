FROM continuumio/miniconda3
RUN apt-get update && apt-get install -y curl
RUN apt update
RUN curl -sL https://deb.nodesource.com/setup_14.x | bash
RUN apt install nodejs -y
RUN npm install -g autorest@3.0.6339
RUN apt-get install git -y
RUN echo "pip install debugpy" | bash
RUN echo "pip install flake8" | bash
