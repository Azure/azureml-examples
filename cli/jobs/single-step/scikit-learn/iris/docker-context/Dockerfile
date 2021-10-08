FROM mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04

# python installs
COPY requirements.txt .
RUN pip install -r requirements.txt && rm requirements.txt

# set command
CMD ["bash"]
