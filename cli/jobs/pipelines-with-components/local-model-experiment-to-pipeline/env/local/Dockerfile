#FROM mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04
FROM python:3.8.5

# python installs
COPY env/local/requirements.txt .
RUN pip install -r requirements.txt && rm requirements.txt

COPY data/sample_data /usr/python/data/sample_data
COPY /1_script_run_on_local/src/script.py /usr/python/1_script_run_on_local/src/
WORKDIR /usr/python

# set command
CMD ["bash", "-c", "cd 1_script_run_on_local/src && python script.py && exit"]