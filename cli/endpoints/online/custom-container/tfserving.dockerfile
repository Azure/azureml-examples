FROM tensorflow/serving

# ENV MODEL_BASE_PATH=/var/azureml-app/azureml-models/tfserving-mounted/1
# ENV MODEL_NAME=half_plus_two2
# CMD ["tensorflow_model_server", "--port=5000", "--rest_api_port=5001"]
# COPY ./half_plus_two/ /models/model