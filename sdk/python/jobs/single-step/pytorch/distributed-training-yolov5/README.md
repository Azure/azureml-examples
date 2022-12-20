### yolov5-AzureML
#### Training Yolov5 with custom data in Azure Machine Learning using Python SDK v2

This repo contains 

1) **objectdetectionAzureML.ipynb** -  notebook which helps in implementing pytorch distributed training of YoloV5 models using Azure ML services (python sdk v2). The data input is provided as a yaml file which consist of the location of the data in a specific format as required by the "train.py" file from the https://github.com/ultralytics/yolov5. 

2) The data files provided by ultralytics-yolov5 will download the data and arrange them into different folders as required by the "train.py" file.

3) If you have a custom data set or would like to work on an open dataset which is not part of the data files provided by YoloV5, the repo also contains an example data processing python file, **dataprep_yolov5_format.py**, which helps in preparing the data in format required by yolov5. 

4) Data is downloaded from https://cvbp-secondary.z19.web.core.windows.net/datasets/object_detection/odFridgeObjects.zip

5) The zipped folder contains two folders **annotations** and **images**. The annotations are in xml format is converted to the yolo required format and the split into train and validation folders. The details of the flow are provided in the **DataProcessingYolov5Format_example.png**

6) Subsequently a sample yaml file "fridge.yaml" is also provided which will use these datasets for training the model.
  


