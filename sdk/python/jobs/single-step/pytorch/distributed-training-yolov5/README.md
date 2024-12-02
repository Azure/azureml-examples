### yolov5-AzureML
#### Training Yolov5 with custom data in Azure Machine Learning using Python SDK v2

In this sample, object detection model training with Fridge Ojects data is demonstrated using the Yolov5 models. To limit the size of the repo, we have only included one sample **yaml** file in "data" folder. If you would like to test with yaml files provided by yolov5, then you may clone this from  https://github.com/ultralytics/yolov5. 

#### Prerequisites

a. Download data from https://cvbp-secondary.z19.web.core.windows.net/datasets/object_detection/odFridgeObjects.zip
b. The zipped folder contains two folders **annotations** and **images**. 
c.The annotations are in xml format is converted to the yolo required format and the split into train and validation folders using  **dataprep_yolov5_format.py** . This will create a new folder structure **datasets/fridgedata** in **yolov5** , which will contain image and folder structure as required by the **train.py**. The details of the flow are provided in the **DataProcessingYolov5Format_example.png**. 
d. A sample yaml file "fridge.yaml" is also provided in **data** folder, which will use the data from **datasets** folder for training the model.
e. Run the notebook **objectdetectionAzureML.ipynb** 



  


