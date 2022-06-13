import json
import time
import torch
import torch.nn.functional as F
import os
import torch.nn as nn
from collections import OrderedDict
# from azureml.monitoring import ModelDataCollector

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(128 * 6 * 6, 120)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 6 * 6)
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Called when the deployed service starts
def init():
    global model
    global device

    # Get the path where the deployed model can be found.
    model_filename = 'model/torch_model.pt'
    model_file_path = os.path.join(os.environ['AZUREML_MODEL_DIR'], model_filename)
    msi_endpoint = os.environ.get("MSI_ENDPOINT", None)
    msi_secret = os.environ.get("MSI_SECRET", None)
    print("msi_endpoint:", msi_endpoint,"msi_secret:", msi_secret)
    global inputs_dc, prediction_dc
    # inputs_dc = ModelDataCollector("torch-model",designation="input")
    # prediction_dc = ModelDataCollector("torch-model", designation="predictions")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = Net()
    model.to(device)
    saved_state_dict = torch.load(model_file_path, map_location=device)

    cleaned_dict = OrderedDict()
    for key, value in saved_state_dict.items():
        if key.startswith("module"):
            first_dot = key.find(".")
            key = key[first_dot + 1:]

        cleaned_dict[key] = value

    model.load_state_dict(cleaned_dict)
    model.eval()


# Handle requests to the service
def run(data):
    try:
        # Pick out the text property of the JSON request.
        # This expects a request in the form of {"text": "some text to score for sentiment"}

        start_at = time.time()
        inputs = json.loads(data)
        img_data_list = inputs["instances"]
        inputs = torch.tensor(img_data_list).to(device) 

        with torch.no_grad():
            predictions = model(inputs)
        info = {
            "input": data,
            "output": predictions.numpy().tolist()
            }
        print(json.dumps(info))
        # inputs_dc.collect(data) #this call is saving our input data into Azure Blob
        # prediction_dc.collect(predictions) #this call is saving our prediction data into Azure Blob
        return {"predicts": predictions.numpy().tolist(),
                "elapsed_time": time.time() - start_at}
    except Exception as e:
        error = str(e)
        print(error)
        raise e
