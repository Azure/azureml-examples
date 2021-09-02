## How to use the PyTorch to ONNX converter utility tool

Functionality derived from: https://thenewstack.io/tutorial-train-a-deep-learning-model-in-pytorch-and-export-it-to-onnx/ and https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html

### Prerequisites
You need to install ONNX, ONNX Runtime and Torch libraries

<pre style="background-color:rgba(0, 0, 0, 0.0470588)"><font size="2">pip install onnx
pip install onnxruntime
pip install torch
</pre>

To convert a PyTorch Model to ONNX with this utility two files need to be provided:
* A file containing the model itself (weights)
* A python file with the definition of the model and the nature of its inputs and outputs

The following variables define the files and their locations, as well as the name of the converted ONNX model

<pre style="background-color:rgba(0, 0, 0, 0.0470588)"><font size="2">pytorch_model_filename='net.pt'

pytorch_model_url='https://github.com/mreyesgomez/model_hub/blob/main/net.pt'

python_model_definition_filename='net_model.py'

python_model_definition_url='https://github.com/Azure/azureml-examples/blob/triton-bert-perf/cli/endpoints/online/triton/batching/modelanalyzer/utils/net_model.py'

onnx_model_filename='net.onnx'</pre>

### Model Definition File
The model definition file is a python file where the structure of the model is defined, the nature of the model expected inputs are also required to be provided, optionally the names of the model inputs and outputs can be provided to ease the graphic view/analysis of the resulting ONNX model

Please take a look at the [sample](https://github.com/Azure/azureml-examples/blob/triton-bert-perf/cli/endpoints/online/triton/batching/modelanalyzer/utils/net_model.py) used in this example

For more referenes on the nature of this file please look at: https://thenewstack.io/tutorial-train-a-deep-learning-model-in-pytorch-and-export-it-to-onnx/ and https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html


### Downloading the material
Create a working directory

<pre style="background-color:rgba(0, 0, 0, 0.0470588)"><font size="2">mkdir $HOME/pytorch_to_onnx_wrk_dir

cd $HOME/pytorch_to_onnx_wrk_dir

</pre>

Download utility and files to working directory

<pre style="background-color:rgba(0, 0, 0, 0.0470588)"><font size="2">wget -O convert_pytorch_to_onnx.py https://github.com/Azure/azureml-examples/blob/triton-bert-perf/cli/endpoints/online/triton/batching/modelanalyzer/utils/convert_pytorch_to_onnx.py?raw=true

wget -O ${pytorch_model_filename} ${pytorch_model_url}?raw=true

wget -O ${python_model_definition_filename} ${python_model_definition_url}?raw=true
</pre>

### Executing the utility
Retrieving the model definition library name from the model definition file:
<pre style="background-color:rgba(0, 0, 0, 0.0470588)"><font size="2">model_info_lib="${python_model_definition_filename%.*}"
</pre>

Modify the utility file to import the model definition file:
<pre style="background-color:rgba(0, 0, 0, 0.0470588)"><font size="2">sed -i "s/model_info_lib/${model_info_lib}/g" convert_pytorch_to_onnx.py</pre>

Run the utility:
<pre style="background-color:rgba(0, 0, 0, 0.0470588)"><font size="2">python convert_pytorch_to_onnx.py --input $pytorch_model_filename --output $onnx_model_filename</pre>
