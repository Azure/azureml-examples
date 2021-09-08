# Finding the optimal model configuration for inference deployment using the NVIDIA Triton Model Analyzer

### Objective: Use the Triton Model Analyzer to find the optimal configuration for a Deep Learning Model (Using as example a Bert ONNX Model)

#### Triton Model Analyzer
The Model Analyzer is a suite of tools that helps users select the optimal model configuration that maximizes performance in Triton. The Model Analyzer benchmarks model performance by measuring throughput (inferences/second) and latency under varying client loads. It can then generate performance charts to help users make tradeoff decisions around throughput and latency, as well as identify the optimal batch size, through the use of dynamic batching, as well as optimal concurrency values to use for the best performance.

By measuring GPU memory utilization, the Model Analyzer also optimizes hardware usage by informing how many models can be loaded onto a GPU and the exact amount of hardware needed to run your models based on performance or memory requirements.

## Compute Requirements	
* Machine with at least one Volta GPU or above
* OS: Linux 18.04 LTS or above
* Nvidia Driver (minimum version 450.57)
* Docker CE 19.03 or above
* +64 GB of disk space

The instructions and results discussed in this document were obtained using the following Azure specifications
* VM_Size: Standard_NC6s_v3 (one 16GiB V100 card)
* Resized 64 GB Disk (default is 32 GB)
* Base Image: Azure Market Place, NVIDIA Image for AI using GPUs - v21.04.1


The "NVIDIA Image for AI using GPUs" available for free in the Azure Market Place has all the prerequisites needed to run the Model Analyzer

<img src="imgs/NvidiaImage.png" width="200">

## 1.- Environment Set Up (One time)
In this section all the code required to pre-process the models is installed in addition to installing and building the Model Analyzer bits. 

<pre style="background-color:rgba(227, 147, 125, 0.36)"><font size="2">This section needs to be ran only once, once this set up has been completed the user only needs to run the subsequent sections to analyze a new model</pre>


#### Setting up Triton, Model Analyzer and Model Repository folders

<pre style="background-color:rgba(0, 0, 0, 0.0470588)"><font size="2">mkdir $HOME/triton-inference-server

mkdir $HOME/triton-inference-server/model_analyzer

mkdir $HOME/triton-inference-server/model_analyzer/model_repository</pre>

The steps described in this document, support the analysis of models of the following types:

* PyTorch
* TensorFlow (saved model format)
* ONNX

To unify the procedure, we first convert the model to the ONNX format and the analysis is done from there

#### Installing Dependencies
Assuming the VM was just set up, we first install all the required libraries

<pre style="background-color:rgba(0, 0, 0, 0.0470588)"><font size="2">sudo apt install python3-pip

python3 -m pip install --upgrade pip setuptools wheel

pip3 install onnx onnxruntime

pip3 install sympy

pip3 install packaging

pip3 install tensorflow

pip3 install torch

pip3 install -U tf2onnx

sudo apt-get install cuda-cudart-11-1

python3 -m pip install colored

python3 -m pip install polygraphy --index-url https://pypi.ngc.nvidia.com
</pre>

Updating environmental variables
<pre style="background-color:rgba(0, 0, 0, 0.0470588)"><font size="2">sudo find / -name 'libcudart.so.11.0'

lib_path=$(sudo find / -name 'libcudart.so.11.0')

lib_path=${lib_path%%libcudart.so.11.0*}libcudart.so.11.0

export LD_LIBRARY_PATH=${lib_path}:${LD_LIBRARY_PATH}

sudo find / -name 'polygraphy'

lib_path=$(sudo find / -name 'polygraphy')

lib_path=${lib_path%%polygraphy*}polygraphy

export LD_LIBRARY_PATH=${lib_path}:${LD_LIBRARY_PATH}
</pre>

<b> Please run the remaining steps in a new terminal</b>

#### Installing the ability to link output nodes to input nodes on an ONNX modek
To fully utilize the power of dynamic batching from model analyzer of Triton inference server, the output nodes of the ONNX model need to be shape-inferred to be linked to the input dynamic shape.

<pre style="background-color:rgba(0, 0, 0, 0.0470588)"><font size="2">git clone https://github.com/microsoft/onnxruntime.git $HOME/triton-inference-server/onnxrt</pre>


#### Getting Model Analyzer bits

We first clone the Model Analyzer GitHub [https://github.com/triton-inference-server/model\_analyzer](https://github.com/triton-inference-server/model_analyzer):

<pre style="background-color:rgba(0, 0, 0, 0.0470588)"><font size="2">git clone https://github.com/triton-inference-server/model_analyzer.git
</pre>

Changing to r21.08 branch from main branch

<pre style="background-color:rgba(0, 0, 0, 0.0470588)"><font size="2">cd model_analyzer/

git checkout r21.08
</pre>

#### Building the Model Analyzer container

Building docker container

<pre style="background-color:rgba(0, 0, 0, 0.0470588)"><font size="2">docker build --pull --build-arg=TRITONSDK_BASE_IMAGE="nvcr.io/nvidia/tritonserver:21.08-py3-sdk" --build-arg=BASE_IMAGE="nvcr.io/nvidia/tritonserver:21.08-py3" . -t model-analyzer

cp nvidia_entrypoint.sh $HOME/triton-inference-server/model_analyzer

chmod 777 $HOME/triton-inference-server/model_analyzer/nvidia_entrypoint.sh
</pre>

## 2.- Model Definition and Folder Structure Set Up 
<pre style="background-color:rgba(227, 147, 125, 0.36)"><font size="3"><b>From here and below are the steps the user needs to run when analyzing a new model</b></pre>

#### Define model to be profiled and optimized
The user needs to define the name of the model to be used, the url location where to download the model from and the name of the file where it should be stored locally

In this case we are using a Bert model located at the ONNX Zoo
<pre style="background-color:rgba(0, 0, 0, 0.0470588)"><font size="2">model_name='bertsquad-8'

model_url='https://github.com/onnx/models/blob/master/text/machine_comprehension/bert-squad/model/bertsquad-8.onnx'

model_filename='bertsquad-8.onnx'
</pre>

<b>If the user wants to run these steps on its own model, the above variables would need to be redefined</b>

#### Set up Triton Inference Server Model folder structure
Next the [model folder structure](https://github.com/triton-inference-server/server/blob/main/docs/model_repository.md) under the model repository folder needs to be created 

<pre style="background-color:rgba(0, 0, 0, 0.0470588)"><font size="2">mkdir $HOME/triton-inference-server/model_analyzer/model_repository/${model_name}

mkdir $HOME/triton-inference-server/model_analyzer/model_repository/${model_name}/ori_model

original_model_dir=$HOME/triton-inference-server/model_analyzer/model_repository/${model_name}/ori_model

mkdir ${original_model_dir}/onnx_model

mkdir $HOME/triton-inference-server/model_analyzer/model_repository/${model_name}/1

model_filename_loc=${original_model_dir}/${model_filename}
</pre>

#### Download model from its url location
<pre style="background-color:rgba(0, 0, 0, 0.0470588)"><font size="2">wget -O $model_filename_loc ${model_url}?raw=true</pre>

## 3.- Model Preprocessing 

#### Converting Model to the required ONNX format

Getting the model root name, extension and defining names and locations of converted ONNX models

<pre style="background-color:rgba(0, 0, 0, 0.0470588)"><font size="2">extension="${model_filename##*.}"

model_filename_root="${model_filename%.*}"

model_onnx_filename=${model_filename_root}.onnx

###Setting up the name of the ONNX model with dynamic shape
model_onnx_filename_ds=${model_filename_root}-ds.onnx

###Setting up the name of the ONNX model with output nodes linked to the input nodes
model_onnx_filename_si=${model_filename_root}-si.onnx

### Setting up file locations
model_onnx_filename_loc=${original_model_dir}/onnx_model/$model_onnx_filename

model_onnx_filename_ds_loc=${original_model_dir}/onnx_model/$model_onnx_filename_ds

model_onnx_filename_si_loc=$HOME/triton-inference-server/model_analyzer/model_repository/${model_name}/1/$model_onnx_filename_si

model_converted=false
</pre>

If it is a TensoFlow model, it gets converted to the ONNX format and placed in the ONNX model folder
<pre style="background-color:rgba(0, 0, 0, 0.0470588)"><font size="2">if [ $extension == 'js' ]|| [ $extension = 'pb' ] 
then
   echo Converting TensorFlow Model to ONNX format.
   python3 -m tf2onnx.convert --saved-model $original_model_dir --opset 13 --output $model_onnx_filename_loc

   model_converted=true
fi
</pre>

If it is a PyTorch Model, to be converted to ONNX, in addition to the file containing the model itself (weights) another file (and its location) with the definition of the model and the nature of its inputs and outputs is required and needs to be provided by the user ($model_definition_file, $model_definition_file_url)
We provide a utility for such conversion please refer to this [file](https://github.com/Azure/azureml-examples/blob/triton-bert-perf/cli/endpoints/online/triton/batching/modelanalyzer/utils/ReadMe.md) for a usage example and the nature and expected format of the model definition file
<pre style="background-color:rgba(0, 0, 0, 0.0470588)"><font size="2">if [ $extension == 'pt' ] || [ $extension = 'pth' ]
then
   echo Converting Pytorch Model to ONNX format.

   model_definition_file='User needs to provide this info'

   model_definition_file_url='User needs to provide this info'

   wget -O ${original_model_dir}/convert_pytorch_to_onnx.py https://github.com/Azure/azureml-examples/blob/triton-bert-perf/cli/endpoints/online/triton/batching/modelanalyzer/utils/convert_pytorch_to_onnx.py?raw=true

   wget -O ${original_model_dir}/${model_definition_file} ${model_definition_file_url}?raw=true

   model_info_lib="${model_definition_file%.*}"

   sed -i "s/model_info_lib/${model_info_lib}/g" ${original_model_dir}/convert_pytorch_to_onnx.py

   python3 ${original_model_dir}/convert_pytorch_to_onnx.py --input $model_filename_loc --output $model_onnx_filename_loc

   model_converted=true
fi
</pre>

If the model is already on ONNX format, just copy it to the ONNX model folder
<pre style="background-color:rgba(0, 0, 0, 0.0470588)"><font size="2">if [ $extension == 'onnx' ]
then
  echo Copying Model to ONNX model folder.
  cp $model_filename_loc  $model_onnx_filename_loc
  model_converted=true
fi
</pre>

if model is not in PyTorch, TensorFlow or ONNX formats, it can not be analyzed
<pre style="background-color:rgba(0, 0, 0, 0.0470588)"><font size="2">if ! $model_converted
then
   echo Model is not in PyTorch, TensorFlow or ONNX formats so it can not be analyzed
fi
</pre>

#### Enabling dynamic batching on model

To fully utilize the power of dynamic batching from the model analyzer of Triton inference server, you need to make sure the onnx model is indeed in dynamic shape.

The user needs to check whether the DL model is a dynamic shape to utilize the powerful “dynamic batching” feature of Triton Inference Server. If the DL model has a fixed shape .i.e. (N,C,H,W >=1), then dynamic batching does not run. The easy way to check is to use online [Netron app](https://netron.app/) to open the model (then click the input node to check shape information). Typically, we want batch size N to be dynamic shape (N == “-1” or “alphanumeric characters”). Please refer to the [blog](https://microsoft.sharepoint.com/:w:/r/teams/1P3P512/_layouts/15/Doc.aspx?sourcedoc=%7BF222EA6C-D67E-4DED-B9C2-81B0A4979BA4%7D&file=AzureML-MLOPs-Triton%20Blog.docx&action=default&mobileredirect=true&share=IQFs6iLyftbtTbnCgbCkl5ukAYrJcUiC2d_gw-BeZBZOUCE) to get further explanation on this regard.

As explained in the [blog](https://microsoft.sharepoint.com/:w:/r/teams/1P3P512/_layouts/15/Doc.aspx?sourcedoc=%7BF222EA6C-D67E-4DED-B9C2-81B0A4979BA4%7D&file=AzureML-MLOPs-Triton%20Blog.docx&action=default&mobileredirect=true&share=IQFs6iLyftbtTbnCgbCkl5ukAYrJcUiC2d_gw-BeZBZOUCE), the downloaded BERT model from ONNX model zoo, has output nodes shapes that are already dynamic and then ORT-TRT cannot run. Se we need to set the static flag to false

<pre style="background-color:rgba(0, 0, 0, 0.0470588)"><font size="2">onnx_model_static=false
</pre>


To convert output shape dynamic, we need to run a script as described in [https://www.onnxruntime.ai/docs/reference/execution-providers/TensorRT-ExecutionProvider.html](https://www.onnxruntime.ai/docs/reference/execution-providers/TensorRT-ExecutionProvider.html)

if necessary, the dynamic shape is enabled by running the following command
<pre style="background-color:rgba(0, 0, 0, 0.0470588)"><font size="2">if $onnx_model_static
then
   echo Enabling dynamic shape on ONNX model
   polygraphy surgeon extract $model_onnx_filename_loc -o $model_onnx_filename_ds_loc --inputs input_ids,Nx128,int64 attention_mask,Nx128,int64 token_type_ids,Nx128,int64 
else
   echo Model is dynamic already just copying to dynamic model expected location
   cp $model_onnx_filename_loc $model_onnx_filename_ds_loc
fi
</pre>

Then the ouput nodes are linked to the input nodes by running:


<pre style="background-color:rgba(0, 0, 0, 0.0470588)"><font size="2">echo Linking outputs to inputs

python3 $HOME/triton-inference-server/onnxrt/onnxruntime/python/tools/symbolic_shape_infer.py --input $model_onnx_filename_ds_loc --output $model_onnx_filename_si_loc --auto_merge
</pre>

Now that we have a model ready to be optimized, we need to get and execute the model analyzer bits
<br/>

#### Setting onnx model and model analyzer config file

<pre style="background-color:rgba(0, 0, 0, 0.0470588)"><font size="2">wget -O $HOME/triton-inference-server/model_analyzer/model_repository/${model_name}/config.pbtxt https://github.com/Azure/azureml-examples/blob/triton-bert-perf/cli/endpoints/online/triton/batching/modelanalyzer/config.pbtxt?raw=true

wget -O $HOME/triton-inference-server/model_analyzer/config.yaml https://github.com/Azure/azureml-examples/blob/triton-bert-perf/cli/endpoints/online/triton/batching/modelanalyzer/config.yaml?raw=true
</pre>

Replacing the model filename placeholders in the model config file with the actual filenames

<pre style="background-color:rgba(0, 0, 0, 0.0470588)"><font size="2">sed -i "s/model_name/${model_name}/g" $HOME/triton-inference-server/model_analyzer/model_repository/${model_name}/config.pbtxt

sed -i "s/model_onnx_filename_si/${model_onnx_filename_si}/g" $HOME/triton-inference-server/model_analyzer/model_repository/${model_name}/config.pbtxt
</pre>

Letting the Model Analyzer know what model to analyze by replacing the model name placeholder in the Model analyzer config file with the actual model name

<pre style="background-color:rgba(0, 0, 0, 0.0470588)"><font size="2">sed -i "s/model_name/${model_name}/g" $HOME/triton-inference-server/model_analyzer/config.yaml
</pre>

## 4.- Model Profiling and Analysis  
### Analyzing the model

#### Setting desired precision 

<pre style="background-color:rgba(0, 0, 0, 0.0470588)"><font size="2">###Either FP16 or FP32
precision_value=FP16
</pre>

Setting up the model config file and output directories to the desired precision value
<pre style="background-color:rgba(0, 0, 0, 0.0470588)"><font size="2">sed -i "s/precision_value/${precision_value}/g" $HOME/triton-inference-server/model_analyzer/model_repository/${model_name}/config.pbtxt

mkdir $HOME/triton-inference-server/model_analyzer/${model_name}_${precision_value}ext_output_repository
</pre>

#### Running the Model Analyzer container
 
The local model repository: $HOME/triton-inference-server/model_analyzer/model_repository should be mounted into the container model repository directory. The directory $HOME/triton-inference-server/model_analyzer containing the Model Analyzer config file should also be mounted into the container file tree, along with a directory where to find the outputs

<pre style="background-color:rgba(0, 0, 0, 0.0470588)"><font size="2">docker run -it --privileged --rm --gpus all \
       -v $HOME/triton-inference-server/model_analyzer/model_repository:/opt/qa_model_repository/ \
       -v $HOME/triton-inference-server/model_analyzer:/opt/triton-model-analyzer \
       -v $HOME/triton-inference-server/model_analyzer/${model_name}_${precision_value}ext_output_repository:/output_model_repository/ \
       --net=host --name model-analyzer \
       model-analyzer /bin/bash
</pre>

Running the container on interactive mode (-it) would bring the terminal inside the container and one should see <b>/opt/triton-model-analyzer#</b> attached to the prompt

<details>
<summary><b>Please click to display the screenshot the user should expect to see</b></summary>
<img src="imgs/Screenshot.png" width="750">
</details>
<br/>

#### Model Analyzer config file
We would be profiling the model using this Model Analyzer config file: https://github.com/Azure/azureml-examples/blob/triton-bert-perf/cli/endpoints/online/triton/batching/modelanalyzer/config.yaml, which we have downloaded and mounted in the /opt/triton-model-analyzer folder in the Model Analyzer container.

<details>
<summary><b>Please click to display the config file used on this document</b></summary>
<pre style="background-color:rgba(227, 147, 125, 0.36)"><font size="2">model_repository: /opt/qa_model_repository
override_output_model_repository: False
output_model_repository_path: /output_model_repository/output
checkpoint_directory: ./checkpoints/

run_config_search_disable: False
run_config_search_max_concurrency: 256
run_config_search_max_instance_count: 5

perf_analyzer_cpu_util: 80000
perf_output: True

triton_launch_mode: local
triton_http_endpoint: localhost:8000
triton_grpc_endpoint: localhost:8001
triton_metrics_url: http://localhost:8002/metrics
triton_output_path: triton.log

batch_size: 1

profile_models:
   - bertsquad-8

#Config For Analyze
checkpoint_directory: ./checkpoints/
summarize: True
num_configs_per_model: 3

analysis_models: 
  bertsquad-8:
    objectives:
      perf_throughput: 10
</pre>
</details>

<br/>
Please refer to the [blog](https://microsoft.sharepoint.com/:w:/r/teams/1P3P512/_layouts/15/Doc.aspx?sourcedoc=%7BF222EA6C-D67E-4DED-B9C2-81B0A4979BA4%7D&file=AzureML-MLOPs-Triton%20Blog.docx&action=default&mobileredirect=true&share=IQFs6iLyftbtTbnCgbCkl5ukAYrJcUiC2d_gw-BeZBZOUCE) for more details about the config file

#### Profiling a model with the Model Analyzer

Run the following command to start the profiling process

<pre style="background-color:rgba(0, 0, 0, 0.0470588)"><font size="2">root:/opt/triton-model-analyzer# model-analyzer profile -f config.yaml
</pre>

<details>
<summary><b>Please click to display the screenshot the user should expect to see</b></summary>
<img src="imgs/Screenshot_Profile.png" width="750">
<img src="imgs/Screenshot_Profile2.png" width="750">
</details>
<br/>
The profiling would take a few hours

#### Analyzing profiling results

To generate reports run:

<pre style="background-color:rgba(0, 0, 0, 0.0470588)"><font size="2">root:/opt/triton-model-analyzer# model-analyzer analyze --checkpoint-directory=`pwd`/checkpoints -f config.yaml
</pre>
 
<details>
<summary><b>Please click to display the screenshot the user should expect to see</b></summary>
<img src="imgs/Screenshot_Analysis.png" width="750">
</details>
<br/>

Reports can be found in /opt/triton-model-analyzer/reports

<details>
<summary><b>Please click to display the the content on file: /opt/triton-model-analyzer/reports/summaries/bertsquad-8/result_summary.pdf </b></summary>
<img src="imgs/MAResultsSummary.png" width="750">
<img src="imgs/MAResultsSummary2.png" width="750">
</details>
<br/>

Please refer to the [blog](https://microsoft.sharepoint.com/:w:/r/teams/1P3P512/_layouts/15/Doc.aspx?sourcedoc=%7BF222EA6C-D67E-4DED-B9C2-81B0A4979BA4%7D&file=AzureML-MLOPs-Triton%20Blog.docx&action=default&mobileredirect=true&share=IQFs6iLyftbtTbnCgbCkl5ukAYrJcUiC2d_gw-BeZBZOUCE) for an explanation of the summary of results

Detailed results can be found in /opt/triton-model-analyzer/results

## 5.-Choosing optimal configuration file 

Automatically generated model repository can be found in /output_model_repository/output

<details>
<summary><b>Please click to display the screenshot the user should expect to see</b></summary>
<img src="imgs/Screenshot_Outputs.png" width="750">
</details>
<br/>

Where the user can find the config files for each of the runs tried by the Model Analyzer, as explained in the [blog](https://microsoft.sharepoint.com/:w:/r/teams/1P3P512/_layouts/15/Doc.aspx?sourcedoc=%7BF222EA6C-D67E-4DED-B9C2-81B0A4979BA4%7D&file=AzureML-MLOPs-Triton%20Blog.docx&action=default&mobileredirect=true&share=IQFs6iLyftbtTbnCgbCkl5ukAYrJcUiC2d_gw-BeZBZOUCE), the user should take the one that corresponds to the optimal configuration and use it to deploy the model to production 
