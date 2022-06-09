#!/bin/sh

# Setup OLive
pip install onnxruntime_olive-0.3.0-py3-none-any.whl
pip install onnxruntime_gpu_tensorrt-1.9.0-cp38-cp38-linux_x86_64.whl

# Setup model analyzer
pip install triton-model-analyzer

# Setup for conversion
pip install transformers

# This folder is missing
mkdir bertsquad/1 
# Run OLive conversion
olive convert --model_path bert-base-cased-squad.pth --model_framework pytorch --framework_version 1.11 --input_names input_names,input_mask,segment_ids --output_names start,end --onnx_opset 11 --input_shapes [[-1,256],[-1,256],[-1,256]]  --input_types int64,int64,int64 --onnx_model_path bertsquad/1/model.onnx 

#Copy default model to model repository
mkdir model_repository
mkdir model_repository/bertsquad_default
cp -r bertsquad/* model_repository/bertsquad_default
sed -i "s/bertsquad/bertsquad_default/g" model_repository/bertsquad_default/config.pbtxt

# Run OLive optimization
olive optimize --model_path bertsquad/1/model.onnx --model_analyzer_config bertsquad/config.pbtxt --providers_list tensorrt --trt_fp16_enabled --result_path bertsquad_model_analyzer

# Copy OLive Optimized model to Model Repository
mkdir model_repository/bertsquad
mkdir model_repository/bertsquad/1
cp bertsquad_model_analyzer/optimized_model.onnx model_repository/bertsquad/1/model.onnx
cp bertsquad_model_analyzer/olive_result.pbtxt model_repository/bertsquad/config.pbtxt

# Create folder needed by Model Analyzer
mkdir checkpoints
mkdir output_model_repository

# Run Model Analyzer Optimization
model-analyzer -v profile -f config_bert-olive.yml

# Run Model Analyzer Analyzer that would create metrics files
model-analyzer analyze -f config_bert_olive_analyze.yml

# Process Metrics Files, plot results and find location of optimal config file
pip install pandas
python3 Plot_Performance_Results.py --output_repository ./output_model_repository/output_bert_olive1 --inference_results_file results/metrics-model-inference.csv --output_figure_file Optimal_Results.png --optimal_location_file Optimal_ConfigFile_Location.txt

