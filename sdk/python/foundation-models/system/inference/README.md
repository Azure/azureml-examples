# Foundation Model Inferencing
The __foundation-model-inference__ container is a curated solution for inferencing foundation models on Azure ML. It incorporates the best inferencing frameworks to ensure optimal request throughput and latency. The container is user-friendly and integrates seamlessly with Azure ML. It utilizes:

## vLLM
vLLM is a high-performance inferencing server that offers several features, making it a top choice for inferencing systems. vLLM provides:
- Top-tier serving throughput
- Efficient handling of attention key and value memory with PagedAttention
- Continuous batching of incoming requests
- Optimized CUDA kernels
- High-throughput serving with various decoding algorithms, including parallel sampling, beam search, and more
- Support for tensor parallelism for distributed inference

## DeepSpeed FastGen
DeepSpeed FastGen, a recent release from DeepSpeed, offers up to 2.3x faster throughput than vLLM, which already outperforms similar frameworks like Huggingface TGI. DeepSpeed-FastGen combines DeepSpeed-MII and DeepSpeed-Inference to deliver a fast and user-friendly serving system.
DeepSpeed FastGen features include:
- Upt to 2.3x faster throughput than vLLM
- Optimized memory handling with a blocked KV cache
- Continuous batching of incoming requests
- Optimized CUDA kernals
- Tensor parallelism support
- New Dynamic Splitfuse technique to increase overall performance and provide better throughput consistency.

DeepSpeed FastGen achieves superior performance by using a new technique called Dynamic Splitfuse. This technique enhances responsiveness, efficiency, and result consistency. For more information, visit the DeepSpeed FastGen [github page](https://github.com/microsoft/DeepSpeed/blob/master/blogs/deepspeed-fastgen/README.md).

## Supported Tasks by the Container
- Text Generation
> More tasks will be supported soon.

For additional information on this container and its use with foundation models, refer to section 3.4 of the [text-generation example](https://github.com/Azure/azureml-examples/blob/main/sdk/python/foundation-models/system/inference/text-generation/llama-safe-online-deployment.ipynb).
