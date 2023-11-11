# Foundation Model Inferencing
For inferencing foundation models on Azure ML, the curated container __foundation-model-inference__ is used. The container leverages the best inferencing frameworks available to have optimal throughput and latency for requests, and is easy to get started with using Azure ML. The container utilizes:

## vLLM
vLLM is an easy to use fast inferencing server with several features that put it at the top of the list of inferencing systems. vLLM has:
- state-of-the-art serving throughput
- Efficient management of attention key and value memory with PagedAttention (see [this blog post](https://vllm.ai/) for more details)
- Continuous batching of incoming requests
- Optimized CUDA kernels
- High-throughput serving with various decoding algorithms, including parallel sampling, beam search, and more
- Tensor parallelism support for distributed inference

## DeepSpeed FastGen
DeepSpeed FastGen is the recently released inferencing framework from DeepSpeed that has shown to have up to 2.3x faster throughout than vLLM, which is already faster than other similar frameworks such as Huggingface TGI. DeepSpeed-FastGen leverages the combination of DeepSpeed-MII and DeepSpeed-Inference to provide a fast, easy-to-use serving system.

To achieve new levels of performance that surpass existing inferencing systems, DeepSpeed FastGen uses a new technique called Dynamic Splitfuse that provides better responsiveness, higher efficiency, and lower variance in results. For more information, see the DeepSpeed FastGen [github page](https://github.com/microsoft/DeepSpeed/blob/master/blogs/deepspeed-fastgen/README.md).

## Container Supported tasks
- Text Generation
> More supported tasks on the way soon.

For more information on this container and to use it with foundation models, see the [text-generation example](https://github.com/Azure/azureml-examples/blob/main/sdk/python/foundation-models/system/inference/text-generation/llama-safe-online-deployment.ipynb).