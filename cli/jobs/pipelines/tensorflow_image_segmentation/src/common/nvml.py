# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This scripts leverages pynvml to get important properties about the GPUs
(to log them as mlflow parameters).
"""
import logging
import traceback


def get_nvml_params() -> dict:
    """Uses pynvml to get all VM related parameters to log"""
    logger = logging.getLogger(__name__)

    # if pynvml isn't there, do not fail
    try:
        import pynvml
    except BaseException:
        logger.critical(
            f"Cannot get CUDA machine parameters because importing pynvml failed.\n\n{traceback.format_exc()}"
        )
        return {}

    try:
        pynvml.nvmlInit()
    except BaseException:
        logger.critical(
            f"Cannot get CUDA machine parameters because pynvml.nvmlInit() failed.\n\n{traceback.format_exc()}"
        )
        return {}

    machine_params = {
        "cuda_driver_version": pynvml.nvmlSystemGetCudaDriverVersion_v2(),
        "cuda_system_driver_version": pynvml.nvmlSystemGetDriverVersion(),
    }

    cuda_device_count = pynvml.nvmlDeviceGetCount()
    machine_params["cuda_device_count"] = cuda_device_count

    if cuda_device_count > 0:
        device_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        # some calls might not be supported by local cuda library
        try:
            machine_params["cuda_device_name"] = pynvml.nvmlDeviceGetName(device_handle)
        except pynvml.nvml.NVMLError_NotSupported:
            logger.warning("nvmlDeviceGetName() not supported.")

        try:
            cuda_device_memory = pynvml.nvmlDeviceGetMemoryInfo(device_handle)
            machine_params["cuda_device_memory"] = (
                str((cuda_device_memory.total // (1024 * 1024 * 1024))) + "G"
            )
        except pynvml.nvml.NVMLError_NotSupported:
            logger.warning("nvmlDeviceGetMemoryInfo() not supported.")

        try:
            cuda_device_attributes = pynvml.nvmlDeviceGetAttributes(device_handle)
            machine_params[
                "cuda_device_processor_count"
            ] = cuda_device_attributes.multiprocessorCount
        except pynvml.nvml.NVMLError_NotSupported:
            logger.warning("nvmlDeviceGetAttributes() not supported.")

    pynvml.nvmlShutdown()

    return machine_params


if __name__ == "__main__":
    # for local testing
    print(get_nvml_params())
