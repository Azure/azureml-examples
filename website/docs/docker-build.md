---
title: 'Azure ML Containers'
---

In this post we explain how Azure ML builds the containers used to run your code.

## Dockerfile

Each job in Azure ML runs with an associated `Environment`. In practice, each environment
corresponds to a Docker image.

There are numerous ways to define an environment - from specifying a set of required Python packages
through to directly providing a custom Docker image. In each case the contents of the associated
dockerfile are available directly from the environment object.

For more background: [Environment](environment)

#### Example

Suppose you create an environment - in this example we will work with Conda:

```yml title="env.yml"
name: pytorch
channels:
    - defaults
    - pytorch
dependencies:
    - python=3.7
    - pytorch
    - torchvision
```

We can create and register this as an `Environment` in our workspace `ws` as follows:

```python
from azureml.core import Environment
env = Environment.from_conda_specification('pytorch', 'env.yml')
env.register(ws)
```

In order to consume this environment in a remote run, Azure ML builds a docker image
that creates the corresponding python environment.

The dockerfile used to build this image is available directly from the environment object.

```python
details = env.get_image_details(ws)
print(details['ingredients']['dockerfile'])
```

Let's take a look:

```docker title="Dockerfile" {1,7-12}
FROM mcr.microsoft.com/azureml/intelmpi2018.3-ubuntu16.04:20200821.v1@sha256:8cee6f674276dddb23068d2710da7f7f95b119412cc482675ac79ba45a4acf99
USER root
RUN mkdir -p $HOME/.cache
WORKDIR /
COPY azureml-environment-setup/99brokenproxy /etc/apt/apt.conf.d/
RUN if dpkg --compare-versions `conda --version | grep -oE '[^ ]+$'` lt 4.4.11; then conda install conda==4.4.11; fi
COPY azureml-environment-setup/mutated_conda_dependencies.yml azureml-environment-setup/mutated_conda_dependencies.yml
RUN ldconfig /usr/local/cuda/lib64/stubs && conda env create -p /azureml-envs/azureml_7459a71437df47401c6a369f49fbbdb6 -
f azureml-environment-setup/mutated_conda_dependencies.yml && rm -rf "$HOME/.cache/pip" && conda clean -aqy && CONDA_ROO
T_DIR=$(conda info --root) && rm -rf "$CONDA_ROOT_DIR/pkgs" && find "$CONDA_ROOT_DIR" -type d -name __pycache__ -exec rm
 -rf {} + && ldconfig
# AzureML Conda environment name: azureml_7459a71437df47401c6a369f49fbbdb6
ENV PATH /azureml-envs/azureml_7459a71437df47401c6a369f49fbbdb6/bin:$PATH
ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/azureml_7459a71437df47401c6a369f49fbbdb6
ENV LD_LIBRARY_PATH /azureml-envs/azureml_7459a71437df47401c6a369f49fbbdb6/lib:$LD_LIBRARY_PATH
COPY azureml-environment-setup/spark_cache.py azureml-environment-setup/log4j.properties /azureml-environment-setup/
RUN if [ $SPARK_HOME ]; then /bin/bash -c '$SPARK_HOME/bin/spark-submit  /azureml-environment-setup/spark_cache.py'; fi
ENV AZUREML_ENVIRONMENT_IMAGE True
CMD ["bash"]
```

Notice:

- The base image here is a standard image maintained by Azure ML. Dockerfiles for all base images are available on
github: https://github.com/Azure/AzureML-Containers
- The dockerfile references `mutated_conda_dependencies.yml` to build the Python environment via Conda.

Get the contents of `mutated_conda_dependencies.yml` from the environment:

```python
print(env.python.conda_dependencies.serialize_to_string())
```

Which looks like

```bash title="mutated_conda_dependencies.yml"
channels:
    - defaults
    - pytorch
dependencies:
    - python=3.7
    - pytorch
    - torchvision
name: azureml_7459a71437df47401c6a369f49fbbdb6
```