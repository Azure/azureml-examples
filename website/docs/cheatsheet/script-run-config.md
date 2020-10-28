---
title: Running Code in the Cloud
description: Guide to running code with Azure ML
keywords:
  - run
  - experiment
  - submit
  - remote
  - ScriptRunConfig
---

## Experiments and Runs

Azure ML is a machine-learning service that facilitates running your code in
the cloud. A `Run` is an abstraction layer around each such submission, and is used to
monitor the job in real time as well as keep a history of your results.

- Run: A run represents a single execution of your code. See [Run](#run) for more details.
- Experiments: An experiment is a light-weight container for `Run`. Use experiments to submit
and track runs.

Create an experiment in your workspace `ws`.

```python
from azureml.core import Experiment
exp = Experiment(ws, '<experiment-name>')
```

## ScriptRunConfig

A common way to run code in the cloud is via the `ScriptRunConfig` which packages
your source code (Script) and run configuration (RunConfig).

Consider the following layout for your code.

```bash
source_directory/
    script.py    # entry point to your code
    module1.py   # modules called by script.py     
    ...
```

To run `script.py` in the cloud via the `ScriptRunConfig`

```python
config = ScriptRunConfig(
    source_directory='<path/to/source_directory>',
    script='script.py',
    compute_target=target,
    environment=env,
    arguments = [
        '--learning_rate', 0.001,
        '--momentum', 0.9,
    ]
)
```

where:

- `source_directory='source_directory'` : Local directory with your code.
- `script='script.py'` : Script to run. This does not need to be at the root of `source_directory`.
- `compute_taget=target` : See [Compute Target](copute-target)
- `environment` : See [Environment](environment)
- `arguments` : See [Arguments](#command-line-arguments)

Submit this code to Azure with

```python
exp = Experiment(ws, '<exp-name>')
run = exp.submit(config)
print(run)
run.wait_for_completion(show_output=True)
```

This will present you with a link to monitor your run on the web (https://ml.azure.com)
as well as streaming logs to your terminal.

## Command Line Arguments

To pass command line arguments to your script use the `arguments` parameter in `ScriptRunConfig`.
Arguments are passed as a list

```python
arguments=[first, second, third, ...]
```

which are then passed to the script as command-line arguments as follows:

```console
$ python script.py first second third ...
```

This also supports using named arguments:

```python
arguments=['--first_arg', first_val, '--second_arg', second_val, ...]
```

Arguments can be of type `int`, `float` `str` and can also be used to reference data.

For more details on referencing data via the command line: [Use dataset in a remote run](dataset#use-dataset-in-a-remote-run)

### Example: `sys.argv`

In this example we pass two arguments to our script. If we were running this from the
console:

```console title="console"
$ python script.py 0.001 0.9
```

To mimic this command using `argument` in `ScriptRunConfig`:

```python title="run.py"
arguments = [0.001, 0.9]

config = ScriptRunConfig(
    source_directory='.',
    script='script.py',
    arguments=arguments,
)
```

which can be consumed as usual in our script:

```python title="script.py"
import sys
learning_rate = sys.argv[1]     # gets 0.001
momentum = sys.argv[2]          # gets 0.9
```

### Example: `argparse`

In this example we pass two named arguments to our script. If we were running this from the
console:

```console title="console"
$ python script.py --learning_rate 0.001 --momentum 0.9
```

To mimic this behavior in `ScriptRunConfig`:

```python title="run.py"
arguments = [
    '--learning_rate', 0.001, 
    '--momentum', 0.9,
    ]

config = ScriptRunConfig(
    source_directory='.',
    script='script.py',
    arguments=arguments,
)
```

which can be consumed as usual in our script:

```python title="script.py"
import argparse
parser = argparse.Argparser()
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--momentum', type=float)
args = parser.parse_args()

learning_rate = args.learning_rate      # gets 0.001
momentum = args.momentum                # gets 0.9
```

## Using Datasets

### via Arguments

Pass a dataset to your ScriptRunConfig as an argument

```py
# create dataset
datastore = ws.get_default_datastore()
dataset = Dataset.File.from_files(path=(datastore, '<path/on/datastore>'))

arguments = ['--dataset', dataset.as_mount()]

config = ScriptRunConfig(
    source_directory='.',
    script='script.py',
    arguments=arguments,
)
```

This mounts the dataset to the run where it can be referenced by `script.py`.

## Run

### Interactive

In an interactive setting e.g. a Jupyter notebook

```python
run = exp.start_logging()
```

#### Example: Jupyter notebook

A common use case for interactive logging is to train a model in a notebook.

```py
from azureml.core import Workspace
from azureml.core import Experiment
ws = Workspace.from_config()
exp = Experiment(ws, 'example')

run = exp.start_logging()                   # start interactive run
print(run.get_portal_url())                 # get link to studio

# toy example in place of e.g. model
# training or exploratory data analysis
import numpy as np
for x in np.linspace(0, 10):
    y = np.sin(x)
    run.log_row('sine', x=x, y=y)           # log metrics

run.complete()                              # stop interactive run
```

Follow the link to the run to see the metric logging in real time.

![](img/run-ex-sine.png)

### Get Context

Code that is running within Azure ML is associated to a `Run`. The submitted code
can access its own run.

```py
from azureml.core import Run
run = Run.get_context()
```

#### Example: Logging metrics to current run context

A common use-case is logging metrics in a training script.

```py title="train.py"
from azureml.core import Run

run = Run.get_context()

# training code
for epoch in range(n_epochs):
    model.train()
    ...
    val = model.evaluate()
    run.log('validation', val)
```

When this code is submitted to Azure ML (e.g. via ScriptRunConfig) it will log metrics to its associated run.

For more details: [Logging Metrics](logging)