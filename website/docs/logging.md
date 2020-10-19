---
title: Logging Metrics
---

## Logging metrics

Logging a metric to a run causes that metric to be stored in the run record in the experiment.
Visualize and keep a history of all logged metrics.


### `log`

Log a single metric value to a run.

```python
from azureml.core import Run
run = Run.get_context()
run.log('metric-name', metric_value)
```

You can log the same metric multiple times within a run, the result being considered a vector
of that metric.

### `log_row`

Log a metric with multiple columns.

```python
from azureml.core import Run
run = Run.get_context()
run.log_row("Y over X", x=1, y=0.4)
```

### With MLFlow

Use MLFlowLogger to log metrics.

```python title="script.py"
from azureml.core import Run

# connect to the workspace from within your running code
run = Run.get_context()
ws = run.experiment.workspace

# workspace has associated ml-flow-tracking-uri
mlflow_url = ws.get_mlflow_tracking_uri()
```

#### Example: PyTorch Lightning

```python
from pytorch_lightning.loggers import MLFlowLogger

mlf_logger = MLFlowLogger(experiment_name=run.experiment.name, tracking_uri=mlflow_url)
mlf_logger._run_id = run.id
```

## Viewing metrics

### Via the SDK

Viewing metrics in a run (for more details on runs: [Run](run))

```python
metrics = run.get_metrics()
# metrics is of type Dict[str, List[float]] mapping mertic names
# to a list of the values for that metric in the given run.

metrics.get('metric-name')
# list of metrics in the order they were recorded
```

To view all recorded values for a given metric `my-metric` in a
given experiment `my-experiment`:

```python
experiments = ws.experiments
# of type Dict[str, Experiment] mapping experiment names the
# corresponding Experiment

exp = experiments['my-experiment']
for run in exp.get_runs():
    metrics = run.get_metrics()
    
    my_metric = metrics.get('my-metric')
    if my_metric:
        print(my_metric)
```