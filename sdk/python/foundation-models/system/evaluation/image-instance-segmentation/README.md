## Instance Segmentation

### List of supported keyword arguments:

|     Keyword Argument     | Description                                                                    | Type             | Sample                                                          |
|:------------------------:|:-------------------------------------------------------------------------------|------------------|-----------------------------------------------------------------|
|         metrics          | List for subset of metrics to be computed. All supported metrics listed below. | list<str>        | ["mean_average_precision", "recall", "precision", "per_label_metrics"] |
| enable_metric_confidence | Computes confidence interval for supported metrics                             | boolean          | true, false                                                     |
|    confidence_metrics    | List of metrics to compute confidence intervals                                | list<str>        | ["mean_average_precision"]                                  |
|    custom_dimensions     | Used to report telemetry data (can later be used to perform PII scrubbing)     | dict             |                                                                 |

### List of supported metrics:

* mean_average_precision
* recall
* precision
* per_label_metrics
