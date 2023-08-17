## Object Detection

### List of supported keyword arguments:

|     Keyword Argument     | Description                                                                    | Type             | Sample                                                          |
|:------------------------:|:-------------------------------------------------------------------------------|------------------|-----------------------------------------------------------------|
|         metrics          | List for subset of metrics to be computed. All supported metrics listed below. | list<str>        | ["mean_average_precision", "recall", "precision", "per_label_metrics", "image_level_binary_classsifier_metrics", "confusion_matrices_per_score_threshold"] |
| enable_metric_confidence | Computes confidence interval for supported metrics                             | boolean          | true, false                                                     |
|    confidence_metrics    | List of metrics to compute confidence intervals                                | list<str>        | ["mean_average_precision"]                                  |
|    custom_dimensions     | Used to report telemetry data (can later be used to perform PII scrubbing)     | dict             |                                                                 |

### List of supported metrics:

* mean_average_precision
* recall
* precision
* per_label_metrics
* image_level_binary_classsifier_metrics
* confusion_matrices_per_score_threshold
