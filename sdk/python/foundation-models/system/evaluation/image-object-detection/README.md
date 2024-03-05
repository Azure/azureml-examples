## Object Detection
Note: The evaluation config is an optional configuration file that can be provided for model evaluation. If not provided, default values for the arguments below will be chosen based on the task type.
### List of supported keyword arguments:

|     Keyword Argument     | Description                                                                    | Type             | Sample                                                          |
|:------------------------:|:-------------------------------------------------------------------------------|------------------|-----------------------------------------------------------------|
|         metrics          | List for subset of metrics to be computed. All supported metrics listed below. | list<str>        | ["mean_average_precision", "recall", "precision", "per_label_metrics", "image_level_binary_classsifier_metrics", "confusion_matrices_per_score_threshold"] |
| iou_threshold | IOU threshold used during inference in non-maximum suppression post processing.                             | float          | 0.5                                                    |
|    box_score_threshold    | During inference, only return proposals with a score greater than `box_score_threshold`. The score is the multiplication of the objectness score and classification probability.                                | float       |   0.3                                 |


### List of supported metrics:

* mean_average_precision
* recall
* precision
* per_label_metrics
* image_level_binary_classsifier_metrics
* confusion_matrices_per_score_threshold
