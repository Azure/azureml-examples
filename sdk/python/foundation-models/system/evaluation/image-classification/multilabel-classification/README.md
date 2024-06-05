## Multilabel Classification

Note: The evaluation config is an optional configuration file that can be provided for model evaluation. If not provided, default values for the arguments below will be chosen based on the task type.

### List of supported keyword arguments:

|     Keyword Argument     | Description                                                                    | Type             | Sample                                                          |
|:------------------------:|:-------------------------------------------------------------------------------|------------------|-----------------------------------------------------------------|
|         metrics          | List for subset of metrics to be computed. All supported metrics listed below. | list<str>        | ["iou", "f1_score_micro", "average_precision_score_macro"] |
|       threshold       | Threshold value applied to the predicted probabilities from the classifier.                        | float | 0.5        |

### List of supported metrics:

* iou
* iou_macro
* iou_micro
* iou_weighted
* log_loss
* average_precision_score_binary
* weighted_accuracy
* AUC_weighted
* f1_score_micro
* f1_score_binary
* precision_score_micro
* precision_score_binary
* recall_score_weighted
* f1_score_weighted
* confusion_matrix
* average_precision_score_micro
* recall_score_binary
* recall_score_macro
* average_precision_score_weighted
* AUC_binary
* matthews_correlation
* precision_score_macro
* accuracy
* average_precision_score_macro
* AUC_macro
* recall_score_micro
* balanced_accuracy
* f1_score_macro
* precision_score_weighted
* accuracy_table
* AUC_micro
* norm_macro_recall