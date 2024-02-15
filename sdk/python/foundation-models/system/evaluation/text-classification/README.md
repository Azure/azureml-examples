## Single Label Classification

### List of supported keyword arguments:

|     Keyword Argument     | Description                                                                    | Type             | Sample                                                          |
|:------------------------:|:-------------------------------------------------------------------------------|------------------|-----------------------------------------------------------------|
|         metrics          | List for subset of metrics to be computed. All supported metrics listed below. | list<str>        | ["accuracy", "f1_score_micro", "average_precision_score_macro"] |
|       class_labels       | List for superset of all existing labels in our dataset                        | list, np.ndarray | [0, 1, 2, 3], ["CONTRADICTION", "NEUTRAL", "ENTAILMENT"]        |
|       train_labels       | List for labels on which model is trained                                      | list, np.ndarray | [0, 1, 2, 3], ["CONTRADICTION", "NEUTRAL", "ENTAILMENT"]        |
|      sample_weights      | List containing the weight associated with each data sample                    | list, np.ndarray | [1, 2, 3, 4, 5, 6]                                              |
|      y_transformer       | Transformer object to be applied on y_pred                                     |                  |                                                                 |
|        use_binary        | Compute metrics only on the true class for binary classification               | boolean          | true, false                                                     |
| enable_metric_confidence | Computes confidence interval for supported metrics                             | boolean          | true, false                                                     |
|        multilabel        | Boolean variable that computes multilabel metrics when set to True             | boolean          | false (Should be false for single label classification)         |
|      positive_label      | Label to be treated as positive label                                          | int/str          | 0, "CONTRADICTION"                                              |
|    confidence_metrics    | List of metrics to compute confidence intervals                                | list<str>        | ["accuracy", "f1_score_micro"]                                  |

### List of supported metrics:

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