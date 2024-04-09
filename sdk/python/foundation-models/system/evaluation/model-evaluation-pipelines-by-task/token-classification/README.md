## Named Entity Recognition

### List of supported keyword arguments:

|     Keyword Argument     | Description                                                                    | Type      | Sample                                                        |
|:------------------------:|:-------------------------------------------------------------------------------|-----------|---------------------------------------------------------------|
|         metrics          | List for subset of metrics to be computed. All supported metrics listed below. | list<str> | ["accuracy", "f1_score_macro", "f1_score_micro"]              |
|       labels_list        | List for supported labels for tokens                                           | list<str> | ["B-PER", "I-PER", "O", "B-LOC", "I-LOC", "B-MISC", "I-MISC"] |

### List of supported metrics:

* f1_score_macro
* precision_score_weighted
* precision_score_macro
* f1_score_weighted
* precision_score_micro
* recall_score_weighted
* f1_score_micro
* accuracy
* recall_score_micro
* recall_score_macro