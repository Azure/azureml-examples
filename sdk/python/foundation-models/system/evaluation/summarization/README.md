## Summarization

### List of supported keyword arguments:

| Keyword Argument  | Description                                                                           | Type      | Sample                                      |
|:-----------------:|:--------------------------------------------------------------------------------------|-----------|---------------------------------------------|
|      metrics      | List for subset of metrics to be computed. All supported metrics listed below.        | list<str> | ["rouge1", "rouge2", "rougeL", "rougeLsum"] |
|    aggregator     | Boolean flag to indicate if need to aggregate rouge scores for individual data points | boolean   | true, false                                 |
|      stemmer      | Boolean flag to indicate whether to use Porter Stemmer for suffixes                   | boolean   | true, false                                 |

### List of supported metrics:

* rouge1
* rouge2
* rougeLsum
* rougeL