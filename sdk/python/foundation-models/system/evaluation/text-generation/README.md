## Text Generation

### List of supported keyword arguments:

| Keyword Argument  | Description                                                                           | Type      | Sample                                   |
|:-----------------:|:--------------------------------------------------------------------------------------|-----------|------------------------------------------|
|      metrics      | List for subset of metrics to be computed. All supported metrics listed below.        | list<str> | ["bleu_1", "bleu_2", "rouge1", "rouge2"] |
|     tokenizer     | Tokenizer object to perform tokenization on provided input text                       |           |                                          |
|     smoothing     | Boolean flag to indicate if bleu score needs to be smoothened                         | boolean   | false, true                              |
|    aggregator     | Boolean flag to indicate if need to aggregate rouge scores for individual data points | boolean   | true, false                              |
|      stemmer      | Boolean flag to indicate whether to use Porter Stemmer for suffixes                   | boolean   | true, false                              |

### List of supported metrics:

* rouge1
* rouge2
* rougeLsum
* rougeL
* bleu_1
* bleu_2
* bleu_3
* bleu_4