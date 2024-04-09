## Translation

### List of supported keyword arguments:

| Keyword Argument  | Description                                                                    | Type      | Sample                                   |
|:-----------------:|:-------------------------------------------------------------------------------|-----------|------------------------------------------|
|      metrics      | List for subset of metrics to be computed. All supported metrics listed below. | list<str> | ["bleu_1", "bleu_2", "bleu_3", "bleu_4"] |
|     tokenizer     | Tokenizer object to perform tokenization on provided input text                |           |                                          |
|     smoothing     | Boolean flag to indicate if bleu score needs to be smoothened                  | boolean   | false, true                              |

### List of supported metrics:

* bleu_1
* bleu_2
* bleu_3
* bleu_4