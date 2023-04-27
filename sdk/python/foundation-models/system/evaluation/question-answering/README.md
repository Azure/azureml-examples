## Question Answering

### List of supported keyword arguments:

|  Keyword Argument  | Description                                                                    | Type      | Sample                      |
|:------------------:|:-------------------------------------------------------------------------------|-----------|-----------------------------|
|      metrics       | List for subset of metrics to be computed. All supported metrics listed below. | list<str> | ["exact_match", "f1_score"] |
|     tokenizer      | Tokenizer object to perform tokenization on provided input text                | boolean   | false, true                 |
| regexes_to_ignore  | List of regex to ignore in our input data points                               | list      | ["$[A-Z]+"]                 |
|    ignore_case     | Boolean flag to indicate whether we need to ignore case                        | boolean   | false, true                 |
| ignore_punctuation | Boolean flag to indicate whether we need to ignore punctuation                 | boolean   | false, true                 |
|   ignore_numbers   | Boolean flag to indicate whether we need to ignore numbers                     | boolean   | false, true                 |
| custom_dimensions  | Used to report telemetry data (can later be used to perform PII scrubbing)     | dict      |                             |

### List of supported metrics:

* rouge1
* rouge2
* rougeLsum
* rougeL