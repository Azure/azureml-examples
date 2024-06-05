## Fill Mask

### List of supported keyword arguments:

| Keyword Argument  | Description                                                                                                      | Type      | Sample                      |
|:-----------------:|:-----------------------------------------------------------------------------------------------------------------|-----------|-----------------------------|
|      metrics      | List for subset of metrics to be computed. All supported metrics listed below.                                   | list<str> | ["perplexities"]            |
|     model_id      | Model used for calculating Perplexity. Perplexity can only be calculated for causal language models.             | str       | "gpt2", "bert-base-uncased" |
|    batch_size     | The batch size to run texts through the model                                                                    | int       | 16                          |
|  add_start_token  | Boolean flag to add the start token to the texts so the perplexity can include the probability of the first word | boolean   | true, false                 |

### List of supported metrics:

* perplexity