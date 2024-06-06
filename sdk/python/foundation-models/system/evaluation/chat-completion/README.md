## Chat Completion

### List of supported keyword arguments:

|    Keyword Argument     | Description                                                                                                      | Type      | Sample                                   |
|:-----------------------:|:-----------------------------------------------------------------------------------------------------------------|-----------|------------------------------------------|
|         metrics         | List for subset of metrics to be computed. All supported metrics listed below.                                   | list<str> | ["bleu_1", "bleu_2", "rouge1", "rouge2"] |
|        tokenizer        | Tokenizer object to perform tokenization on provided input text                                                  |           |                                          |
|        smoothing        | Boolean flag to indicate if bleu score needs to be smoothened                                                    | boolean   | false, true                              |
|       aggregator        | Boolean flag to indicate if need to aggregate rouge scores for individual data points                            | boolean   | true, false                              |
|         stemmer         | Boolean flag to indicate whether to use Porter Stemmer for suffixes                                              | boolean   | true, false                              |
|        model_id         | Model used for calculating Perplexity. Perplexity can only be calculated for causal language models.             | str       | "gpt2", "bert-base-uncased"              |
|       batch_size        | The batch size to run texts through the model                                                                    | int       | 16                                       |
|     add_start_token     | Boolean flag to add the start token to the texts so the perplexity can include the probability of the first word | boolean   | true, false                              |
|      openai_params      | Dictionary containing credentials for openai API (propagated directly to openai APIs).                           | dict      | {}                                       |
|  openai_api_batch_size  | # of prompts to be batched in one API call (applicable only for models with completion API support).             | int       | 16                                       |
| use_chat_completion_api | Boolean flag to indicate if openai chat completion API needs to be used (default=None)                           | boolean   | true, false                              |
|      score_version      | Version of the prompt template to compute rag based metrics (default="v1")                                       | str       | "v1"                                     |

### List of supported metrics:

* bleu_1
* bleu_2
* bleu_3
* bleu_4
* conversation_groundedness_score
* gpt_groundedness
* gpt_relevance
* gpt_retrieval_score
* perplexity
* rouge1
* rouge2
* rougeL
* rougeLsum