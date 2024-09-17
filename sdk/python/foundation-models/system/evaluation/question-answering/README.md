## Question Answering

### List of supported keyword arguments:

|   Keyword Argument    | Description                                                                    | Type            | Sample                      |
|:---------------------:|:-------------------------------------------------------------------------------|-----------------|-----------------------------|
|        metrics        | List for subset of metrics to be computed. All supported metrics listed below. | list<str>       | ["exact_match", "f1_score"] |
|       tokenizer       | Tokenizer object to perform tokenization on provided input text                | python function | --                          |
|   regexes_to_ignore   | List of regex to ignore in our input data points                               | list            | ["$[A-Z]+"]                 |
|      ignore_case      | Boolean flag to indicate whether we need to ignore case                        | boolean         | false                |
|  ignore_punctuation   | Boolean flag to indicate whether we need to ignore punctuation                 | boolean         | false                |
|    ignore_numbers     | Boolean flag to indicate whether we need to ignore numbers                     | boolean         | false               |
|        lang           | String of two letters indicating the language of the sentences, in ISO 639-1 format. (default="en") | string | "en" |
|      model_type       | String specifying which model to use, according to the BERT specification. (default="microsoft/deberta-large") | string | "microsoft/deberta-large" |
|          idf          | Boolean flag to use idf weights during computation of BERT score. (default=False) | boolean | false |
| rescale_with_baseline | Boolean flag to rescale BERTScore with the pre-computed baseline. (default=True)  | boolean | true |

### List of supported metrics:

- ada_similarity
- bertscore
- exact_match
- f1_score
- gpt_coherence
- gpt_fluency
- gpt_groundedness
- gpt_relevance
- gpt_similarity
- llm_coherence
- llm_fluency
- llm_groundedness
- llm_relevance
- llm_similarity