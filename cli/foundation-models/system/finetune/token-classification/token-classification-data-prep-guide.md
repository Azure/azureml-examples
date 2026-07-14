# `Data preparation`

## 1. conll2003 dataset

The dataset has named entities tags, parts of speech tags and chunk tags. We use ner_tags for the purpose of this document. The dataset has 5 keys - `id`, `tokens`, `pos_tags`, `ner_tags` and `chunk_tags`. The keys `tokens` and `ner_tags` needs to be mentioned in the *parameters* section of [DataPreprocess component](../finetune/docs/component_docs/preprocess_component.md). The dataset could contain additional keys along with `tokens` and `ner_tags` which will be ignored during data preprocessing. In the conll2003 dataset, `id`, `pos_tags` and `chunk_tags` are ignored during preprocessing.

Below is the script to download the data from Hugging Face and save it in the **jsonl** format

```python

    # install Hugging Face datasets package using `pip install datasets==2.3.2`
    from datasets import load_dataset
    from functools import partial

    def format_ner_tags(example, class_names):
        example.pop("pos_tags")
        example.pop("chunk_tags")
        example["ner_tags_formatted"] = [class_names[ner_id] for ner_id in example["ner_tags"]]
        return example

    dataset_name = "conll2003"
    split = "test"  # change the split to download train and validation data
    ds = load_dataset(dataset_name, split=split)
    ds = ds.map(
        partial(
            format_ner_tags,
            class_names=ds.features["ner_tags"].feature.names
        )
    )
    ds.to_json("{}_{}.jsonl".format(dataset_name, split))

```

## 2. Dataset filter

During data preprocessing, few or more lines of the jsonl file gets filtered if the value of the keys are in the invalid values shown below

|Column Name|Invalid values|
|---|---|
|**tokens_key**|None, [](empty list)|
|**tag_key**|None, [](empty list)|


## 3. Accepted data types

Currently 4 types of datasets are supported - CSV, JSONL, paraquet, MLTable

The Azureml finetune components accepts two keys, referred to as `token_key`, `tag_key` in the component parameters. The table below shows the accepted data types for the value of each of the keys.

|token_key|tag_key|
|---|---|
|List[string]|List[string]|

To interpret this in the above conll2003 dataset, **tokens** (*token_key*) should be of list of string dtype and **ner_tags_formatted** (*tag_key*) should also be of list of string dtype.


## 4. Example

### Here is an example of how the data should look like

Token classification requires the training data to include 2 fields, ‘Tokens’ and ‘Tags’. The tags could contain any strings depending on the finetune use case. Please note that the NER tags should be passed as an array of strings. 

| Tokens (Tokens) | NER Tags (Tags) |
| :- | :- |
| ["Results","of","French","first","division"] | ["O","O","B-MISC","O","O"] |
| ["Nippon","Telegraph","and","Telephone","Corp","(","NTT",")","said","on","Friday","that","it","hopes","to","move","into","the","international","telecommunications","business","as","soon","as","possible","following","the","government","'s","decision","to","split","NTT","into","three","firms","under","a","holding","company","."] | ["B-ORG","I-ORG","I-ORG","I-ORG","I-ORG","O","B-ORG","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","B-ORG","O","O","O","O","O","O","O","O"] |