# `Data preparation`

## 1. Xsum Dataset

The dataset has the documents and summaries of the news articles from BBC. The dataset has 3 keys - `document`, `summary`, `id`. The keys `document` and `summary`, or any customized versions of the keys, needs to be mentioned in the *parameters* section of [DataPreprocess component](../finetune/docs/component_docs/preprocess_component.md).The dataset could contain additional keys along with `document` and `summary` which will be ignored during data preprocessing. For instance, `id` key will be ignored during preprocessing.

Below is the script to download the data from Hugging Face and save it in the **jsonl** format

```python

    # install Hugging Face datasets package using `pip install datasets==2.3.2`
    from datasets import load_dataset

    dataset_name = "xsum"
    split = "test"  # change the split to download train and validation data
    ds = load_dataset(dataset_name, split=split)
    ds.to_json("{}_{}.jsonl".format(dataset_name, split))

```

## 2. Dataset filter

During data preprocessing, few or more lines of the jsonl file gets filtered if the value of the keys are in the invalid values shown below

|Column Name|Invalid values|
|---|---|
|**document_key**|None, ""(empty string)|
|**summary_key**|None, ""(empty string)|

## 3. Accepted data types

Currently 4 types of datasets are supported - CSV, JSONL, paraquet, MLTable

The Azureml finetune components accepts two keys, referred to as `document_key` and `summary_key` in the component parameters. The table below shows the accepted data types for the value of each of the keys.

|document_key|summary_key|
|---|---|
|string|string|

To interpret this in the above xsum dataset, **document** (*document_key*) should be of string dtype and **summary** (*summary_key*) should also be of string dtype.

## 4. Example

##### Here is an example of how the data should look like

The summarization dataset is expected to have 2 fields – document, summary like shown below.

| Article (Document) | Highlights (Summary) |
| :- | :- |
| (CNN) -- Former baseball slugger Jose Canseco accidentally shot himself in his left finger while cleaning a gun, police said. He was in surgery Tuesday night, his fiancee tweeted. \"This is Leila . Thank you all for the kind words and prayers . Jose is in still surgery and will be ok. Please pray for his finger !!,\" she said in a tweet posted to his account. | Canseco hit more than 450 home runs .\nHis semiautomatic handgun accidentally went off . |
| (CNN) -- Zlatan Ibrahimovic scored all four goals in Sweden's 4-2 win over England -- but his final shot was something special. His audacious overhead volley from 30 yards was labeled on social networking sites as the greatest ever soccer goal. What do you think? Share your views on Ibrahimovic's wonder goal. | Zlatan Ibrahimovic scores a 30-yard overhead kick .\nIbrahimovic scored all four goals for Sweden . |