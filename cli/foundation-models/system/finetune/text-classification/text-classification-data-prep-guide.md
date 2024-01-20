# `Data preparation`

## 1. Data format
Your dataset should be similar to one of the following datasets
1. When you have only one sentence to be classified, it should be similar to [`CoLA` dataset](https://huggingface.co/datasets/glue/viewer/cola)
2. When you have two sentences to be classified (e.g., sentence entailment), then it should be similar to [`MRPC` dataset](https://huggingface.co/datasets/glue/viewer/qnli)

For your reference, below are dataset format of CoLA and MRPC

### 1.1 CoLA dataset

The dataset classifies whether the sentence is a grammatical  English sentence. The dataset has 3 keys - `sentence`, `label`, `id`. The keys `sentence` and `label`, or any customized versions of the keys, needs to be mentioned in the *parameters* section of [DataPreprocess component](../finetune/docs/component_docs/preprocess_component.md).The dataset could contain additional keys along with `sentence` and `label` which will be ignored during data preprocessing. For instance, `id` key will be ignored during preprocessing.

### 1.2 MRPC dataset

The dataset tests whether the input sentences, namely sentence1 and sentence2, are semantically equivalent. The dataset has 4 keys - `sentence1`, `sentence2`, `label`, `id`. The keys `sentence1`, `sentence2` and `label`, or any customized versions of the keys, needs to be mentioned in the *parameters* section of [DataPreprocess component](../finetune/docs/component_docs/preprocess_component.md).The dataset could contain additional keys along with `sentence1`, `sentence2` and `label` which will be ignored during data preprocessing. For instance, `id` key will be ignored during preprocessing.

Below is the script to download the CoLA or MRPC data from Hugging Face and save it in the **jsonl** format

```python

    # install Hugging Face datasets package using `pip install datasets==2.3.2`
    from datasets import load_dataset

    dataset_name = "cola"  # replace with "mrpc" to download mrpc dataset
    split = "train"  # change the split to download train and validation data
    ds = load_dataset("glue", dataset_name, split=split)
    ds.to_json("{}_{}.jsonl".format(dataset_name, split))
```

## 2. Dataset filter

During data preprocessing, few or more lines of the data gets filtered if the value of the keys are in the invalid values as shown below

|Column Name|Invalid values|
|---|---|
|**sentence1_key**|None, ""(empty string)|
|**sentence2_key**|None, ""(empty string)|
|**label_key**|None, ""(empty string)|

## 2.4. Accepted data types

Currently 4 types of datasets are supported - CSV, JSONL, paraquet, MLTable

The Azureml finetune components accepts 3 keys, referred to as `sentence1_key`, `sentence2_key` and `label_key` in the component parameters. The table below shows the accepted data types for the value of each of the keys.
|sentence1_key|sentence2_key|label_key|
|---|---|---|
|string|string|string or integer|

To interpret this in the above CoLA dataset, **sentence** (*sentence1_key*) should be of string dtype and **label** (*label_key*) should also be of string or int dtype. Similarly, in the MRPC dataset **sentence1** (*sentence1_key*) should be of string dtype, **sentence2** (*sentence2_key*) should be of string dtype, and **label** (*label_key*) should be of string or int dtype

## 4. Example

### Here is an example of how the data should look like

Single text classification requires the training data to include at least 2 fields – one for ‘Sentence1’ and ‘Label’ like in this example. Sentence 2 can be left blank in this case. The below examples are from Emotion dataset. 

| Text (Sentence1) | Label (Label) |
| :- | :- |
| i feel so blessed to be able to share it with you all | joy | 
| i feel intimidated nervous and overwhelmed and i shake like a leaf | fear | 

 

Text pair classification, where you have two sentences to be classified (e.g., sentence entailment) will need the training data to have 3 fields – for ‘Sentence1’, ‘Sentence2’ and ‘Label’ like in this example. The below examples are from Microsoft Research Paraphrase Corpus dataset. 

| Text1 (Sentence 1) | Text2 (Sentence 2) | Label_text (Label) |
| :- | :- | :- |
| Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence . | Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence . | equivalent |
| Yucaipa owned Dominick 's before selling the chain to Safeway in 1998 for $ 2.5 billion . | Yucaipa bought Dominick 's in 1995 for \$ 693 million and sold it to Safeway for \$ 1.8 billion in 1998 . | not equivalent |
