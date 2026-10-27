# `Data preparation`

## 1. SQuAD Dataset

The [`Standford Question Answering Dataset (SQuAD)`](https://huggingface.co/datasets/squad) reading comprehension dataset consists of questions from a set of wikipedia articles where the answer is a segment of text from the passage. The dataset has 5 keys - `question`, `context`, `answers`, `id` and `title`. Aditionally, `answers` key has `start_key` and `text` as its sub-keys. The keys `question` and `context`, `answers`, `answer_start` and `text`, or any customized versions of the keys, needs to be mentioned in the *parameters* section of [DataPreprocess component](../finetune/docs/component_docs/preprocess_component.md). The dataset could contain additional keys along with `question`, `context`, `answers` with `answer_start` and `text` sub keys, which will be ignored during data preprocessing. For instance, `id` and `title` keys will be ignored during preprocessing.

This is an extractive question answering task i.e. the answer is present in the context.

Below is the script to download the data from Hugging Face and save it in the **jsonl** format

```python
    # install Hugging Face datasets package using `pip install datasets==2.3.2`
    from datasets import load_dataset

    dataset_name = "squad"
    split = "validation"  # change the split to download train data
    ds = load_dataset(dataset_name, split=split)
    ds.to_json("{}_{}.jsonl".format(dataset_name, split))

```

## 2. Accepted data types

Currently 4 types of datasets are supported - CSV, JSONL, paraquet, MLTable

The Azureml finetune components accepts 5 keys, referred to as `question_key`, `context_key`, `answers_key`, `answer_start_key` and `text_key` in the component parameters. The table below shows the accepted data types for the value of each of the keys.

|question_key|context_key|answers_key|answer_start_key|text_key|
|---|---|--|--|--|
|string|string|dict|List[int]|List[string]

To interpret this in the above squad dataset, **question** (*question_key*) should be of string dtype, **context** (*context_key*) should also be of string dtype, **answers** (*answers_key*) should be a python dictionary, **answer_start**(*answer_start_key*) should be a list of integer(s), **text**(*text_key*) should be a list of string(s).

## 3. Example

#### Here is how the dataset should look like

| Question | Context | Answers |
| :- | :- | :- |
| What does Phosphorylation do? | After a chloroplast polypeptide is synthesized on a ribosome in the cytosol, an enzyme specific to chloroplast proteins phosphorylates, or adds a phosphate group to many (but not all) of them in their transit sequences. Phosphorylation helps many proteins bind the polypeptide, keeping it from folding prematurely. This is important because it prevents chloroplast proteins from assuming their active form and carrying out their chloroplast functions in the wrong place\u2014the cytosol. At the same time, they have to keep just enough shape so that they can be recognized by the chloroplast. These proteins also help the polypeptide get imported into the chloroplast. | {"text”: ["helps many proteins bind the polypeptide","helps many proteins bind the polypeptide", "helps many proteins bind the polypeptide"], "answer_start": [236,236,236]} |
| What is the basic unit of organization within the UMC? | The Annual Conference, roughly the equivalent of a diocese in the Anglican Communion and the Roman Catholic Church or a synod in some Lutheran denominations such as the Evangelical Lutheran Church in America, is the basic unit of organization within the UMC. The term Annual Conference is often used to refer to the geographical area it covers as well as the frequency of meeting. Clergy are members of their Annual Conference rather than of any local congregation, and are appointed to a local church or other charge annually by the conference's resident Bishop at the meeting of the Annual Conference. In many ways, the United Methodist Church operates in a connectional organization of the Annual Conferences, and actions taken by one conference are not binding upon another. | {"text": ["The Annual Conference","synod","The Annual Conference"],"answer_start": [0,120,0]} |