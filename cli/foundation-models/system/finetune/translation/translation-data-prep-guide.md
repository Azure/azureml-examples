# `Data preparation`

## 1. wmt16 dataset

The dataset has multiple versions based on the language pair, we are interested in i.e. source and the translated languages. We use the `Czech to English (cs-en)` version for the purpose of this document. The dataset has 2 keys - `cs`, `en`, whose values represents the text in Czech and English languages respectively. The keys `cs` and `en` needs to be mentioned in the *parameters* section of [DataPreprocess component](../finetune/docs/component_docs/preprocess_component.md). Please note that the keys of the dataset are not customizable and must follow the format accepted by the model. The model specific format information, described as _source_lang_ and _target_lang_, can be found [here](../finetune/docs/component_docs/preprocess_component.md/#35-translation). The dataset could contain additional keys along with `cs` and `en` which will be ignored during data preprocessing.

Below is the script to download the data from Hugging Face and save it in the **jsonl** format

```python

    # install Hugging Face datasets package using `pip install datasets==2.3.2`
    from datasets import load_dataset

    def format_data(example):
        return example.pop("translation")

    dataset_name = "wmt16"
    from_to_lang_code = "cs-en"
    split = "test"  # change the split to download train and validation data
    ds = load_dataset(dataset_name, from_to_lang_code, split=split)
    ds = ds.map(format_data)  # format the data row
    ds.to_json("{}_{}.jsonl".format(dataset_name, split))

```

## 2. Dataset filter

During data preprocessing, few or more lines of the jsonl file gets filtered if the value of the keys are in the invalid values shown below

|Column Name|Invalid values|
|---|---|
|**source_lang**|None, ""(empty string)|
|**target_lang**|None, ""(empty string)|

## 3. Accepted data types

Currently 4 types of datasets are supported - CSV, JSONL, paraquet, MLTable

The Azureml finetune components accepts two keys, referred to as `source_lang`, `target_lang` in the component parameters. The table below shows the accepted data types for the value of each of the keys.

|source_lang|target_lang|
|---|---|
|string|string|

To interpret this in the above wmt16, cs-en dataset, **cs** (*source_lang*) should be of string dtype and **en** (*target_lang*) should also be of string dtype. Also, note that the _source_lang_ and _target_lang_ can be interchanged.

## 4. Example

### Here is an example of how the data should look like

The translation dataset should have 2 fields – source language and target language. The field names that map to source and target languages need to be language codes supported by the model. Please refer to the model card for details on supported languages.

| en (Source_language) | Ro (Target_language) |
| :- | :- |
| Beethoven, Brahms, Bartok, Enescu were working people, artists, and not commercial representatives. | Beethoven, Brahms, Bartok, Enescu erau oameni care munceau, care erau artisti \u0219i nu reprezentanti comerciali. |
| Colleague Damien Collins MP attacked The Voice, saying that too wasn't original | Colegul Damien Collins a atacat The Voice, afirm\u00e2nd c\u0103 nici aceast\u0103 emisiune nu este original\u0103 |