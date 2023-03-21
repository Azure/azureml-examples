
# Tools

- [sample_idxs_to_text.py](./sample_idxs_to_text.py) - want to see which text was feed at specific iterations? for example to understand why the training went astray? Then use this script. The pre-amble of the script contains the documentation and usage examples.


## A few notes on how we created the datasets:

### Creating the Json Lines text file

First you need to create a jsonl file containing your dataset. For this we exported from the HF-datasets format. For example for C4:

```
from datasets import load_dataset
c4 = load_dataset("c4", "en")
c4["train"].to_json("c4_en_train.jsonl")
c4["validation"].to_json("c4_en_valid.jsonl")
```

This creates quite a large file compared to the size of the HF dataset on disk (810GB vs 305 for C4 for example)

### Megatron pre-processing

Then you need to pass that text file to the `preprocess_data.py` script for tokenization and memory-mapping, creating two files, one to store the tokens indices and one to store the document start and ends. The result will be slightly bigger than the text dataset. (360GB vs 305GB for C4 for example). You can choose one of the default Megatron tokenizers (but then you have to pass merges and vocab files) or one from HF-tokenizers. For example, in our GPT-like models reusing a T5 sentencepiece-bpe tokenizer:

`python tools/preprocess_data.py   --input ~/c4_en_train.jsonl        --output-prefix c4_en_train --dataset-impl mmap        --tokenizer-type PretrainedFromHF --tokenizer-name-or-path t5-small        --workers 30        --append-eod`

Do note that adding too many workers can be counterproductive for very large dataset: as the bottleneck becomes disk writing, the intermediary process results pool up and can flood the RAM. In our experiments on GCP machines, running with 60 workers on C4 inevitably led the program to fail.
