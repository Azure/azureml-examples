import datasets


def get_dataset(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("dair-ai/emotion", split=split)
    dataset = dataset.map(
        lambda sample: tokenizer(
            sample["text"],
            truncation=True,
            padding="max_length",
            max_length=dataset_config.max_input_length,
        ),
        batched=True,
        remove_columns=["text"],
    )
    return dataset
