from datasets import *
from transformers import *
from tokenizers import *
import os
import json


if __name__ == "__main__":

    parser = HfArgumentParser(TrainingArguments)
    training_args, args = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )

    wikiit = load_dataset("wikipedia", "20220301.it", split="train")
    dataset = wikiit

    d = dataset.train_test_split(test_size=0.1)
    d["train"], d["test"]

    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"]
    # 30,522 vocab is BERT's default vocab size, feel free to tweak
    vocab_size = 30_522
    # maximum sequence length, lowering will result to faster training (when increasing batch size)
    max_length = 512

    model_path = "pretrained-bert"

    # make the directory if not already there
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    # dumping some of the tokenizer config to config file,
    # including special tokens, whether to lower case and the maximum sequence length
    with open(os.path.join(model_path, "config.json"), "w") as f:
        tokenizer_cfg = {
            "do_lower_case": True,
            "unk_token": "[UNK]",
            "sep_token": "[SEP]",
            "pad_token": "[PAD]",
            "cls_token": "[CLS]",
            "mask_token": "[MASK]",
            "model_max_length": max_length,
            "max_len": max_length,
        }
        json.dump(tokenizer_cfg, f)

    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    truncate_longer_samples = True

    def encode_with_truncation(examples):
        """Mapping function to tokenize the sentences passed with truncation"""
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_special_tokens_mask=True,
        )

    def encode_without_truncation(examples):
        """Mapping function to tokenize the sentences passed without truncation"""
        return tokenizer(examples["text"], return_special_tokens_mask=True)

    # the encode function will depend on the truncate_longer_samples variable
    encode = (
        encode_with_truncation if truncate_longer_samples else encode_without_truncation
    )
    # tokenizing the train dataset
    train_dataset = d["train"].map(encode, batched=True)
    # tokenizing the testing dataset
    test_dataset = d["test"].map(encode, batched=True)

    model_config = BertConfig(vocab_size=vocab_size, max_position_embeddings=max_length)
    model = BertForMaskedLM(config=model_config)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # Perfrom pre-training and save the model
    result = trainer.train()
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")

    # compute the number of floating-point operations per forward pass
    # calculate FLOPs for embeddings layer
    embedding_size = 768
    sequence_length = 512
    embedding_flops = (
        embedding_size * sequence_length * 2
    )  # 2 FLOPs for each multiplication and addition operation

    # calculate FLOPs for transformer layers
    hidden_size = 768
    num_layers = 12
    transformer_flops = (
        hidden_size * sequence_length * 3 * 2 * num_layers
    )  # 3 matrix multiplications, 2 layer norm ops

    # calculate total FLOPs
    flops_per_pass = embedding_flops + transformer_flops

    # compute the number of forward passes per second
    compute_training_samples_per_second = result.metrics["train_samples_per_second"]
    forward_passes_per_second = (
        compute_training_samples_per_second / training_args.per_device_train_batch_size
    )

    # compute the number of floating-point operations per second
    flops_per_second = flops_per_pass * forward_passes_per_second

    # compute the number of teraflops
    tflops = flops_per_second / 1e12

    # print the number of teraflops
    print(f"Estimated teraflops: {tflops:.2f} TFLOPS")
