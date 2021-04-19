# Deepspeed Transformers Example

This example uses [deepspeed](https://www.deepspeed.ai/) to train _large_ transformer
models - including training [Huggingface's](https://huggingface.co/transformers/pretrained_models.html)
`gpt2-xl` model with 1.6 billion parameters.

In this example deepspeed is configured through `ds_config.json` which is part
of the `src` directory. See [deepspeed documentation](https://www.deepspeed.ai/docs/config-json/) for a full description of possible configuration.

## Usage

The `job.py` script is configurable, allowing you to try other
[Hugginface models](https://huggingface.co/transformers/pretrained_models.html),
other [GLUE tasks](https://gluebenchmark.com/) both with and without deepspeed.

Running

```bash
python job.py
```

with its default settings will finetune `distilbert-base-uncased` model
on `cola` task using a (single node) 4 V100 GPUs compute target (`gpu-V100-4`).

The model and task are easy to change - see `job.py` for some other examples
- including `gpt2-xl` model with 1.6 billion parameters.

## What you can expect

Testing with `gpu-V100-4` compute target we were able to train `gpt2-xl` 1.6B parameter model
in a couple of hours.
