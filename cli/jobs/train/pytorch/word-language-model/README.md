# Word-level language modeling

This example shows how to train a word-level language model on the Wikitext-2 dataset with PyTorch on Azure ML.

The training code is adapted from the official [pytorch/examples](https://github.com/pytorch/examples/tree/master/word_language_model) repo. For more information on using the provided training scripts, see `./src/README.md`.

## Running the example

### Create an Azure ML dataset from the provided local data
The Wikitext-2 training data is provided in `./src/data/wikitext-2`.

The following command will upload the data to your workspace's default datastore and create a dataset with the specified name and version. To use a different datastore, use the `--datastore` argument. 

```
az ml data upload --name wikitext-2 --version 1 --path ./data/wikitext-2
```

### Create the training job
The example job specification is provided in `job.yml`.

Run the following command to create the job:

```
az ml job create --file job.yml
```