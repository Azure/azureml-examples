# Train a model

## Prerequisites
Ensure you have created an environment in Azure ML for running R jobs.

```bash
cd examples/r
az ml environment create --file environment.yaml
```

Upload and register the iris dataset to Azure ML using:

```bash
cd examples/r/train-model
az ml data upload -n iris -v 1 --path ./data
```

## Run the job

To run the job use:

```bash
cd examples/r/train-model
az ml job create --file job.yml --name $(uuidgen) --stream
```

## Notes

This job uses the optparse package to pass parameters via RScript. 
