Note: copied from https://github.com/FluxML/model-zoo/tree/master/other/iris

# Use Flux to do Logistic Regression on the Iris dataset

This is a very simple model, with a single layer that outputs to softmax.

Logistic regression can basically be thought of as a [single layer neural network](https://sebastianraschka.com/faq/docs/logisticregr-neuralnet.html).

## Data Source

The data source is Fisher's classic dataset, retrieved from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris).

## Usage

`cd` into `model-zoo/other/iris`, start the Julia REPL and instantiate the environment:

```julia

julia> using Pkg; Pkg.activate("."); Pkg.instantiate()

```

Then train and evaluate the model:

```julia

julia> include("iris.jl")
Starting training.

Accuracy: 0.94

Confusion Matrix:

3×3 Array{Int64,2}:
 16   0   0
  0  16   1
  0   2  15

julia>

```
