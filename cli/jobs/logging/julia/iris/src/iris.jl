using Flux
using Flux: logitcrossentropy, normalise, onecold, onehotbatch
using Statistics: mean
using MLFlowLogger
using Logging
using Parameters: @with_kw

@with_kw mutable struct Args
    lr::Float64 = 0.5
    repeat::Int = 110
end

function get_processed_data(args)
    labels = Flux.Data.Iris.labels()
    features = Flux.Data.Iris.features()

    # Subract mean, divide by std dev for normed mean of 0 and std dev of 1.
    normed_features = normalise(features, dims=2)

    klasses = sort(unique(labels))
    onehot_labels = onehotbatch(labels, klasses)

    # Split into training and test sets, 2/3 for training, 1/3 for test.
    train_indices = [1:3:150 ; 2:3:150]

    X_train = normed_features[:, train_indices]
    y_train = onehot_labels[:, train_indices]

    X_test = normed_features[:, 3:3:150]
    y_test = onehot_labels[:, 3:3:150]

    #repeat the data `args.repeat` times
    train_data = Iterators.repeated((X_train, y_train), args.repeat)
    test_data = (X_test,y_test)

    return train_data, test_data
end

# Accuracy Function
accuracy(x, y, model) = mean(onecold(model(x)) .== onecold(y))

# Function to build confusion matrix
function confusion_matrix(X, y, model)
    ŷ = onehotbatch(onecold(model(X)), 1:3)
    y * transpose(ŷ)
end

function train(; kws...)
    # Initialize hyperparameter arguments
    args = Args(; kws...)	

    # Setting up the MLFlowLogger
    logger = MLFLogger()


    #Loading processed data
    train_data, test_data = get_processed_data(args)

    # Declare model taking 4 features as inputs and outputting 3 probabiltiies, 
    # one for each species of iris.
    model = Chain(Dense(4, 3))
	
    # Defining loss function to be used in training
    # For numerical stability, we use here logitcrossentropy
    loss(x, y) = logitcrossentropy(model(x), y)
	
    # Training
    # Gradient descent optimiser with learning rate `args.lr`
    optimiser = Descent(args.lr)

    function MLFCallback()
        with_logger(logger) do
          @info "train" loss=loss(first(iterate(train_data))...) acc=accuracy(first(iterate(train_data))..., model) log_step_increment=0
          @info "test" loss=loss(test_data...) acc=accuracy(test_data..., model)
        end
    end    

    println("Starting training.")
    Flux.@epochs 15 Flux.train!(loss, params(model), train_data, optimiser, cb = Flux.throttle(MLFCallback, 5))
	
    return model, test_data
end

function test(model, test)
    # Testing model performance on test data 
    X_test, y_test = test
    accuracy_score = accuracy(X_test, y_test, model)

    println("\nAccuracy: $accuracy_score")

    # Sanity check.
    @assert accuracy_score > 0.8

    # To avoid confusion, here is the definition of a Confusion Matrix: https://en.wikipedia.org/wiki/Confusion_matrix
    println("\nConfusion Matrix:\n")
    display(confusion_matrix(X_test, y_test, model))
end

cd(@__DIR__)
model, test_data = train()
test(model, test_data)
