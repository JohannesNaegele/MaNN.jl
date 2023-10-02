using MaNN # Tipp: Revise
using MLDatasets
using Plots

trainset = MNIST(:train)
testset = MNIST(:test)

# TODO: write proper struct with less data usage
function onehot(input, labels=0:9)
    return [input == r ? 0.0 : 1.0 for r in labels]
end

function onecold(y, labels=0:9)
    indices = argmax(y, dims=1)
    return labels[indices[1]]
end

function deep_neural_network(sizes, activations)
    return Chain(
        [Dense(sizes[i], activations[i]) for i in eachindex(sizes)]
    )
end

# TODO: test on test data
function validate(model, data)
    correct = 0  # Counter for correctly classified examples

    for (x, y_true) in data
        y_pred = model(x)  # Forward propagate through the model
        predictions = onecold(y_pred)  # Get the predicted class labels
        actuals = onecold(y_true)  # Get the actual class labels
        correct += sum(predictions .== actuals)  # Count correct predictions
    end

    total = length(data)  # Total number of examples
    accuracy = correct / total  # Compute accuracy

    return accuracy
end

my_first_net = deep_neural_network([10, 512, 10], [leakyrelu, leakyrelu, softmax])

# TODO: train
# note: this is comprehensive notation for an array of tuples
train_hardcoded!(my_first_net, loss, [(trainset.features, onehot.(trainset.targets, 0:9))])

validate(my_first_net, [(testset.features, onehot.(testset.targets, 0:9))])

# TODO: plot metrics