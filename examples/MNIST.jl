using MaNN # Tipp: Revise
using MLDatasets
using Plots
using Random
Random.seed!(1)

# note: this data is already normalized to be between 0.0 and 1.0
trainset = MNIST(:train)
testset = MNIST(:test)

# TODO: write proper struct with less data usage
function onehot(input, labels=0:9)
    return [(input == r) ? 1.0 : 0.0 for r in labels]
end

function onehotbatch(input, labels=0:9)
    return map(x -> onehot(x, labels), input)
end

function onecold(y, labels=0:9)
    indices = argmax(y, dims=1)
    return labels[indices[1]]
end

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

my_first_net = Chain(
    Dense(784 => 256, MaNN.leakyrelu),
    Dense(256 => 128, MaNN.leakyrelu),
    Dense(128 => 10, MaNN.leakyrelu),
    MaNN.softmax
)

my_first_net(rand(784))

# TODO: use episodes
# TODO: implement train with autodiff
@time train_hardcoded!(
    my_first_net,
    cross_entropy,
    # TODO: write proper flatten function
    [
        (vec(trainset.features[:, :, i]), onehotbatch(trainset.targets[i], 0:9))
        for i in eachindex(trainset.targets)
    ],
    BoringOptimizer(0.01)
)

accuracy = validate(
    my_first_net,
    [(vec(testset.features[:, :, i]), onehotbatch(testset.targets[i], 0:9)) for i in eachindex(testset.targets)]
)
println("Accuracy: $accuracy")

# TODO: visualize images
function plot_image(feature) end
# TODO: plot metrics

# Demo for profiler
@profview train_hardcoded!(
    my_first_net,
    cross_entropy,
    # TODO: write proper flatten function
    [(vec(trainset.features[:, :, i]), onehotbatch(trainset.targets[i], 0:9)) for i in eachindex(trainset.targets[1:1000])],
    BoringOptimizer(0.01)
)