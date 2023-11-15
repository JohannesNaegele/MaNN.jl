using .MaNN # Tipp: Revise
using MLDatasets
using Plots
using Random
Random.seed!(1)

# note: this data is already normalized to be between 0.0 and 1.0

trainset = MNIST(:train)
testset = MNIST(:test)

struct OneHot1{T} <: AbstractVector{T}
    num_labels::Int
    true_label::Int
end

Base.size(oh::OneHot1{T}) where T = (oh.num_labels,)
Base.getindex(oh::OneHot1{T}, i) where T = i == oh.true_label ? one(T) : zero(T)
# Base.setindex!(oh::OneHot1{T}, value, i) where {T} = (i == oh.true_label) && (value == one(T)) ? nothing : error("Cannot modify true label")
Base.length(oh::OneHot1{T}) where T = oh.num_labels
Base.keys(oh::OneHot1{T}) where T = [oh.true_label]

# Define getindex for range
function Base.getindex(oh::OneHot1{T}, r::AbstractUnitRange{T}) where T
    return [i == oh.true_label ? one(T) : zero(T) for i in r]
end

# Update onehot function
function onehot1(input, labels=0:9)
    true_label = findfirst(x -> x == input, labels)
    return OneHot1{Int}(length(labels), true_label)
end

a = onehot1(2, 0:9)
a[1:3]


# Update onehotbatch function
function onehotbatch(input, labels=0:9)
    return [onehot1(x, labels) for x in input]
end

# Update onecold function
function onecold(y, labels=0:9)
    true_label = argmax(y)
    return findfirst(x -> x == true_label, labels)
end    

Base.summarysize(onehot1(3, 0:9))

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

# Convert OneHot1 to array
#function convert_to_array(oh::OneHot1)
#    return [i == oh.true_label ? 1.0 : 0.0 for i in 1:oh.num_labels]
#end

# Inside the scope where train_hardcoded! is called
#data = [
#    (vec(trainset.features[:, :, i]), convert_to_array(onehot1(trainset.targets[i], 0:9)))
#    for i in eachindex(trainset.targets)
#]

#@time train_hardcoded!(my_first_net, cross_entropy, data, BoringOptimizer(0.01))

#accuracy = validate(
#    my_first_net,
#    [(vec(testset.features[:, :, i]), onehotbatch(testset.targets[i], 0:9)) for i in eachindex(testset.targets)]
#)
#println("Accuracy: $accuracy")

# TODO: visualize images
function plot_image(feature) end
# TODO: plot metrics
#@profview train_hardcoded!(
#    my_first_net,
#    cross_entropy, data, BoringOptimizer(0.01))


data = [
        (vec(trainset.features[:, :, i]), onehot1(trainset.targets[i], 0:9))
        for i in eachindex(trainset.targets)
    ]
    
@time train_hardcoded!(my_first_net, cross_entropy, data, BoringOptimizer(0.01))
accuracy = validate(
    my_first_net,
    [(vec(testset.features[:, :, i]), onehotbatch(testset.targets[i], 0:9)) for i in eachindex(testset.targets)]
)

@profview train_hardcoded!(
    my_first_net,
    cross_entropy, data, BoringOptimizer(0.01))


# Struct containing all necessary info
mutable struct Adam1
    theta::AbstractArray{Float64} # Parameter array
    loss::Function                # Loss function
    grad::Function                # Gradient function
    m::AbstractArray{Float64}     # First moment
    v::AbstractArray{Float64}     # Second moment
    b1::Float64                   # Exp. decay first moment
    b2::Float64                   # Exp. decay second moment
    a::Float64                    # Step size
    eps::Float64                  # Epsilon for stability
    t::Int                        # Time step (iteration)
end
  
  # Outer constructor
function Adam1(theta::AbstractArray{Float64}, loss::Function, grad::Function)
    m   = zeros(size(theta))
    v   = zeros(size(theta))
    b1  = 0.9
    b2  = 0.999
    a   = 0.001
    eps = 1e-8
    t   = 0
    Adam1(theta, loss, grad, m, v, b1, b2, a, eps, t)
end
  
  # Step function with optional keyword arguments for the data passed to grad()
function step!(opt::Adam1; data...)
    opt.t += 1
    gt    = opt.grad(opt.theta; data...)
    opt.m = opt.b1 .* opt.m + (1 - opt.b1) .* gt
    opt.v = opt.b2 .* opt.v + (1 - opt.b2) .* gt .^ 2
    mhat = opt.m ./ (1 - opt.b1^opt.t)
    vhat = opt.v ./ (1 - opt.b2^opt.t)
    opt.theta -= opt.a .* (mhat ./ (sqrt.(vhat) .+ opt.eps))
end

params_chain = MaNN.params(my_first_net)
adam_optimizer = Adam1(collect(MaNN.params(params_chain)), cross_entropy, gradient_hardcoded)
num_epochs = 10  # Adjust as needed

for epoch in 1:num_epochs
    for (x, y_true) in data
        gradients = similar(params_chain)

        # Forward propagation
        h1 = MaNN.leakyrelu(my_first_net.layers[1](x))
        h2 = MaNN.leakyrelu(my_first_net.layers[2](h1))
        h3 = MaNN.leakyrelu(my_first_net.layers[3](h2))
        y_pred = MaNN.softmax(my_first_net.layers[4](h3))

        # Backpropagation
        δ = cross_entropy(y_pred, y_true, derivative=true)

        gradients[7], gradients[8] = δ * h3', δ
        δ = my_first_net.layers[3].weights' * δ .* MaNN.leakyrelu'(h2)

        gradients[5], gradients[6] = δ * h2', δ
        δ = my_first_net.layers[2].weights' * δ .* MaNN.leakyrelu'(h1)

        gradients[3], gradients[4] = δ * h1', δ
        δ = my_first_net.layers[1].weights' * δ .* MaNN.leakyrelu'(x)

        step!(adam_optimizer; data...)
    end
end


# Train the model for 10 epochs
for epoch in 1:num_epochs
    for (x, y) in data
        # Perform one training step using the Adam1 optimizer
        step!(adam_optimizer, data=(x, y))
    end
end