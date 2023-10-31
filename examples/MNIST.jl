using MaNN # Tipp: Revise
using MLDatasets
using Plots
using Random
Random.seed!(1)

# note: this data is already normalized to be between 0.0 and 1.0
trainset = MNIST(:train)
testset = MNIST(:test) # mit den Testdaten testen wir, wie gut unser neuronales Netz ist

# TODO: write proper struct with less data usage
###? Indikatorfunktion
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
    accuracy = correct / total  # Compute accuracy(Genauigkeit) correct predictions im Verhältnis zu total Anzahl von Beispielen
    # accuracy gibt uns an, wie viel Prozent der Daten richtig vorhergesagt werden

    return accuracy
end

# konstruiere neuronales Netz mit 3 hidden layern
# 1.hidden layer hat 784 Neuronen, 2. 256 und 3. 128
# es gibt 10 output-Werte, Zahlen von 0-9 
my_first_net = Chain(
    Dense(784 => 256, MaNN.leakyrelu),
    Dense(256 => 128, MaNN.leakyrelu),
    Dense(128 => 10, MaNN.leakyrelu),
    MaNN.softmax
)

# neuronales Netz wird mit random Werten durchgelaufen 
model = my_first_net(rand(784))

# TODO: use episodes
# TODO: implement train with autodiff

# @time ist zur Leistungsmessung
#################################################
## implementiere jeweils 3 Gewichts- und 3 Biasmatrizen
# Define the neural network architecture
using Zygote

function backpropagate!(model, data, y_true, learning_rate)
    for data in trainset
        data = trainset.features
        ableitung = gradient(model) do m
            result = m(input)
            loss = MaNN.cross_entropy(y_true, result)
            return loss
            print(loss)
        end
        #Gewichte updaten
        Dense.weights -= learning_rate*ableitung[weights]
        Dense.biases -= learning_rate*ableitung[biases]
    end
end

backpropagate!(model, trainset.features)
#################################################

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

##############################################
using Plots
heatmap(trainset[3].features, c=:grays)



function visualize_results(model, data)
    num_samples = 10  # Number of samples to visualize
    selected_indices = rand(1:length(data[1]), num_samples)
    plot_array = []
    for i in selected_indices
        x = data[1][:, i]
        y_true = data[2][i]
        y_pred = onecold(model(x))

        img = reshape(x, 28, 28)
        title = "True: "+y_true+" Predicted: "+y_pred

        push!(plot_array, heatmap(img, color=:grays, title=title))
    end

    plot(plot_array, layout=(2, 5))
end

for i in eachindex(testset.targets)
    visualize_results(my_first_net, vec(testset.features[ :, i]))
end

for i in eachindex(testset.targets)
    misclassified_data = validate(my_first_net, vec(testset.features[ :, i]))
    heatmap(testset[misclassified_data].features)
end
    
num_examples_to_visualize = 10
plot_array = []
for (y_true, true_label, y_pred) in misclassified_data[1:num_examples_to_visualize]
    push!(plot_array, heatmap(reshape(x, 28, 28), color=:grays, title="True: $true_label, Predicted: $y_pred"))
end


##############################################

# Demo for profiler
@profview train_hardcoded!(
    my_first_net,
    cross_entropy,
    # TODO: write proper flatten function
    [(vec(trainset.features[:, :, i]), onehotbatch(trainset.targets[i], 0:9)) for i in eachindex(trainset.targets[1:1000])],
    BoringOptimizer(0.01)
)