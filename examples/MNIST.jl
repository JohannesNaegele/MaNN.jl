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
    indices = argmax(y, dims = 1)
    return labels[indices][1]
end

function validate(model, data)
    correct = 0  # Counter for correctly classified examples
    missclasified_indices = []

    for (i, (x, y_true)) in enumerate(data)
        y_pred = model(x)  # Forward propagate through the model
        predictions = onecold(y_pred)  # Get the predicted class labels
        actuals = onecold(y_true)  # Get the actual class labels
        correct += sum(predictions .== actuals)  # Count correct predictions
        if predictions!= actuals
            for index in range(0,100000)
                push!(missclasified_indices, [index, i, actuals, predictions])
            end
        end
    end

    total = length(data)  # Total number of examples
    accuracy = correct / total  # Compute accuracy(Genauigkeit) correct predictions im Verhältnis zu total Anzahl von Beispielen
    # accuracy gibt uns an, wie viel Prozent der Daten richtig vorhergesagt werden

    return accuracy, missclasified_indices
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

accuracy, misclassified_indices = validate(
    my_first_net,
    [(vec(testset.features[:, :, i]), onehotbatch(testset.targets[i], 0:9)) for i in eachindex(testset.targets)]
)
println("Accuracy: $accuracy")

# TODO: visualize images
function plot_image(feature) end
# TODO: plot metrics

##############################################
#visualize some of the misclassified numbers
using ImageView
title1 = "Original: $(misclassified_indices[1][3]), Predicted: $(misclassified_indices[1][4])"
imshow(testset[misclassified_indices[1][2]].features')
heatmap(testset[misclassified_indices[1][2]].features, c=:grays, title= title1, legend = false, axis=false)
title2 = "Original: $(testset[218].targets), Predicted: $(misclassified_indices[5][3])"
imshow(testset[218].features')
heatmap(testset[218].features, c=:grays, title= title2)

##############################################

# Demo for profiler
@profview train_hardcoded!(
    my_first_net,
    cross_entropy,
    # TODO: write proper flatten function
    [(vec(trainset.features[:, :, i]), onehotbatch(trainset.targets[i], 0:9)) for i in eachindex(trainset.targets[1:1000])],
    BoringOptimizer(0.01)
)