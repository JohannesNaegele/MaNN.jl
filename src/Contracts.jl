abstract type NeuronalNetwork end
abstract type AbstractLayer end

function train!(net::NeuronalNetwork, data, loss) end

function predict(net::NeuronalNetwork, data) end

# struct MultiLayerPerceptron <: NeuronalNetwork
#     weights
#     activations
# end