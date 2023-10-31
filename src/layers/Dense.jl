struct Dense <: AbstractLayer
    # TODO: parametrize
    # input
    weights
    biases
    activation
end

function init_glorot(in_size::Int, out_size::Int)
    limit = sqrt(6 / (in_size + out_size))
    return rand(-limit:limit:limit, out_size, in_size)
end

function Dense((in, out)::Pair{<:Integer, <:Integer}, activation=leakyrelu)
    weights = init_glorot(in, out)
    biases = zeros(out)
    return Dense(weights, biases, activation)
end

function (layer::Dense)(input::AbstractArray)
    return layer.activation.(layer.weights * input .+ layer.biases)
end

function params(layer::Dense)
    return [layer.weights, layer.biases]
end