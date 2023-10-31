struct Dense <: AbstractLayer
    # TODO: parametrize
    # input
    weights
    biases
    activation
end

# initialisiert random Werte in gewählten Grenzen
function init_glorot(in_size::Int, out_size::Int)
    limit = sqrt(6 / (in_size + out_size))
    return rand(-limit:limit:limit, out_size, in_size)
end

function Dense((in, out)::Pair{<:Integer, <:Integer}, activation=leakyrelu)
    # weights sind zufällige Werte in einer in init_glorot gewählten Grenze
    # wird hier eine Matrix mit so vielen Spalten wie input und so vielen Zeilen wie output erzeugt?
    weights = init_glorot(in, out)
    # biases ist ein Vektor von out vielen 0
    biases = zeros(out)
    return Dense(weights, biases, activation)
end

function (layer::Dense)(input::AbstractArray)
    # hier wird der forward pass durchgeführt
    return layer.activation.(layer.weights * input .+ layer.biases)
end

function params(layer::Dense)
    return [layer.weights, layer.biases]
end