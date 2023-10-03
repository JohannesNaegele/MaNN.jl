struct Chain{T <: Vector{S}, S <: AbstractLayer}
    layers::T
end

function (chain::Chain)(input::AbstractArray)
    output = deepcopy(input)
    for layer in Chain.layers
        output = layer(output)
    end
    return output
end