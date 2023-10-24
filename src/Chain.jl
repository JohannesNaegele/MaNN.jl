struct Chain{T <: AbstractVector}
    layers::T
end

Chain(args...) = Chain([args...])

function (chain::Chain)(input::AbstractArray)
    output = deepcopy(input)
    for layer in chain.layers
        output = layer(output)
    end
    return output
end

params(x) = Float64[]

"""
This function returns the parameters of our layers.
"""
function params(chain::Chain{T}) where T
    return vcat([params(layer) for layer in chain.layers]...)
end