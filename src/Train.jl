function gradient_hardcoded(chain::Chain, loss, label, input::Vector{Vector{T}}, gs) where T <: Number

    # forward propagation: all outputs of layers are calculated
    for i in eachindex(chain.layers)
        input[i + 1] .= chain.layers[i](input[i])
    end
    
    # backpropagation:
    # calculate derivative of loss function
    δ = loss(input[end], label, derivative=true) # Note: type instability, speedup e.g. ::Vector{Float32}
    # calculate derivatives inside of layers
    #Werden hier nur die Gradienten nach den Gewichten berechnet?
    for i in reverse(eachindex(chain.layers))
        layer = chain.layers[i]
        if layer isa AbstractLayer
            #delta wird elementweise multiplitiert mit Aktivierungsfkt(Output des Layers)
            δ .= δ .* layer.activation.(input[i + 1], derivative=true)
            # gs ist ein array, in dem die Gradienten gespeichert werden
            # gs[2*i-1] wird die Ableitung nach den Gewichten gespeichert
            # gs[2*i] wird die Ableitung nach den Bias gespeichert
            gs[2 * i - 1], gs[2 * i] = δ * input[i]', δ
            δ = layer.weights' * δ
        else # if it is softmax: do nothing
            error_msg = "This backpropagation method works only for a specific configuration."
            @assert (layer == softmax) error_msg
        end
    end
end

"""
This is how training with backpropagation *by hand* works.
Note, however, that we can use this only on our type `Chain` filled with `Dense` layers and `cross_entropy` loss.
The basic problem is that for the application of the chain rule we need
- all intermediate results produced by individual layers
- to know the explicit derivative of our activation function(s) and the loss function
"""
function train_hardcoded!(chain::Chain, loss, data::Vector{Tuple{S, T}}, opt) where {S, T}
    #opt is optimizer in this case BoringOptimizer(0.01)
    # preallocation for speedup
    ps = params(chain) #params ist ein Vektor mit allen Parametern des Models
    #gs ist eine Kopie von ps
    gs = deepcopy(ps) # gradients will have the same shape as our parameters
    #überprüft ob layers Gewichte und Bias hat, wenn ja, wird ein neues array implementiert mit dem gleichen Typ
    # und der gleichen Größe des Bias
    #wenn nein, wird ein neues array implementiert mit der gleichen Größe wie der BiasVektor des vorherigen Layers
    os = [chain.layers[i] isa AbstractLayer ? similar(chain.layers[i].biases) : similar(chain.layers[i - 1].biases)
        for i in eachindex(chain.layers)
    ] # outputs of our layers is like bias
    #hier werden die Gradienten berechnet und die Parameter aktualisiert
    for (input, label) in data
        # println(gs)
        gradient_hardcoded(chain, loss, label, [deepcopy(input), os...], gs)
        update!(opt, ps, gs)
    end
end

"""
This is proper training: With autodiff we just need to
- insert the parameters without knowledge of the model
- use a general loss function without knowledge of what exactly happens internally
"""
function train!(ps, data, loss)
    for (i, d) in enumerate(data)
        # TODO: batching
        losses, gradients = withgradient(chain, loss(d))
        # TODO: check whether losses is finite i.e. use withgradient function
        update!(opt, ps, gradients)
    end
end