function gradient_hardcoded(chain::Chain, loss, label, input, gs)
    # forward propagation: all outputs of layers are calculated
    for i in eachindex(chain.layers)
        input[i + 1] = chain.layers[i](input[i])
    end
    
    # backpropagation:
    # calculate derivative of loss function
    δ = loss(input[end], label, derivative=true)
    # calculate derivatives inside of layers
    for i in rev(eachindex(chain.layers))
        δ = δ .* layer.activation(input[i], derivative=true)
        gs[i] = δ * input', δ
        δ = layer.weights' * δ
    end
end

"""
This is how training with backpropagation *by hand* works.
Note, however, that we can use this only on our type `Chain` filled with `Dense` layers and `cross_entropy` loss.
The basic problem is that for the application of the chain rule we need
- all intermediate results produced by individual layers
- to know the explicit derivative of our activation function(s) and the loss function
"""
function train_hardcoded!(chain::Chain, data, loss)
    # preallocation for speedup
    ps = params(Chain)
    gs = similar(ps) # gradients will have the same shape as our parameters
    os = [similar(l.bias) for l in Chain.layers] # outputs of our layers is like bias
    for (input, label) in enumerate(data)
        gradient_hardcoded(chain, loss, label, [input, os...], gs)
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