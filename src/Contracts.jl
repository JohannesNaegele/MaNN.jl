abstract type AbstractLayer end

"""
A layer is a function which produces some output/prediction given some input values.
"""
function (layer::AbstractLayer)(input) end

"""
Returns the loss and the corresponding gradient.
We need this because we don't have automatic differentiation!
"""
function withgradient(layer::AbstractLayer, loss(d)) end

"""
    update!(layer, grad, opt)
Perform gradient descent on the layer given a gradient with an optimizer (which chooses e.g. the learning rate).
"""
function update!(layer::AbstractLayer, grad, opt) end # example