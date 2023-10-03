abstract type AbstractLayer end
abstract type AbstractOptimizer end

"""
    model = Dense(3 => 10)
    model([1, 2, 3])
A layer is a function which produces some output/prediction given input values.
"""
function (layer::AbstractLayer)(input) end

"""
    update!(params, grad, opt)
Perform gradient descent on parameters given a gradient with an optimizer (which chooses e.g. the learning rate).
Has to be implemented by an optimizer.
"""
function update!(opt::AbstractOptimizer, params, grad) end