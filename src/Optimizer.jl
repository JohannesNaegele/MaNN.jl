struct BoringOptimizer <: AbstractOptimizer
    learning_rate
end

"""
The type/layout of params might get changed in the future.
"""
function update!(opt::BoringOptimizer, params, grad)
    for i in eachindex(params)
        _update!(opt.learning_rate, params[i], grad[i])
    end
end

_update!(lr, ps, gs) = (ps .-= lr .* gs)