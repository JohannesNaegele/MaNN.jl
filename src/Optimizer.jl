struct BoringOptimizer <: AbstractOptimizer
    learning_rate
end

"""
The type/layout of params might get changed in the future.
"""
function update!(opt::BoringOptimizer, params, grad)
    for i in eachindex(params)
        for j in eachindex(params[i])
            params[i][j] .-= opt.learning_rate .* grad[i][j]
        end
    end
end