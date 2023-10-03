struct BoringOptimizer <: AbstractOptimizer
    learning_rate
end

function update!(opt::BoringOptimizer, params, grad) 
    for i in eachindex(params)
        params[i] -= opt.learning_rate .* grad[i]
    end
end