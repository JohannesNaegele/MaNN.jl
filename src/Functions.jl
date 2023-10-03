relu(x, α=0.1) = (x > 0) ? x : 0

function leakyrelu(x, α=0.1; derivative=false)
    if !derivative
        return (x > 0) ? x : x * α
    else
        return (x > 0) ? 1 : α
    end
end

function cross_entropy(y_pred::AbstractArray, y_true::AbstractArray; derivative=false)
    if !derivative
        -sum(y_true .* log.(y_pred))
    else
        y_pred - y_true
    end
end

function softmax(z::AbstractArray)
    exp_z = exp.(z .- maximum(z)) # important, otherwise we might get float overflow!
    return exp_z ./ sum(exp_z)
end

# TODO: write more loss/activation functions