module MaNN

export gradient, gradient_hardcoded, train_hardcoded!
export Chain, Dense
export BoringOptimizer
export cross_entropy

include("Contracts.jl")
include("Functions.jl")
include("./layers/Dense.jl")
include("Chain.jl")
include("Optimizer.jl")
include("Train.jl")
include("Metrics.jl")

end # end module Mann