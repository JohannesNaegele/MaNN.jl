# use atodiff in layers
using Flux
f(x) = x^2
df(x) = gradient(f, x)[1]
df(1)


# Linear Regression
# using gradient to imporove the weigths
W = rand(2, 5)
b = rand(2)
predict(x) = W*x .+ b

function loss(x, y)
  ŷ = predict(x)
  sum((y .- ŷ).^2)
end

x, y = rand(5), rand(2) # Dummy data
loss(x, y) # ~ 3

gs = gradient(() -> loss(x, y), Flux.params(W, b))

W̄ = gs[W]
W .-= 0.1 .* W̄
loss(x, y) # ~ 2.5


# TODO Wy do we use backward diff instead of forward diff?
# -> it's faster 
# -> show that
# Zygote an ReverseDiff arw both reverse-mode Diff
using ForwardDiff
f(x::Real) = exp(x)^2
g =ForwardDiff.gradient(f, x)





# TODO write a function for training the weigths