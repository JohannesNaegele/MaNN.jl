using Zygote
using MaNN

struct Foo
    A
end

foo = Foo(rand(2, 2))
bar = Chain([Foo(rand(2, 2)), Foo(rand(1, 2))])
(model::Foo)(x) = model.A * x
bar([1, 2])

# implicit version with the following Params container
# you can basically copy paste this with minor changes into the train! function
grad = gradient(() -> sum(foo([1, 1])), Params([foo.A]))
grad[foo.A]
grad = gradient(() -> bar([1, 1])[1], Params([l.A for l in bar.layers]))
[grad[l.A] for l in bar.layers]

# This is the explicit method fyi and is used recently more often.
# But for our purposes the approach from above is sufficient and probably the easiest way.
# see https://fluxml.ai/Zygote.jl/stable/#Explicit-and-Implicit-Parameters-1
grad = gradient(model -> sum(model([1, 1])), foo)
grad[1][:A]
grad = gradient(model -> model([1, 1])[1], bar)