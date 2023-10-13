# TODO: write convolutional layer
# https://en.wikipedia.org/wiki/Convolution#Discrete_convolution
# https://en.wikipedia.org/wiki/Discrete_Fourier_transform#Circular_convolution_theorem_and_cross-correlation_theorem
# https://en.wikipedia.org/wiki/Fast_Fourier_transform

struct Convolution <: AbstractLayer
filter
input
end

function set_filter(conv::Convolution)(matrix)
    conv.filter = matrix
end

function set_

# TODO: (padding + striding)