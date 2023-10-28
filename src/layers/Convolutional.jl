# TODO: write convolutional layer
# https://en.wikipedia.org/wiki/Convolution#Discrete_convolution
# https://en.wikipedia.org/wiki/Discrete_Fourier_transform#Circular_convolution_theorem_and_cross-correlation_theorem
# https://en.wikipedia.org/wiki/Fast_Fourier_transform

# Definiere einen Convolution struct mit filter, input und output 
struct Convolution <: AbstractLayer
    filter
end

#Berechnung des Outputs (ggf. mit striding) durch schrittweise Summierung der Ergebnisses der Multplikation der Inputs und dem Filter
function calc(conv::Convolution)(input, filter, stride)
    size_in = size(input)
    size_fil = size(filter)

    output = zeros((ceil(Int,(size_in[1] - size_fil[1] + 1) / stride), ceil(Int,(size_in[2] - size_fil[2] + 1) / stride), size_in[3]))

    for k in 1:size_in[3]
        for i in 1:(ceil(Int,(size_in[1] - size_fil[1] + 1) / 2))
            for j in 1:(ceil(Int,(size_in[2] - size_fil[2] + 1) / stride))
                output[i, j, k] = sum(input[((i-1)*stride+1):(((i-1)*stride+1)+size_fil[1]-1), ((j-1)*stride+1):(((j-1)*stride+1)+size_fil[2]-1), k] .* filter)
            end
        end
    end
    
    return output
end

#Berechnung der Convolution 
function calc_conv(conv::Convolution)(input, padd::Int, factor::Int, stride::Int)
    size_in = size(input)
    size_fil = size(conv.filter)
    
    if size_in[1] + padd < size_fil[1] || size_in[2] + padd < size_fil[2]
        error("Filter and Input have unworkable dimensions")
    end

    if stride <= 0
        error("Stride needs to be at least 1")
    end

    if p > 0

        e_l = ones(padd, size_in[2], size_in[3]) * factor
        input = [e_l; input; e_l]
        size_in = size(input)

        e_h = ones((size_in[1], padd, size_in[3])) * factor
        input = [e_h input e_h]
        
    
        output = conv.calc(input,conv.filter,stride)
            
    else
        
        output = conv.calc(input,conv.filter,stride)

    end

    return output
end
