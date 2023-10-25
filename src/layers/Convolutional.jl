# TODO: write convolutional layer
# https://en.wikipedia.org/wiki/Convolution#Discrete_convolution
# https://en.wikipedia.org/wiki/Discrete_Fourier_transform#Circular_convolution_theorem_and_cross-correlation_theorem
# https://en.wikipedia.org/wiki/Fast_Fourier_transform

# Definiere einen Convolution struct mit filter, input und output 
struct Convolution <: AbstractLayer
    filter
end

#Primitive Berechnung des Outputs durch schrittweise Summierung der Ergebnisses der Multplikation der Inputs und dem Filter
function calc(conv::Convolution)(input, output)
    size_in = size(input)
    size_fil = size(filter)

    for k in 1:size_in[3]
        for i in 1:(size_in[1]-size_fil[1]+1)
            for j in 1:(size_in[2]-size_fil[2]+1)
                output[i, j, k] = sum(input[i:i+size_fil[1]-1, j:j+size_fil[2]-1, k] .* conv.filter)
            end
        end
    end
    
    return output
end

#Berechnung mit optinalem Padding und factor zur Befüllung
function calc_output_pad(conv::Convolution)(padd::Int, factor::Int, input, output)
    size_in = size(x)
    size_fil = size(y)
    
    if size_in[1] + p < size_fil[1] || size_in[2] + p < size_fil[2]
        error("Filter and Input have unworkable dimensions")
    end

    if p > 0

        e_l = ones(p, size_in[2], size_in[3]) * w
        input = [e_l; input; e_l]
        size_in = size(input)

        e_h = ones((size_in[1], p, size_in[3])) * w
        input = [e_h_l input e_h_r]
        
    
        output = conv.calc_output_prim(input,output)
            
    else
        
        output = conv.calc_output_prim(input,output)

    end

    return output
end


# TODO: (striding)