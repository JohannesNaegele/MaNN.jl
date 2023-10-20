# TODO: write convolutional layer
# https://en.wikipedia.org/wiki/Convolution#Discrete_convolution
# https://en.wikipedia.org/wiki/Discrete_Fourier_transform#Circular_convolution_theorem_and_cross-correlation_theorem
# https://en.wikipedia.org/wiki/Fast_Fourier_transform

# Definiere einen Convolution struct mit filter, input und output 
struct Convolution <: AbstractLayer
    filter
    input
    output
end

#Setter func zum Übergeben der nötigen Daten
function set_filter(conv::Convolution)(matrix)
    conv.filter = matrix
end

function set_input(conv::Convolution)(matrix)
    conv.input = matrix
end

#Primitive Berechnung des Outputs durch schrittweise Summierung der Ergebnisses der Multplikation der Inputs und dem Filter
function calc_output_prim(conv::Convolution)(padd::Int)
    size_in = size(conv.input)
    size_fil = size(filter)

    
    for t in 1:size_in[4]
        for i in 1:(size_in[1]-size_fil[1]+1)
            for j in 1:(size_in[2]-size_fil[2]+1)
                for k in 1:size_in[3]
                    conv.output[i, j, k, t] = sum(conv.input[i:i+size_fil[1]-1, j:j+size_fil[2]-1, k, t] .* conv.filter)
                end
            end
        end
    end

end

#Berechnung mit optinalem Padding und factor zur Befüllung
function calc_output_pad(conv::Convolution)(padd::Int,factor::Int)
    size_in = size(conv.input)
    size_fil = size(filter)

    if padd > 0

        index = size(size_in)
    
        e_l_o = ones((floor(Int, (size_fil[1] - 1) / 2), size_in[2], size_in[3], size_in[4])) * factor
        e_l_u = ones((floor(Int, size_fil[1] / 2), size_in[2], size_in[3], size_in[4])) * factor
        x = [e_l_o; conv.input; e_l_u]
        size_in = size(x)
    
        e_h_l = ones((size_in[1], floor(Int, (size_fil[2] - 1) / 2), size_in[3], size_in[4])) * factor
        e_h_r = ones((size_in[1], floor(Int, size_fil[2] / 2), size_in[3], size_in[4])) * factor
        x = [e_h_l conv.input e_h_r]
        
    
        for t in 1:index[4]
            for i in 1:index[1]
                for j in 1:index[2]
                    for k in 1:index[3]
                        conv.output[i, j, k, t] = sum(conv.input[i:i+size_fil[1]-1, j:j+size_fil[2]-1, k, t] .* conv.filter)
                    end
                end
            end
        end
    
    else
    
        for t in 1:size_in[4]
            for i in 1:(size_in[1]-size_fil[1]+1)
                for j in 1:(size_in[2]-size_fil[2]+1)
                    for k in 1:size_in[3]
                        conv.output[i, j, k, t] = sum(conv.input[i:i+size_fil[1]-1, j:j+size_fil[2]-1, k, t] .* conv.filter)
                    end
                end
            end
        end

    end
end

#Getter Func für Output
function get_output(conv::Convolution)
    return conv.output
end

# TODO: (padding + striding)