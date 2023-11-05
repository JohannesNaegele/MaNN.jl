using MaNN
using Test

@testset "MaNN.jl" begin
    # Write your tests here.


    Convolution t([4 5; 5 6])
    @test t.calc([1 1 1 1; 2 2 2 2; 3 3 3 3],[4 5; 5 6],1) == [31 31 31; 51 51 51]
    @test t.calc([1 1 1 1; 2 2 2 2; 3 3 3 3],[4 5; 5 6],2) == [31 31]
    @test t.calc_conv([1 1 1 1; 2 2 2 2; 3 3 3 3],0,0,1) == [31 31 31; 51 51 51]
    @test t.calc_conv([1 1 1 1; 2 2 2 2; 3 3 3 3],0,0,2) == [31 31]
    @test t.calc_conv([1 1 1 1; 2 2 2 2; 3 3 3 3],1,0,1) == [6 11 11 11 5; 11 31 31 31 14; 28 51 51 51 23; 15 27 27 27 12]
    @test_throws "Stride needs to be at least 1" t.calc_conv([1 1 1 1; 2 2 2 2; 3 3 3 3],0,0,0) == [31 31]
    @test_throws "Filter and Input have unworkable dimensions" @test t.calc_conv([1; 2],0,0,1) == [31 31]
end
