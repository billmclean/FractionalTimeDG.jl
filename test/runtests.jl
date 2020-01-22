using FractionalTimeDG
using Test

@testset "FractionalTimeDG.jl" begin
    include("test_coef_G_K.jl")
    include("test_legendre.jl")
end
