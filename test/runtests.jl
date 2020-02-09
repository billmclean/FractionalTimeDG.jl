using FractionalTimeDG
using Test
using OffsetArrays
import FractionalTimeDG, SpecialFunctions
using QuadGK: quadgk

Γ = SpecialFunctions.gamma

@testset "FractionalTimeDG.jl" begin
    include("test_coef_G_K.jl")
    include("test_legendre.jl")
    include("test_ABC.jl")
    include("test_H_uniform.jl")
    include("test_H.jl")
end
