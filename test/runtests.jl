using FractionalTimeDG
using OffsetArrays
using Test

include("common_defs.jl")

@testset "FractionalTimeDG.jl" begin
    include("test_coef_G_K.jl")
    include("test_legendre.jl")
    include("test_ABC.jl")
    include("test_H_uniform.jl")
end
