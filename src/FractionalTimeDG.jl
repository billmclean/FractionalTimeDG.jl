module FractionalTimeDG

using ArgCheck
using OffsetArrays
import GaussQuadrature
import SpecialFunctions

export coef_G, coef_K, coef_H0, coef_H1
export coef_H1_uniform, coef_H_uniform, coef_H_uniform!

Γ(x) = SpecialFunctions.gamma(x)

include("parts/legendre_utils.jl")
include("parts/low_level.jl")
include("parts/coefficients.jl")

end # module FractionalTimeDG
