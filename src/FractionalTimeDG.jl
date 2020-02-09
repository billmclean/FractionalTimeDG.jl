module FractionalTimeDG

using ArgCheck
using OffsetArrays
import GaussQuadrature
import SpecialFunctions

export setup, coef_G, coef_K, coef_H0, coef_H1
export coef_H1_uniform, coef_H_uniform, coef_H_uniform!

Γ(x) = SpecialFunctions.gamma(x)

struct Store{T<: AbstractFloat}
    # shared data for low-level routines
    # initialise with store = setup(α, rmax, Mmax)
    α::T          # fraction diffusion exponent
    rmax::Integer # (max degree of piecewise polynomials) + 1
    Mmax::Integer # maximum number of Gauss points
    unitlegendre::Vector{Matrix{T}} # w(x) = 1 on [0,1]
    legendre::Vector{Matrix{T}}     # w(x) = 1 on [-1,1]
    jacobi1::Vector{Matrix{T}}      # w(x) = (1-x)^(α-1)
    jacobi2::Vector{Matrix{T}}      # w(x) = (1+x)^(α-1)
    jacobi3::Vector{Matrix{T}}      # w(x) = (1-x)^α
    jacobi4::Vector{Matrix{T}}      # w(x) = (1+x)^α
    jacobi5::Vector{Matrix{T}}      # w(x) = (1-x)(1+x)^(α-1)
    Ψ::Matrix{T}
    dΨ::Matrix{T}
    A::Vector{T}
    B::Vector{T}
    C::Matrix{T}
end

include("parts/store.jl")
include("parts/legendre_utils.jl")
#include("parts/low_level.jl")
#include("parts/low_level_uniform.jl")
#include("parts/coefficients.jl")
#include("parts/coefficients_uniform.jl")
#include("parts/C1_integrals.jl")
include("parts/coef_uniform.jl")
include("parts/coef.jl")

end # module FractionalTimeDG
