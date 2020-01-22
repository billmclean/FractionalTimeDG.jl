module FractionalTimeDG

using ArgCheck
using GaussQuadrature

export coef_G, coef_K
export legendre_polys!, deriv_legendre_polys!

function coef_G(::Type{T}, rn::Integer) where T <: AbstractFloat
    G = ones(T, rn, rn)
    for j = 1:rn
        for i = j+1:2:rn
            G[i,j] = -G[i,j]
        end
    end
    return G
end

coef_G(rn) = coef_G(Float64, rn)

function coef_K(::Type{T}, rn::Integer, rnm1::Integer) where T <:AbstractFloat
    K = ones(T, rn, rnm1)
    for j = 1:rnm1
        for i = 2:2:rn
            K[i,j] = -K[i,j]
        end
    end
    return K
end

coef_K(rn, rnm1) = coef_K(Float64, rn, rnm1)

"""
    legendre_polys!(P, τ)

If `τ` is a scalar, then `P` is a vector with `P[n+1]` equal to the value 
of the Legendre polynomial of degree `n` at `τ`.

If `τ` is a vector, then `P` is a matrix with `P[n+1,j]` equal to the value
of the Legendre polynomial of degree `n` at `τ[j]`.
"""
function legendre_polys!(P::Vector{T}, τ::T) where T <: AbstractFloat
    I = length(P)
    if I ≥ 1
        P[1] = one(T)
    end
    if I ≥ 2
        P[2] = τ
    end
    for j = 1:J, i = 1:I-2
        P[i+2] = ((2i+1) * τ * P[n+1] - i * P[i] ) / (i+1)
    end
end

function legendre_polys!(P::Matrix{T}, τ::AbstractVector{T}
                        ) where T <: AbstractFloat
    I = size(P, 1)
    J = size(P, 2)
    @argcheck length(τ) == J
    if I ≥ 1
        for j = 1:J
            P[1,j] = one(T)
        end
    end
    if I ≥ 2
        for j = 1:J
            P[2,j] = τ[j]
        end
    end
    for j = 1:J, i = 1:I-2
        P[i+2,j] = ((2i+1) * τ[j] * P[i+1,j] - i * P[i,j] ) / (i+1)
    end
end

function deriv_legendre_polys!(dP::Vector{T}, τ::T) where T <: AbstractFloat
    I = length(I)
    if I ≥ 1
        dP[1] = zero(T)
    end
    if I ≥ 2
        dP[2] = one(T)
    end
    for i = 1:I-2
        dP[i+2] = ( (2i+1) * τ * dP[i+1] - (i+1) * dP[i] ) / i
    end
end

function deriv_legendre_polys!(dP::Matrix{T}, τ::AbstractVector{T}
                               ) where T <: AbstractFloat
    I = size(dP, 1)
    J = size(dP, 2)
    @argcheck length(τ) == J
    if I ≥ 1
        for j = 1:J
            dP[1,j] = zero(T)
        end
    end
    if I ≥ 2
        for j = 1:J
            dP[2,j] = one(T)
        end
    end
    for j = 1:J, i = 1:I-2
        dP[i+2,j] = ((2i+1) * τ[j] * dP[i+1,j] - (i+1) * dP[i,j] ) / i
    end
end

function coef_H0(r::Integer, α::T) where T <: AbstractFloat
    H0 = zeros(T, r, r)
    z, wz = legendre(T, 2r)
    y, wy = jacobi(2r, one(T), α-1)
    Ψ = Array{T}(undef, r)
    dΨ = Array{T}(undef, r)
    Φ = Array{T}(undef, J2)
    for j = 1:r, i = 1:r
        for my = 1:length(y)
            Φ[my] = zero(T)
            for mz = 1:length(z)
                τ = ( y[my] - z[mz] * y[my] + 1 + z[mz] ) / 2
                σ = τ - 1 - y[my]
                legendre_polys!(Ψ, σ)
                deriv_legendre_polys!(dΨ, τ)
                Φ[my] += wy[my] * Ψ[j] * dΨ[i]
            end
            Φ[my] /= 2
        end
        for my = 1:length(y)
            H0[i,j] -= wy[my] * Φ[my]
        end
    end
    σ, w = jacobi(ceil(Integer, r/2), α-1, zero(T))
    Ψ = Array{T}(undef, r, length(σ))
    legendre_polys!(Ψ, σ)
    c = 1 / ( 2^α * Γ(α) )
    for j = 1:r
        for m = 1:length(σ)
            H0[1,j] += w[m] * Ψ[j,m]
        end
        for i = 2:r
            H0[i,j] += H0[1,j]
        end
        H0[i,j] *= c
    end
    return H0
end

end # module FractionalTimeDG
