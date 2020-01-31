module FractionalTimeDG

using ArgCheck
import GaussQuadrature
import SpecialFunctions

export coef_G, coef_K, coef_H0, coef_H1
export coef_H1_uniform, coef_H_uniform, coef_H_uniform!

Γ(x) = SpecialFunctions.gamma(x)

function gauss_legendre_rules(::Type{T}, M::Integer) where T <: AbstractFloat
    x = Vector{Vector{T}}(undef, M)
    w = Vector{Vector{T}}(undef, M)
    for m = 1:M
        x[m], w[m] = GaussQuadrature.legendre(T, m)
    end
    return x, w
end

function gauss_jacobi_rules(M::Integer, α::T, β::T) where T <: AbstractFloat
    x = Vector{Vector{T}}(undef, M)
    w = Vector{Vector{T}}(undef, M)
    for m = 1:M
        x[m], w[m] = GaussQuadrature.jacobi(m, α, β)
    end
    return x, w
end

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
    P(n, τ)

Returns the value of the Legendre polynomial of degree `n` at `τ`.
(Recursive function used in `../test`).
"""
function P(n::Integer, τ::T) where T <: AbstractFloat
    if n == 0
        return one(T)
    elseif n == 1
        return τ
    else
        return ( (2n-1) * τ * P(n-1, τ) - (n-1) * P(n-2, τ) ) / n
    end
end

"""
    legendre_polys!(P, τ)

If `τ` is a scalar, then `P` is a vector with `P[n+1]` equal to the value 
of the Legendre polynomial of degree `n` at `τ`.

If `τ` is a vector, then `P` is a matrix with `P[n+1,j]` equal to the value
of the Legendre polynomial of degree `n` at `τ[j]`.
"""
function legendre_polys!(P::AbstractVector{T}, τ::T) where T <: AbstractFloat
    I = length(P)
    if I ≥ 1
        P[1] = one(T)
    end
    if I ≥ 2
        P[2] = τ
    end
    for j = 1:I, i = 1:I-2
        P[i+2] = ((2i+1) * τ * P[i+1] - i * P[i] ) / (i+1)
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

"""
    dP(n, τ)

Returns the value of the derivative of the Legendre polynomial of degree `n` 
at `τ`.  (Recursive function used in `../test`).
"""
function dP(n::Integer, τ::T) where T <: AbstractFloat
    if n == 0
        return zero(T)
    elseif n == 1
        return one(T)
    else
        return ( (2n-1) * τ * dP(n-1, τ) - n * dP(n-2, τ) ) / ( n - 1 )
    end
end

function deriv_legendre_polys!(dP::AbstractVector{T}, τ::T
                              ) where T <: AbstractFloat
    I = length(dP)
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
    H0 = Array{T}(undef, r, r)
    coef_H0!(H0, α)
    return H0
end

function coef_H0!(H0::Matrix{T}, α::T) where T <: AbstractFloat
    r = size(H0, 1)
    @argcheck size(H0, 2) == r
    y, wy = gauss_jacobi_rules(2r, one(T), α-1)
    z, wz = gauss_legendre_rules(T, r-1)
    Ψ = Array{T}(undef, r)
    dΨ = Array{T}(undef, r)
    Φ = Array{T}(undef, 2r)
    for j = 1:r, i = 1:r
        H0[i,j] = zero(T)
        My = ceil(Integer, (i+j-1)/2)
        Mz = ceil(Integer, (i+j)/2) - 1
        for my = 1:My
            Φ[my] = zero(T)
            y_ = y[My][my]
            for mz = 1:Mz
                z_ = z[Mz][mz]
                τ = ( y_ - z_ * y_ + 1 + z_ ) / 2
                σ = τ - 1 - y_
                legendre_polys!(view(Ψ, 1:j), σ)
                deriv_legendre_polys!(view(dΨ, 1:i), τ)
                Φ[my] += wz[Mz][mz] * Ψ[j] * dΨ[i]
            end
            Φ[my] /= 2
        end
        for my = 1:My
            H0[i,j] -= wy[My][my] * Φ[my]
        end
    end
    M = ceil(Integer, r/2)
    σ, w = gauss_jacobi_rules(M, α-1, zero(T))
    Ψ = Array{T}(undef, r)
    c = 1 / ( 2^α * Γ(α) )
    for j = 1:r
        M = ceil(Integer, j/2)
        for m = 1:M
            σ_ = σ[M][m]
            legendre_polys!(view(Ψ, 1:j), σ_)
            H0[1,j] += w[M][m] * Ψ[j]
        end
        for i = 2:r
            H0[i,j] += H0[1,j]
        end
        for i = 1:r
            H0[i,j] *= c
        end
    end
end

function coef_H1!(H1::AbstractMatrix{T}, α::T,
                  kn:T, knm1::T, M:integer) where T <: AbstractFloat
    r = size(H1, 1)
    @argcheck size(H1, 2) == r
    σ1, w1 = GaussQuadrature.legendre(T, M)
    Ψ = Array{T}(undef, r, M)
    legendre_polys!(Ψ, σ1)
    two_D_n_nm1 = kn + knm1 
    D_n_nm1 = two_D_n_nm1
    for j = 1:r
        Aj = zero(T)
        for m = 1:M
            Δ = ( kn - σ1[m]*knm1 ) / two_D_n_nm1
            Aj += w1[m] * ( 1 + Δ )^(α-1) * Ψ[j,m]
        end
        for i = 1:r
            H1[i,j] = Aj
        end
    end
    σ2, w2 = GaussQuadrature.jacobi(M, α-1, zero(T))
    legendre_polys!(Ψ, σ2)
    for j = 1:r
        Bj = zero(T)
        for m = 1:M
            Bj += w2[m] * Ψ[j,m]
        end
        Bj *= ( two_D_n_nm1 / knm1 )^(1-α)
        pow = one(T)
        for i = 1:r
            pow = -pow
            H1[i,j] += pow * Bj
        end
    end
    σ3, w3 = GaussQuadrature.jacobi(M, zero(T), α)
    z, wz = GaussQuadrature.legendre(T, M)
    z .= ( z .+ 1 ) / 2
    wz .= wz/2
    dΨ = Array{T}(undef, r, M)
    deriv_legendre_polys!(dΨ, σ3)
    for j = 1:r
        for i = 1:r
            Cij_1 = zero(T)
            for m3 = 1:M
                σ_ = σ3[m3]
                inner = zero(T)
                for mz = 1:M
                    z_ = z[mz]
                    legendre_polys!(view(Ψ, 1:j, 1), 1 - z_*(1+σ_))
                    inner += wz[mz] * ((kn+z_*knm1)/two_D_n_nm1)^(α-1) * Ψ[j,1]
                end
                Cij_1 += w3[m3] * dΨ[i,m3] * inner
            end
            H1[i,j] -= Cij_1
        end
    end
    σ4, w4 = GaussQuadrature.jacobi(M, α, zero(T))
    legendre_polys!(Ψ, σ4)
    c = D_n_nm1^(α-1) * knm1 / ( 2Γ(α) )
    for j = 1:r
        for i = 1:r
            Cij_2 = zero(T)
            for m4 = 1:M
                σ_ = σ4[m4]
                inner = zero(T)
                for mz = 1:M
                    z_ = z[mz]
                    deriv_legendre_polys!(view(dΨ, 1:i, 1), z_*(1-σ_)-1)
                    inner += wz[mz] * ((z_*nk+knm1)/two_D_n_nm1)^(α-1) * dΨ[i,1]
                end
                Cij_2 += w4[m4] * Ψ[j,m4] * inner
            end
            H1[i,j] -= Cij_2
            H1[i,j] *= c
        end
    end
end

function coef_H1_uniform(r::Integer, α::T, M::Integer) where T <: AbstractFloat
    H1 = Array{T}(undef, r, r)
    coef_H1_uniform!(H1, α, M)
    return H1
end

function coef_H1_uniform!(H1::AbstractMatrix{T}, α::T, 
                  M::Integer) where T <: AbstractFloat
    r = size(H1, 1)
    @argcheck size(H1, 2) == r
    σ1, w1 = GaussQuadrature.legendre(T, M)
    Ψ = Array{T}(undef, r, M)
    legendre_polys!(Ψ, σ1)                 
    for j = 1:r
        Aj = zero(T)
        for m = 1:M
            Δ = ( 1 - σ1[m] ) / 2
            Aj += w1[m] * ( 1 + Δ )^(α-1) * Ψ[j,m]
        end
        for i = 1:r
            H1[i,j] = Aj
        end
    end
    σ2, w2 = GaussQuadrature.jacobi(M, α-1, zero(T))
    legendre_polys!(Ψ, σ2)
    for j = 1:r
        Bj = zero(T)
        for m = 1:M
            Bj += w2[m] * Ψ[j,m]
        end
        Bj *= 2^(1-α)
        pow = one(T)
        for i = 1:r
            pow = -pow
            H1[i,j] += pow * Bj
        end
    end
    σ3, w3 = GaussQuadrature.jacobi(M, zero(T), α)
    z, wz = GaussQuadrature.legendre(T, M)
    z .= ( z .+ 1 ) / 2
    wz .= wz/2
    dΨ = Array{T}(undef, r, M)
    deriv_legendre_polys!(dΨ, σ3)
    for j = 1:r
        for i = 1:r
            Cij_1 = zero(T)
            for m3 = 1:M
                σ_ = σ3[m3]
                inner = zero(T)
                for mz = 1:M
                    z_ = z[mz]
                    legendre_polys!(view(Ψ, 1:j, 1), 1 - z_*(1+σ_))
                    inner += wz[mz] * (1+z_)^(α-1) * Ψ[j,1]
                end
                Cij_1 += w3[m3] * dΨ[i,m3] * inner
            end
            Cij_1 *= 2^(1-α)
            H1[i,j] -= Cij_1
        end
    end
    σ4, w4 = GaussQuadrature.jacobi(M, α, zero(T))
    legendre_polys!(Ψ, σ4)
    for j = 1:r
        for i = 1:r
            Cij_2 = zero(T)
            for m4 = 1:M
                σ_ = σ4[m4]
                inner = zero(T)
                for mz = 1:M
                    z_ = z[mz]
                    deriv_legendre_polys!(view(dΨ, 1:i, 1), z_*(1-σ_)-1)
                    inner += wz[mz] * (z_+1)^(α-1) * dΨ[i,1]
                end
                Cij_2 += w4[m4] * Ψ[j,m4] * inner
            end
            Cij_2 *= 2^(1-α)
            H1[i,j] -= Cij_2
            H1[i,j] /= 2Γ(α)
        end
    end
end

function coef_H_uniform(N::Integer, r::Integer, α::T, 
                        M::Integer) where T <: AbstractFloat
    H = Array{T}(undef, r, r, N-1)
    coef_H1_uniform!(view(H, :, :, 1), α, M)
    σ, w = GaussQuadrature.legendre(T, M)
    Ψ = Array{T}(undef, r, M)
    legendre_polys!(Ψ, σ)
    τ = σ
    dΨ = Array{T}(undef, r, M)
    deriv_legendre_polys!(dΨ, τ)
    for ℓ = 2:N-1
        c = ℓ^(α-1) / ( 2Γ(α) )
        for j = 1:r
            Aj = zero(T)
            Bj = zero(T)
            for m = 1:M
                Δ = ( 1 - σ[m] ) / ( 2ℓ )
                Aj += w[m] * (1+Δ)^(α-1) * Ψ[j,m]
                Δ = ( 1 + σ[m] ) / ( 2ℓ )
                Bj += w[m] * (1-Δ)^(α-1) * Ψ[j,m]
            end
            pow = one(T)
            for i = 1:r
                Cij = zero(T)
                for mτ = 1:M
                    inner = zero(T)
                    for mσ = 1:M
                        Δ = ( τ[mτ] - σ[mσ] ) / ( 2ℓ )
                        inner += w[mσ] * (1+Δ)^(α-1) * Ψ[j,mσ]
                    end
                    Cij += w[mτ] * dΨ[i,mτ] * inner
                end
                pow = -pow
                H[i,j,ℓ] = c * ( Aj + pow * Bj - Cij )
            end
        end
    end
    return H
end

function coef_H_uniform!(H::Array{T,3}, ℓ_range::UnitRange, 
                         r::Integer, α::T, M::Integer) where T <: AbstractFloat
    @argcheck ℓ_range.start ≥ 2
    @argcheck ℓ_range.stop ≤ size(H, 3)
    r = size(H, 1)
    @argcheck size(H, 2) == r
    σ, w = GaussQuadrature.legendre(T, M)
    Ψ = Array{T}(undef, r, M)
    legendre_polys!(Ψ, σ)
    coef_H_uniform!(H, ℓ_range, α, w, σ, Ψ)
end

function coef_H_uniform!(H::Array{T,3}, ℓ_range::UnitRange, 
                         α::T, w::Vector{T}, σ::Vector{T},
                         Ψ::Matrix{T}) where T <: AbstractFloat
    r = size(H, 1)
    τ = σ
    M = length(w)
    for ℓ in ℓ_range
        c = - ((1-α)/(4Γ(α))) * ℓ^(α-2) 
        for j = 1:r
            for i = 1:r
                outer = zero(T)
                for mτ = 1:M
                    inner = zero(T)
                    for mσ = 1:M
                        Δ = ( τ[mτ] - σ[mσ] ) / ( 2ℓ )
                        inner += w[mσ] * (1+Δ)^(α-2) * Ψ[j,mσ]
                    end
                    outer += w[mτ] * Ψ[i,mτ] * inner
                end
                H[i,j,ℓ] = c * outer
            end
        end
    end
end

end # module FractionalTimeDG
