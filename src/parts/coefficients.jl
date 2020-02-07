
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

function coef_H(r::Vector{Integer}, α::T, t::OffsetVector{T}, 
                M::Integer) where T <: AbstractFloat
    n = length(r)
    H = Vector{Matrix{T}}
    for ℓ = 1:n
        H[ℓ] = Array{T}(undef, r[n], r[ℓ])
    end
    coef_H!(H, α, t, M)
end

function coef_H!(H::Vector{Matrix{T}}, α::T, t::OffsetVector{T},
                 M::Integer) where T <: AbstractFloat
    n = length(H)
    rn = size(H[n], 1)
    coef_H0(H[n], α)
    kn = t[n] - t[n-1]
    for j = 1:rn, i = 1:rn
        H[n][i,j] *= kn^α 
    end
    rnm1 = size(H[n-1], 2)
    coef_H1!(H[n-1], n, α, t, M)
    rmax = rn
    for ℓ = 1:n-2
        rℓ = size(H[ℓ], 2)
        rmax = max(rmax, rℓ)
    end
    σ, wσ = GaussQuadrature.legendre(T, M)
    Ψ = Array{T}(rmax, M)
    legendre_polys!(Ψ, σ)
    for ℓ = n-2:-1:1
        H_history!(H[ℓ], n, ℓ, t, Ψ, σ, wσ)
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

function coef_H1(rn::Integer, rnm1::Integer, n::Integer, α::T, 
                 t::OffsetVector{T}, M::Integer) where T <: AbstractFloat
    H1 = Array{T}(undef, rn, rnm1)
    coef_H1!(H1, α, t, M)
    return H1
end

function coef_H1!(H1::AbstractMatrix{T}, n::Integer, α::T, 
                  t::OffsetVector{T}, M::Integer) where T <: AbstractFloat
    rn = size(H1, 1)
    rnm1 = size(H1, 2)
    A = A_integral(rnm1, n, n-1, α, t, M)
    B1 = B1_integral(rnm1, n, α, M)
    C1_first, C1_second = C1_integrals(rn, rnm1, n, α, t, M)
    D = (t[n]+t[n-1])/2 - (t[n-1]+t[n-2])/2
    knm1 = t[n-1]-t[n-2]
    c = D^(α-1) * knm1 / ( 2Γ(α) )
    for j = 1:rnm1
        pow = one(T)
        for i = 1:rn
            pow = -pow
            Cij = C1_first[i,j] + C1_second[i,j]
            H1[i,j] = c * ( A[j] + pow * B[j] - Cij )
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
    A = A_integral_uniform(r, 1, α, M)
    B = B1_integral_uniform(r, α)
    C1_first, C1_second = C1_integrals_uniform(r, α, M)
    c = 1 / ( 2Γ(α) )
    for j = 1:r
        pow = one(T)
        for i = 1:r
            pow = - pow
            Cij = C1_first[i,j] + C1_second[i,j]
            H1[i,j] = c * ( A[j] + pow * B[j] - Cij )
        end
    end
end

function coef_H_uniform(r::Integer, α::T, t::OffsetVector{T},
                        M::Integer, version=1) where T <: AbstractFloat
    N = axes(t, 1).indices.stop
    H = OffsetVector{Matrix{T}}(undef, 0:N-1)
    for ℓ = 0:N-1
        H[ℓ] = Array{T}(undef, r, r)
    end
    coef_H0(H[0], α)
    coef_H1_uniform(H[1], α, M)
    σ, w = GaussQuadrature.legendre(T, M)
    Ψ = Array{T}(undef, r, M)
    legendre_polys!(Ψ, M)
    if version == 1
        dΨ = Array{T}(undef, r, M)
        deriv_legendre_polys!(dΨ, σ)
        coef_H_uniform_ver1!(H[ℓ], ℓ, α, σ, w, Ψ, dΨ)
    elseif verson == 2
        for ℓ = 2:N-1
            coef_H_uniform_ver2!(H[ℓ], ℓ, α, w, Ψ)
        end
    else
        throw(ArgumentError("version must be 1 or 2"))
    end
    return H
end

# H<ℓ> for ℓ ≥ 2, first version
function coef_H_uniform_ver1!(Hℓ::Matrix{T}, ℓ::Integer, α::T, 
                           σ::Vector{T}, w::Vector{T}, Ψ::Matrix{T}, 
                           dΨ::Matrix{T}) where T <: AbstractFloat
    r = size(Hℓ, 1)
    M = length(σ)
    c = ℓ^(α-1) / ( 2Γ(α) )
    τ = σ
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
            Hℓ[i,j] = c * ( Aj + pow * Bj - Cij )
        end
    end
end

# H<ℓ> for ℓ ≥ 2, second version
function coef_H_uniform_ver2!(Hℓ::Matrix{T}, ℓ::Integer, 
                             α::T, σ::Vector{T}, w::Vector{T},
                             Ψ::Matrix{T}) where T <: AbstractFloat
    r = size(Hℓ, 2)
    A = Vector{T}(undef, r)
    A_integral_uniform!(A, ℓ, α, Ψ, σ, w)
    B = Vector{T}(undef, r)
    B_integral_uniform!(B, ℓ, α, Ψ, σ, w)
    r = size(Hℓ, 1)
    τ = σ
    M = length(w)
    c = - ((1-α)/(4Γ(α))) * ℓ^(α-2) 
    for j = 1:r, i = 1:r
        outer = zero(T)
        for mτ = 1:M
            inner = zero(T)
            for mσ = 1:M
                Δ = ( τ[mτ] - σ[mσ] ) / ( 2ℓ )
                inner += w[mσ] * (1+Δ)^(α-2) * Ψ[j,mσ]
            end
            outer += w[mτ] * Ψ[i,mτ] * inner
        end
        Hℓ[i,j] = c * outer
    end
end

function coef_H_uniform_2!(Hℓ::AbstractMatrix{T}, ℓ::Integer, α::T, 
                           σ::Vector{T}, w::Vector{T},
                           Ψ::Matrix{T}) where T <: AbstractFloat
    r = size(Hℓ, 1)
    M = length(σ)
    c = ℓ^(α-1) / ( 2Γ(α) )
    τ = σ
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
            Hℓ[i,j] = c * outer
        end
    end
end

function coef_H_uniform(N::Integer, r::Integer, α::T, M::Integer,
                        version::Integer) where T <: AbstractFloat
    H = Array{T}(undef, r, r, N-1)
    coef_H1_uniform!(view(H, :, :, 1), α, M)
    σ, w = GaussQuadrature.legendre(T, M)
    Ψ = Array{T}(undef, r, M)
    legendre_polys!(Ψ, σ)
    τ = σ
    if version == 1
        dΨ = Array{T}(undef, r, M)
        deriv_legendre_polys!(dΨ, τ)
        for ℓ = 2:N-1
            coef_H_uniform_1!(view(H, :, :, ℓ), ℓ, α, σ, w, Ψ, dΨ)
        end
    elseif version == 2
        for ℓ = 2:N-1
            coef_H_uniform_2!(view(H, :, :, ℓ), ℓ, α, σ, w, Ψ)
        end
    else
        throw(ArgumentError("version must be 1 or 2"))
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
    coef_H_uniform!(H, ℓ_range, α, σ, w, Ψ)
end

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
