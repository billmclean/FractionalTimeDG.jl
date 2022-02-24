function coef_H_uniform!(r::Integer, N::Integer, M::Integer,
                         store::Store{T}, version=2) where T <: AbstractFloat
    H = OffsetVector{Matrix{T}}(undef, 0:N-1)
    for ℓbar = 0:N-1
        H[ℓbar] = Array{T}(undef, r, r)
    end
    coef_H_uniform!(H, M, store, version)
    return H
end

function coef_H_uniform!(H::OffsetVector{Matrix{T}}, M::Integer,
                         store::Store{T}, version=2) where T <: AbstractFloat
    N = length(H) 
    coef_H0_uniform!(H[0], store)
    if N ≥ 2
        coef_H1_uniform!(H[1], M, store)
    end
    if N ≥ 3
        if version == 1
            coef_H_uniform_ver1!(H, 2:N-1, M, store)
        elseif version == 2
            coef_H_uniform_ver2!(H, 2:N-1, M, store)
        else
            throw(ArgumentError("version must be 1 or 2"))
        end
    end
end

# H^0
function coef_H0_uniform!(r::Integer, store::Store{T}) where T <: AbstractFloat
    H0 =Array{T}(undef, r, r)
    coef_H0_uniform!(H0, store)
    return H0
end

function coef_H0_uniform!(H0::Matrix{T},
                          store::Store{T}) where T <: AbstractFloat
    r = size(H0, 1)
    α = store.α
    @argcheck r ≤ store.rmax
    @argcheck size(H0, 2) == r
    y, wy = rules(store.jacobi5)
    z, wz = rules(store.legendre)
    Φ  = view(store.C, 1:2r)
    for j = 1:r, i = 1:r
        Ψ  = view(store.Ψ, 1:j)
        dΨ = view(store.dΨ, 1:i)
        My = ceil(Integer, (i+j-1)/2)
        Mz = ceil(Integer, (i+j)/2)
        for my = 1:My
            Φ[my] = zero(T)
            y_ = y[My][my]
            for mz = 1:Mz
                z_ = z[Mz][mz]
                τ = ( y_ - z_ * y_ + 1 + z_ ) / 2
                σ = τ - 1 - y_
                legendre_polys!(Ψ, σ)
                deriv_legendre_polys!(dΨ, τ)
                Φ[my] += wz[Mz][mz] * Ψ[j] * dΨ[i]
            end
            Φ[my] /= 2
        end
        s = zero(T)
        for my = 1:My
            s += wy[My][my] * Φ[my]
        end
        H0[i,j] = s
    end
    σ, w = rules(store.jacobi1)
    c = 1 / ( 2^α * Γ(α) )
    for j = 1:r
        Ψ = view(store.Ψ, 1:j)
        M = ceil(Integer, j/2)
        s = zero(T)
        for m = 1:M
            σ_ = σ[M][m]
            legendre_polys!(Ψ, σ_)
            s += w[M][m] * Ψ[j]
        end
        for i = 1:r
            H0[i,j] = c * ( s - H0[i,j] )
        end
    end
end

# H^1
function coef_H1_uniform!(r::Integer, M::Integer,
                          store::Store{T}) where T <: AbstractFloat
    H1 = Array{T}(undef, r, r)
    coef_H1_uniform!(H1, M, store)
    return H1
end

function coef_H1_uniform!(H1::Matrix{T}, M::Integer,
                          store::Store{T}) where T <: AbstractFloat
    r = size(H1, 1)
    α = store.α
    @argcheck size(H1, 2) == r
    @argcheck r ≤ store.rmax

    σ, wσ = rule(store.legendre[M])
    Ψ = view(store.Ψ, 1:r, 1:M)
    legendre_polys!(Ψ, σ)
    A = view(store.A, 1:r)
    c = 2^(1-α)
    for j = 1:r
        s = zero(T)
        for m = 1:M
            s += wσ[m] * ( 3 - σ[m] )^(α-1) * Ψ[j,m]
        end
        A[j] = c * s
    end

    Mσ = ceil(Integer, r/2)
    σ, wσ = rule(store.jacobi1[Mσ])
    Ψ = view(store.Ψ, 1:r, 1:Mσ)
    legendre_polys!(Ψ, σ)
    B = view(store.B, 1:r)
    for j = 1:r
        s = zero(T)
        for m = 1:Mσ
            s+= wσ[m] * Ψ[j,m]
        end
        B[j] = 2^(1-α) * s
    end

    Mτ = max(1, ceil(Integer, r-1))
    Mσ = Mτ
    Mz = M
    z, wz = rule(store.unitlegendre[Mz])
    σ, wσ = rule(store.jacobi3[Mσ])
    τ, wτ = rule(store.jacobi4[Mτ])
    dΨ = view(store.dΨ, 1:r, 1:Mτ)
    deriv_legendre_polys!(dΨ, τ)
    C = view(store.C, 1:r, 1:r)
    for j = 1:r
        Ψ = view(store.Ψ, 1:j)
        for i = 1:r
            outer = zero(T)
            for mτ = 1:Mτ
                τ_ = τ[mτ]
                inner = zero(T)
                for mz = 1:Mz
                    z_ = z[mz]
                    legendre_polys!(Ψ, 1 - z_*(1+τ_))
                    inner += wz[mz] * (1+z_)^(α-1) * Ψ[j]
                end
                outer += wτ[mτ] * dΨ[i,mτ] * inner
            end
            C[i,j] = 2^(1-α) * outer
        end
    end
    Ψ = view(store.Ψ, 1:r, 1:Mσ)
    legendre_polys!(Ψ, σ)
    for j = 1:r
        for i = 1:r
            dΨ = view(store.dΨ, 1:i)
            outer = zero(T)
            for mσ = 1:Mσ
                σ_ = σ[mσ]
                inner = zero(T)
                for mz = 1:Mz
                    z_ = z[mz]
                    deriv_legendre_polys!(dΨ, z_*(1-σ_)-1)
                    inner += wz[mz] * (z_+1)^(α-1) * dΨ[i]
                end
                outer += wσ[mσ] * Ψ[j,mσ] * inner
            end
            C[i,j] += 2^(1-α) * outer
        end
    end

    c = 1 / ( 2Γ(α) )
    for j = 1:r
        pow = one(T)
        for i = 1:r
            pow = -pow
            H1[i,j] = c * ( A[j] + pow * B[j] - C[i,j] )
        end
    end
end

# H^ℓbar for ℓbar in rng, first version (rng must not include ℓbar ≤ 1)
function coef_H_uniform_ver1!(H::OffsetVector{Matrix{T}}, rng::UnitRange,
                              M::Integer, store::Store{T}
                             ) where T <: AbstractFloat
    @argcheck rng.start ≥ 2
    r = size(H[rng.start], 1)
    A = view(store.A, 1:r)
    B = view(store.B, 1:r)
    C = view(store.C, 1:r, 1:r)

    σ, wσ = rule(store.legendre[M])
    τ, wτ = σ, wσ
    Ψ  = view(store.Ψ, 1:r, 1:M)
    dΨ = view(store.dΨ, 1:r, 1:M)
    legendre_polys!(Ψ, σ)
    deriv_legendre_polys!(dΨ, τ)
    α = store.α
    c = 1 / ( 2Γ(α) )
    for ℓbar in rng

        for j = 1:r
            s = zero(T)
            for m = 1:M
                Δ = ( 1 - σ[m] ) / (2ℓbar)
                s += wσ[m] * ( 1 + Δ )^(α-1) * Ψ[j,m]
            end
            A[j] = s
        end

        for j = 1:r
            s = zero(T)
            for m = 1:M
                 Δ = ( 1 + σ[m] ) / ( 2ℓbar )
                 s += wσ[m] * (1-Δ)^(α-1) * Ψ[j,m]
            end
            B[j] = s
        end

        τ, wτ = σ, wσ
        for j = 1:r, i = 1:r
            outer = zero(T)
            for mτ = 1:M
                inner = zero(T)
                for mσ = 1:M
                    Δ = ( τ[mτ] - σ[mσ] ) / ( 2ℓbar )
                    inner += wσ[mσ] * (1+Δ)^(α-1) * Ψ[j,mσ]
                end
                outer += wτ[mτ] * dΨ[i,mτ] * inner
            end
            C[i,j] = outer
        end

        for j = 1:r
            pow = one(T)
            for i = 1:r
                pow = -pow
                H[ℓbar][i,j] = c * ℓbar^(α-1) * ( A[j] + pow * B[j] - C[i,j] )
            end
        end
    end
end

# H^ℓbar for ℓbar in rng, second version (rng must not include ℓbar ≤ 1)
function coef_H_uniform_ver2!(H::OffsetVector{Matrix{T}}, rng::UnitRange,
                              M::Integer, store::Store{T}
                             ) where T <: AbstractFloat
    @argcheck rng.start ≥ 2
    r = size(H[rng.start], 1)
    σ, wσ = rule(store.legendre[M])
    τ, wτ = σ, wσ
    Ψ  = view(store.Ψ, 1:r, 1:M)
    legendre_polys!(Ψ, σ)
    α = store.α
    c = - (1-α) / ( 4Γ(α) )
    for ℓbar in rng
        for j = 1:r, i = 1:r
            outer = zero(T)
            for mτ = 1:M
                inner = zero(T)
                for mσ = 1:M
                    Δ = ( τ[mτ] - σ[mσ] ) / ( 2ℓbar )
                    inner += wσ[mσ] * (1+Δ)^(α-2) * Ψ[j,mσ]
                end
                outer += wτ[mτ] * Ψ[i,mτ] * inner
            end
            H[ℓbar][i,j] = c * ℓbar^(α-2) * outer
        end
    end
end

"""
    coef_H(r)

Returns diagonal matrix `H` for the classical case `α=1`.
"""
function coef_H(::Type{T}, r::Integer) where T <: AbstractFloat
    d = T[ one(T) / (2j+1) for j = 0:r-1 ]
    return LinearAlgebra.Diagonal(d)
end

