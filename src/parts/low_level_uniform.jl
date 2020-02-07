# A<ℓ> for ℓ ≥ 1.
function A_integral_uniform!(r::Integer, ℓ::Integer, M::Integer, 
                             store::Store{T}) where T <: AbstractFloat
    σ, w = rule(store.legendre[M])
    Ψ = view(store.Ψ, 1:r, 1:M)
    legendre_polys!(Ψ, σ)
    α = store.α
    for j = 1:r
        s = zero(T)
        for m = 1:M
            Δ = ( 1 - σ[m] ) / 2
            s += w[m] * ( 1 + Δ )^(α-1) * Ψ[j,m]
        end
        store.A[j] = s
    end
end

# B<1> 
function B1_integral_uniform!(r::Integer, 
                              store::Store{T}) where T <: AbstractFloat
    M = ceil(Integer, r/2)
    σ, w = rule(store.jacobi1[M])
    Ψ = view(store.Ψ, 1:r, 1:M)
    legendre_polys!(Ψ, σ)
    α = store.α
    for j = 1:r
        s = zero(T)
        for m = 1:M
            s+= w[m] * Ψ[j,m]
        end
        store.B[j] = 2^(1-α) * s
    end
end

# B<ℓ> when ℓ ≥ 2
function B_integral_uniform!(ℓ::Integer, α::T, M::Integer,
                             store::Store{T}) where T <: AbstractFloat
    σ, w = rule(store.legendre[M])
    Ψ = view(store.Ψ, 1:r, 1:M)
    for j = 1:r
        s = zero(T)
        for m = 1:M
            Δ = ( 1 + σ[m] ) / ( 2ℓ )
            s += w[m] * (1-Δ)^(α-1) * Ψ[j,m]
        end
        store.B[j] = s
    end
end

# C<1>
function C1_integral_uniform!(r::Integer, Mz::Integer, 
                             store::Store{T}) where T <: AbstractFloat
    Mτ = ceil(Integer, r-1)
    Mσ = Mτ
    z, wz = rule(store.unitlegendre[Mz])
    σ, wσ = rule(store.jacobi3[Mσ])
    τ, wτ = rule(store.jacobi4[Mτ])
    Ψ = view(store.Ψ, 1:r, 1:Mσ)
    dΨ = view(store.dΨ, 1:r, 1:Mτ)
    deriv_legendre_polys!(dΨ, τ)
    α = store.α
    for j = 1:r, i = 1:r
        outer = zero(T)
        for mτ = 1:Mτ
            τ_ = τ[mτ]
            inner = zero(T)
            for mz = 1:Mz
                z_ = z[mz]
                legendre_polys!(view(Ψ, 1:j), 1 - z_*(1+τ_))
                inner += wz[mz] * (1+z_)^(α-1) * Ψ[j]
            end
            outer += wτ[mτ] * dΨ[i,mτ] * inner
        end
        store.C[i,j] = 2^(1-α) * outer
    end
    legendre_polys!(Ψ, σ)
    for j = 1:r, i = 1:r
        outer = zero(T)
        for mσ = 1:Mσ
            σ_ = σ[mσ]
            inner = zero(T)
            for mz = 1:Mz
                z_ = z[mz]
                deriv_legendre_polys!(view(dΨ, 1:i), z_*(1-σ_)-1)
                inner += wz[mz] * (z_+1)^(α-1) * dΨ[i]
            end
            outer += wσ[mσ] * Ψ[j,mσ] * inner
        end
        store.C[i,j] += 2^(1-α) * outer
    end
end

function C1_integrals_uniform(r::Integer, α::T, 
                              Mz::Integer) where T <: AbstractFloat
    C1_first = Array{T}(undef, r, r)
    C1_second = Array{T}(undef, r, r)
    Mτ = ceil(Integer, r-1)
    Mσ = Mτ
    z, wz = GaussQuadrature.legendre(T, Mz)
    z .= ( z .+ 1 ) / 2
    wz .= wz / 2
    σ, wσ = GaussQuadrature.jacobi(Mσ, α, zero(T))
    τ, wτ = GaussQuadrature.jacobi(Mσ, zero(T), α)
    Ψ = Array{T}(undef, r, Mσ) 
    dΨ = Array{T}(undef, r, Mτ) 
    deriv_legendre_polys!(dΨ, τ)
    for j = 1:r
        for i = 1:r
            outer = zero(T)
            for mτ = 1:Mτ
                τ_ = τ[mτ]
                inner = zero(T)
                for mz = 1:Mz
                    z_ = z[mz]
                    legendre_polys!(view(Ψ, 1:j, 1), 1 - z_*(1+τ_))
                    inner += wz[mz] * (1+z_)^(α-1) * Ψ[j,1]
                end
                outer += wτ[mτ] * dΨ[i,mτ] * inner
            end
            C1_first[i,j] = 2^(1-α) * outer
        end
    end
    legendre_polys!(Ψ, σ)
    for j = 1:r
        for i = 1:r
            outer = zero(T)
            for mσ = 1:Mσ
                σ_ = σ[mσ]
                inner = zero(T)
                for mz = 1:Mz
                    z_ = z[mz]
                    deriv_legendre_polys!(view(dΨ, 1:i, 1), z_*(1-σ_)-1)
                    inner += wz[mz] * (z_+1)^(α-1) * dΨ[i,1]
                end
                outer += wσ[mσ] * Ψ[j,mσ] * inner
            end
            C1_second[i,j] = 2^(1-α) * outer
        end
    end
    return C1_first, C1_second
end

# C<ℓ> for ℓ ≥ 2.
function C_integral_uniform!(r::Integer, ℓ::Integer, M::Integer,
                            store::Store{T}) where T <: AbstractFloat
    σ, wσ = rule(store.legendre[M])
    τ, wτ = σ, wσ
    Ψ  = view(store.Ψ, 1:r, 1:M)
    dΨ = view(store.dΨ, 1:r, 1:M)
    legendre_polys!(Ψ, σ)
    deriv_legendre_polys!(dΨ, τ)
    α = store.α
    for j = 1:r, i = 1:r
        outer = zero(T)
        for mτ = 1:M
            inner = zero(T)
            for mσ = 1:M
                Δ = ( τ[mτ] - σ[mσ] ) / ( 2ℓ )
                inner += wσ[mσ] * (1+Δ)^(α-1) * Ψ[j,mσ]
            end
            outer += wτ[mτ] * dΨ[i,mτ] * inner
        end
        store.C[i,j] = outer
    end
end
