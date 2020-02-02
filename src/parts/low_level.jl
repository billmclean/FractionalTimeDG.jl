
# A<ℓ> for ℓ ≥ 1.
function A_integral_uniform(r::Integer, ℓ::Integer, α::T, 
                            M::Integer) where T <: AbstractFloat
    A = Vector{T}(undef, r)
    σ, w = GaussQuadrature.legendre(T, M)
    Ψ = Array{T}(undef, r, M)
    legendre_polys!(Ψ, σ)
    A_integral_uniform!(A, ℓ, α, Ψ, σ, w)
    return A
end

function A_integral_uniform!(A::Vector{T}, ℓ::Integer, α::T, Ψ::Matrix{T},
                             σ::Vector{T}, 
                             w::Vector{T}) where T <: AbstractFloat
    r = length(A)
    M = length(σ)
    for j = 1:r
        A[j] = zero(T)
        for m = 1:M
            Δ = ( 1 - σ[m] ) / 2
            A[j] += w[m] * ( 1 + Δ )^(α-1) * Ψ[j,m]
        end
    end
end

# B<1> 
function B1_integral_uniform(r::Integer, α::T) where T <: AbstractFloat
    B1 = Vector{T}(undef, r)
    M = ceil(Integer, r/2)
    σ, w = GaussQuadrature.jacobi(M, α-1, zero(T))
    Ψ = Array{T}(undef, r, M)
    legendre_polys!(Ψ, σ)
    for j = 1:r
        s = zero(T)
        for m = 1:M
            s+= w[m] * Ψ[j,m]
        end
        B1[j] = 2^(1-α) * s
    end
    return B1
end

# B<ℓ> when ℓ ≥ 2
function B_integral_uniform!(B::Matrix{T}, ℓ::Integer, α::T, Ψ::Matrix{T},
                             σ::Vector{T}, 
                             w::Vector{T}) where T <: AbstractFloat
    r = length(A)
    M = length(σ)
    for j = 1:r
        B[j] = zero(T)
        for m = 1:M
            Δ = ( 1 + σ[m] ) / ( 2ℓ )
            B[j] += w[m] * (1-Δ)^(α-1) * Ψ[j,m]
        end
    end
end

# C<1>
function C1_integral_uniform(r::Integer, α::T, 
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
function C_integral_uniform(r::Integer, ℓ::Integer, 
                            α::T, M::Integer) where T <: AbstractFloat
    C = Array{T}(undef, r, r)
    σ, wσ = GaussQuadrature.legendre(T, M)
    τ, wτ = σ, wσ
    Ψ = Array{T}(undef, r, M)
    legendre_polys!(Ψ, σ)
    dΨ = Array{T}(undef, r, M)
    deriv_legendre_polys!(dΨ, τ)
    for j = 1:r
        for i = 1:r
            outer = zero(T)
            for mτ = 1:M
                inner = zero(T)
                for mσ = 1:M
                    Δ = ( τ[mτ] - σ[mσ] ) / ( 2ℓ )
                    inner += w[mσ] * (1+Δ)^(α-1) * Ψ[j,mσ]
                end
                outer += w[mτ] * dΨ[i,mτ] * inner
            end
            C[i,j] = outer
        end
    end
end
