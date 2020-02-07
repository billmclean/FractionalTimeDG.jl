function setup(α::T, rmax::Integer, Mmax::Integer
              ) where T <: AbstractFloat
    unitlegendre = Vector{Matrix{T}}(undef, Mmax)
    legendre = Vector{Matrix{T}}(undef, Mmax)
    jacobi1 = Vector{Matrix{T}}(undef, Mmax)
    jacobi2 = Vector{Matrix{T}}(undef, Mmax)
    jacobi3 = Vector{Matrix{T}}(undef, Mmax)
    jacobi4 = Vector{Matrix{T}}(undef, Mmax)
    for M = 1:Mmax
        unitlegendre[M] = Array{T}(undef, M, 2)
        legendre[M] = Array{T}(undef, M, 2)
        x, w = GaussQuadrature.legendre(T, M)
        legendre[M][:,1], legendre[M][:,2] = x, w
        unitlegendre[M][:,1] = ( x .+ 1 ) / 2
        unitlegendre[M][:,2] = w / 2
        jacobi1[M] = Array{T}(undef, M, 2)
        jacobi1[M][:,1], jacobi1[M][:,2] = GaussQuadrature.jacobi(M, α-1, 
                                                                  zero(T))
        jacobi2[M] = Array{T}(undef, M, 2)
        jacobi2[M][:,1], jacobi2[M][:,2] = GaussQuadrature.jacobi(M, zero(T), 
                                                                  α-1)
        jacobi3[M] = Array{T}(undef, M, 2)
        jacobi3[M][:,1], jacobi3[M][:,2] = GaussQuadrature.jacobi(M, α, zero(T))
        jacobi4[M] = Array{T}(undef, M, 2)
        jacobi4[M][:,1], jacobi4[M][:,2] = GaussQuadrature.jacobi(M, zero(T), α)
    end
    Ψ  = Array{T}(undef, rmax, Mmax)
    dΨ = Array{T}(undef, rmax, Mmax)
    A = Vector{T}(undef, rmax)
    B = Vector{T}(undef, rmax)
    C = Array{T}(undef, rmax, rmax)
    return Store(α, rmax, Mmax, unitlegendre, legendre, 
                 jacobi1, jacobi2, jacobi3, jacobi4,
                 Ψ, dΨ, A, B, C)
end

function rule(storerule::Matrix{T}) where T <: AbstractFloat
    return storerule[:,1], storerule[:,2]
end

# Anℓ for 1 ≤ ℓ ≤ n.
function A_integral!(r::Integer, n::Integer, ℓ::Integer,  
                     t::OffsetVector{T}, M::Integer,
                     store::Store{T}) where T <: AbstractFloat
    kn = t[n] - t[n-1]
    kℓ = t[ℓ] - t[ℓ-1]
    two_Dnℓ = (t[n]+t[n-1]) - (t[ℓ]+t[ℓ-1])
    σ, w = rule(store.legendre[M])
    Ψ = view(store.Ψ, 1:r, 1:M)
    legendre_polys!(Ψ, σ)
    α = store.α
    for j = 1:r
        s = zero(T)
        for m = 1:M
            Δ = ( kn - σ[m]*kℓ ) / two_Dnℓ
            s += w[m] * ( 1 + Δ )^(α-1) * Ψ[j,m]
        end
        store.A[j] = s
    end
end

# Bnℓ when ℓ = n-1
function B1_integral!(rℓ::Integer, n::Integer, t::OffsetVector{T},
                     store::Store{T}) where T <: AbstractFloat
    M = ceil(Integer, rℓ/2)
    σ, w = rule(store.jacobi1[M])
    Ψ = view(store.Ψ, 1:rℓ, 1:M)
    legendre_polys!(Ψ, σ)
    kn = t[n] - t[n-1]
    knm1 = t[n-1] - t[n-2] 
    α = store.α
    for j = 1:rℓ
        s = zero(T)
        for m = 1:M
            s+= w[m] * Ψ[j,m]
        end
        store.B[j] = ( (kn+knm1) / knm1 )^(1-α) * s
    end
end

# Bnℓ when ℓ ≤ n-2
function B_integral!(r::Integer, n::Integer, ℓ::Integer, 
                    t::OffsetVector{T}, M::Integer,
                    store::Store{T}) where T <: AbstractFloat
    kn = t[n] - t[n-1]
    kℓ = t[ℓ] - t[ℓ-1]
    two_Dnℓ = (t[n]+t[n-1]) - (t[ℓ]+t[ℓ-1])
    σ, w = rule(store.legendre[M])
    Ψ = view(store.Ψ, 1:M, 1:r)
    legendre_polys!(Ψ, σ)
    α = store.α
    for j = 1:r
        s = zero(T)
        for m = 1:M
            Δ = -( kn + σ[m]*kℓ ) / two_Dnℓ
            s += w[m] * ( 1 + Δ )^(α-1) * Ψ[j,m]
        end
        store.B[j] = s
    end
end

# Cnℓ when ℓ = n-1
function C1_integral!(rn::Integer, rℓ::Integer, n::Integer, 
                      t::OffsetVector{T}, M::Integer,
                      store::Store{T}) where T <: AbstractFloat
    Mz = Mτ = Mσ = M
    z, wz = rule(store.unitlegendre[Mz])
    τ, wτ = rule(store.jacobi4[Mτ])
    σ, wσ = rule(store.jacobi3[Mσ])
    dΨ = view(store.dΨ, 1:rn, 1:M)
    Ψ  = view(store.Ψ, 1:rℓ, 1:M)
    kn   = t[n] - t[n-1]
    knm1 = t[n-1] - t[n-2]
    two_D = kn + knm1
    α = store.α
    deriv_legendre_polys!(dΨ, τ)
    for j = 1:rℓ, i = 1:rn
        outer = zero(T)
        for mτ = 1:Mτ
            τ_ = τ[mτ]
            inner = zero(T)
            for mz = 1:Mz
                z_ = z[mz]
                legendre_polys!(view(Ψ, 1:j), 1 - z_*(1+τ_))
                inner += wz[mz] * ((kn+z_*knm1)/two_D)^(α-1) * Ψ[j]
            end
            outer += wτ[mτ] * dΨ[i,mτ] * inner
        end
        store.C[i,j] = outer # first term
    end
    legendre_polys!(Ψ, σ)
    for j = 1:rℓ, i = 1:rn
        outer = zero(T)
        for mσ = 1:Mσ
            σ_ = σ[mσ]
            inner = zero(T)
            for mz = 1:Mz
                z_ = z[mz]
                deriv_legendre_polys!(view(dΨ, 1:i), z_*(1-σ_)-1)
                inner += wz[mz] * ((z_*kn+knm1)/two_D)^(α-1) * dΨ[i]
            end
            outer += wσ[mσ] * Ψ[j,mσ] * inner
        end
        store.C[i,j] += outer # add second term
    end
end

# Hnℓ for 1 ≤ ℓ ≤ n-2, first version
function H_history_ver1!(Hnℓ::Matrix{T}, n::Integer, ℓ::Integer, 
                         t::OffsetVector{T}, Ψ::Matrix{T}, dΨ::Matrix{T}, 
                         σ::Vector{T}, wσ::Vector{T}) where T <: AbstractFloat
    kn = t[n] - t[n-1]
    kℓ = t[ℓ] - t[ℓ-1]
    two_Dnℓ = (t[n]+t[n-1]) - (t[l]+t[ℓ+1])
    Dnℓ = two_Dnℓ / 2
    rn, rℓ = size(Hnℓ)
    A = Vector{T}(undef, rℓ)
    A_integral_uniform!(A, ℓ, α, Ψ, σ, w)
    B = Vector{T}(undef, r)
    B_integral_uniform!(B, ℓ, α, Ψ, σ, w)
    C_integral!(Hnℓ, kn, kℓ, two_Dnℓ, α, Ψ, dΨ, σ, wσ)
    for j = 1:rℓ
        pow = -one(T)
        for i = 1:rn
            pow = -pow
            Hnℓ[i,j] = c * ( A[j] + pow * B[j] - Hnℓ[i,j] )
        end
    end
end

# Hnℓ for 1 ≤ ℓ ≤ n-2, second version
function H_history_ver2!(Hnℓ::Matrix{T}, n::Integer, ℓ::Integer, 
                    t::OffsetVector{T}, Ψ::Matrix{T}, σ::Vector{T}, 
                    wσ::Vector{T}) where T <: AbstractFloat
    kn = t[n] - t[n-1]
    kℓ = t[ℓ] - t[ℓ-1]
    two_Dnℓ = (t[n]+t[n-1]) - (t[l]+t[ℓ+1])
    Dnℓ = two_Dnℓ / 2
    c = - (1-α) * kn * kℓ * Dnℓ^(α-2) / ( 4Γ(α) )
    for j = 1:rℓ, i = 1:rn
        outer = zero(T)
        for mτ = 1:M
            inner = zero(T)
            for mσ = 1:M
                Δ = ( τ[mτ] * kn - σ[mσ] * kℓ ) / two_Dnℓ
                inner += wσ[mσ] * (1+Δ)^(α-2) * Ψ[j,mσ]
            end
            outer += wτ[mτ] * Ψ[i,mτ] * inner
        end
        H[ℓ][i,j] = c * outer
    end
end

# Cnℓ for 1 ≤ ℓ ≤ n-2.
function C_integral!(rn::Integer, rℓ::Integer, n::Integer, ℓ::Integer, 
                    t::OffsetVector{T}, M::Integer,
                    store::Store{T}) where T <: AbstractFloat
    kn = t[n] - t[n-1]
    kℓ = t[ℓ] - t[ℓ-1]
    two_Dnℓ = (t[n]+t[n-1]) - (t[ℓ]+t[ℓ-1])
    σ, wσ = rule(store.legendre[M])
    τ, wτ = σ, wσ
    dΨ = view(store.dΨ, 1:rn, 1:M)
    Ψ  = view(store.Ψ,  1:rℓ, 1:M)
    deriv_legendre_polys!(dΨ, τ)
    legendre_polys!(Ψ, σ)
    α = store.α
    for j = 1:rℓ, i = 1:rn
        outer = zero(T)
        for mτ = 1:M
            inner = zero(T)
            for mσ = 1:M
                Δ = ( τ[mτ]*kn - σ[mσ]*kℓ ) / ( two_Dnℓ )
                inner += wσ[mσ] * (1+Δ)^(α-1) * Ψ[j,mσ]
            end
            outer += wτ[mτ] * dΨ[i,mτ] * inner
        end
        store.C[i,j] = outer
    end
end

function C1_integrals!(rn::Integer, rℓ::Integer, n::Integer, α::T, 
                      t::OffsetVector{T}, M::Integer,
                      store::Store{T}) where T <: AbstractFloat
    C1_first  = Array{T}(undef, r1, r2)
    C1_second = Array{T}(undef, r1, r2)
    kn   = t[n] - t[n-1]
    knm1 = t[n-1] - t[n-2]
    z, wz = GaussQuadrature.legendre(T, M)
    z  .= ( z .+ 1 ) / 2
    wz .= wz / 2
    τ, wτ = GaussQuadrature.jacobi(M, zero(T), α)
    dΨ = Array{T}(undef, r1, M) 
    Ψ  = Array{T}(undef, r2, M) 
    deriv_legendre_polys!(dΨ, τ)
    C1_first_term!(C1_first, kn, knm1, α, Ψ, dΨ, τ, wτ, z, wz)
    σ, wσ = GaussQuadrature.jacobi(M, α, zero(T))
    legendre_polys!(Ψ, σ)
    C1_second_term!(C1_second, kn, knm1, α, Ψ, dΨ, σ, wσ, z, wz)
    return C1_first, C1_second
end

# first term in Cnℓ when ℓ = 1
function C1_first_term!(C1_first::Matrix{T}, kn::T, knm1::T,
                        α::T, Ψ::Matrix{T}, dΨ::Matrix{T},
                        τ::Vector{T}, wτ::Vector{T}, 
                        z::Vector{T}, wz::Vector{T}) where T <: AbstractFloat
    r1,r2 = size(C1_first)
    Mτ = length(wτ)
    Mz = length(z)
    two_D = kn + knm1
    for j = 1:r2
        for i = 1:r1
            outer = zero(T)
            for mτ = 1:Mτ
                τ_ = τ[mτ]
                inner = zero(T)
                for mz = 1:Mz
                    z_ = z[mz]
                    legendre_polys!(view(Ψ, 1:j, 1), 1 - z_*(1+τ_))
                    inner += wz[mz] * ((kn+z_*knm1)/two_D)^(α-1) * Ψ[j,1]
                end
                outer += wτ[mτ] * dΨ[i,mτ] * inner
            end
            C1_first[i,j] = outer
        end
    end
end

# second term in Cnℓ when ℓ = 1
function C1_second_term!(C1_second::Matrix{T}, kn::T, knm1::T, α::T, 
                         Ψ::Matrix{T}, dΨ::Matrix{T},
                         σ::Vector{T}, wσ::Vector{T}, 
                         z::Vector{T}, wz::Vector{T}) where T <: AbstractFloat
    r1, r2 = size(C1_second)
    Mσ = length(wσ)
    Mz = length(z)
    two_D = kn + knm1
    for j = 1:r2
        for i = 1:r1
            outer = zero(T)
            for mσ = 1:Mσ
                σ_ = σ[mσ]
                inner = zero(T)
                for mz = 1:Mz
                    z_ = z[mz]
                    deriv_legendre_polys!(view(dΨ, 1:i, 1), z_*(1-σ_)-1)
                    inner += wz[mz] * ((z_*kn+knm1)/two_D)^(α-1) * dΨ[i,1]
                end
                outer += wσ[mσ] * Ψ[j,mσ] * inner
            end
            C1_second[i,j] = outer
        end
    end
end
