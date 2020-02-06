
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

function coef_H1(r::Integer, α::T, kn::T, knm1::T, 
                 M::Integer) where T <: AbstractFloat
    H1 = Array{T}(undef, r, r)
    coef_H1!(H1, α, kn, knm1, M)
    return H1
end

function coef_H1!(H1::AbstractMatrix{T}, α::T,
                  kn::T, knm1::T, M::Integer) where T <: AbstractFloat
    rn = size(H1, 1)
    rnm1 = size(H1, 2)
    σ1, w1 = GaussQuadrature.legendre(T, M)
    Ψ = Array{T}(undef, rnm1, M)
    legendre_polys!(Ψ, σ1)
    two_D_n_nm1 = kn + knm1 
    D_n_nm1 = two_D_n_nm1 / 2
    for j = 1:rnm1
        Aj = zero(T)
        for m = 1:M
            Δ = ( kn - σ1[m]*knm1 ) / two_D_n_nm1
            Aj += w1[m] * ( 1 + Δ )^(α-1) * Ψ[j,m]
        end
        for i = 1:rn
            H1[i,j] = Aj
        end
    end
    σ2, w2 = GaussQuadrature.jacobi(M, α-1, zero(T))
    legendre_polys!(Ψ, σ2)
    for j = 1:rnm1
        Bj = zero(T)
        for m = 1:M
            Bj += w2[m] * Ψ[j,m]
        end
        Bj *= ( two_D_n_nm1 / knm1 )^(1-α)
        pow = one(T)
        for i = 1:rn
            pow = -pow
            H1[i,j] += pow * Bj
        end
    end
    σ3, w3 = GaussQuadrature.jacobi(M, zero(T), α)
    z, wz = GaussQuadrature.legendre(T, M)
    z .= ( z .+ 1 ) / 2
    wz .= wz/2
    dΨ = Array{T}(undef, rn, M)
    deriv_legendre_polys!(dΨ, σ3)
    fill!(H1, zero(T))
    for j = 1:rnm1
        for i = 1:rn
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
    dΨ = Array{T}(undef, rnm1, M)
    c = D_n_nm1^(α-1) * knm1 / ( 2Γ(α) )
    for j = 1:rnm1
        for i = 1:rn
            Cij_2 = zero(T)
            for m4 = 1:M
                σ_ = σ4[m4]
                inner = zero(T)
                for mz = 1:M
                    z_ = z[mz]
                    deriv_legendre_polys!(view(dΨ, 1:i, 1), z_*(1-σ_)-1)
                    inner += wz[mz] * ((z_*kn+knm1)/two_D_n_nm1)^(α-1) * dΨ[i,1]
                end
                Cij_2 += w4[m4] * Ψ[j,m4] * inner
            end
#            println("C2_$i,$j = $Cij_2")
            H1[i,j] -= Cij_2
#            H1[i,j] *= c
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

function coef_H_uniform_1!(Hℓ::AbstractMatrix{T}, ℓ::Integer, α::T, 
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

function coef_H_uniform!(H::Array{T,3}, ℓ_range::UnitRange, 
                         α::T, σ::Vector{T}, w::Vector{T},
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
