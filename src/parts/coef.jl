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

function coef_Hn!(r::Integer, n::Integer, ℓbar_hi::Integer, t::OffsetVector{T}, 
                 M::Integer, store::Store{T}, 
                 version=2) where T <: AbstractFloat
    Hn = OffsetVector{Matrix{T}}(undef, 0:ℓbar_hi)
    for ℓbar = 0:ℓbar_hi
        Hn[ℓbar] = Array{T}(undef, r, r)
    end
    coef_Hn!(Hn, n, ℓbar_hi, t, M, store, version)
    return Hn
end

function coef_Hn!(Hn::OffsetVector{Matrix{T}}, n::Integer, ℓbar_hi::Integer,
                 t::OffsetVector{T}, M::Integer,
                 store::Store{T}, version=2) where T <: AbstractFloat
    N = length(t) - 1
    @argcheck 1 ≤ n ≤ N
    @argcheck 0 ≤ ℓbar_hi ≤ n-1
    @argcheck length(Hn) ≥ ℓbar_hi + 1
    rn = size(Hn[0], 1)
    @argcheck size(Hn[0], 2) == rn
    @argcheck rn ≤ store.rmax
    coef_H0_uniform!(Hn[0], store)
    kn = t[n] - t[n-1]
    α = store.α
    for j = 1:rn, i = 1:rn
        Hn[0][i,j] *= kn^α 
    end
    if ℓbar_hi ≥ 1
        coef_Hn1!(Hn[1], n, t, M, store)
    end
    if ℓbar_hi ≥ 2
        if version == 1
            coef_Hn_ver1!(Hn, n, 2:ℓbar_hi, t, M, store)
        elseif version == 2
            coef_Hn_ver2!(Hn, n, 2:ℓbar_hi, t, M, store)
        else
            throw(ArgumentError("version must be 1 or 2"))
        end
    end
end

function coef_Hn1!(rn::Integer, rnm1::Integer, n::Integer, t::OffsetVector{T}, 
                   M::Integer, store::Store{T}) where T <: AbstractFloat
    Hn1 = Array{T}(undef, rn, rnm1)
    coef_Hn1!(Hn1, n, t, M, store)
    return Hn1
end

function coef_Hn1!(Hn1::Matrix{T}, n::Integer, t::OffsetVector{T}, M::Integer,
                   store::Store{T}) where T <: AbstractFloat
    rn, rnm1 = size(Hn1)
    @argcheck rnm1 ≤ store.rmax
    A = view(store.A, 1:rnm1)
    B = view(store.B, 1:rnm1)
    C = view(store.C, 1:rn, 1:rnm1)
    α = store.α
    kn = t[n] - t[n-1]
    knm1 = t[n-1] - t[n-2]
    ρn = kn / knm1

    σ, wσ = rule(store.legendre[M])
    Ψ = view(store.Ψ, 1:rnm1, 1:M)
    legendre_polys!(Ψ, σ)
    c = (1+ρn)^(1-α)
    for j = 1:rnm1
        s = zero(T)
        for m = 1:M
            s += wσ[m] * ( 2ρn + 1 - σ[m] )^(α-1) * Ψ[j,m]
        end
        A[j] = c * s
    end

    Mσ = ceil(Integer, rnm1/2)
    σ, wσ = rule(store.jacobi1[Mσ])
    Ψ = view(store.Ψ, 1:rnm1, 1:Mσ)
    legendre_polys!(Ψ, σ)
    for j = 1:rnm1
        s = zero(T)
        for m = 1:Mσ
            s+= wσ[m] * Ψ[j,m]
        end
        B[j] = c * s
    end

    Mτ = ceil(Integer, rn-1)
    Mσ = Mτ
    Mz = M
    z, wz = rule(store.unitlegendre[Mz])
    σ, wσ = rule(store.jacobi3[Mσ])
    τ, wτ = rule(store.jacobi4[Mτ])
    dΨ = view(store.dΨ, 1:rn, 1:Mτ)
    deriv_legendre_polys!(dΨ, τ)
    for j = 1:rnm1
        Ψ = view(store.Ψ, 1:j)
        for i = 1:rn
            outer = zero(T)
            for mτ = 1:Mτ
                τ_ = τ[mτ]
                inner = zero(T)
                for mz = 1:Mz
                    z_ = z[mz]
                    legendre_polys!(Ψ, 1 - z_*(1+τ_))
                    inner += wz[mz] * ( ρn + z_ )^(α-1) * Ψ[j]
                end
                outer += wτ[mτ] * dΨ[i,mτ] * inner
            end
            C[i,j] = c * outer
        end
    end
    Ψ = view(store.Ψ, 1:rnm1, 1:Mσ)
    legendre_polys!(Ψ, σ)
    for j = 1:rnm1
        for i = 1:rn
            dΨ = view(store.dΨ, 1:i)
            outer = zero(T)
            for mσ = 1:Mσ
                σ_ = σ[mσ]
                inner = zero(T)
                for mz = 1:Mz
                    z_ = z[mz]
                    deriv_legendre_polys!(dΨ, z_*(1-σ_)-1)
                    inner += wz[mz] * ( ρn*z_ + 1 )^(α-1) * dΨ[i]
                end
                outer += wσ[mσ] * Ψ[j,mσ] * inner
            end
            C[i,j] += c * outer
        end
    end

    D = ( kn + knm1 ) / 2
    c = D^(α-1) * knm1 / ( 2Γ(α) )
    for j = 1:rnm1
        pow = one(T)
        for i = 1:rn
            pow = -pow
            Hn1[i,j] = c * ( A[j] + pow * B[j] - C[i,j] )
        end
    end
end

function coef_Hn_ver1!(Hn::OffsetVector{Matrix{T}}, n::Integer, rng::UnitRange,
                       t::OffsetVector{T}, M::Integer, 
                       store::Store{T}) where T <: AbstractFloat
    @argcheck rng.start ≥ 2
    rn = size(Hn[rng.start], 1)
    kn = t[n] - t[n-1]

    σ, wσ = rule(store.legendre[M])
    τ, wτ = σ, wσ
    Ψ  = view(store.Ψ,  1:store.rmax, 1:M)
    dΨ = view(store.dΨ, 1:store.rmax, 1:M)
    legendre_polys!(Ψ, σ)
    deriv_legendre_polys!(dΨ, τ)
    α = store.α
    c = 1 / ( 2Γ(α) )
    for ℓbar in rng
        rℓ = size(Hn[ℓbar], 2)
        @argcheck rℓ ≤ store.rmax
        ℓ = n - ℓbar
        kℓ = t[ℓ] - t[ℓ-1]
        two_D = (t[n]+t[n-1]) - (t[ℓ]+t[ℓ-1])

        A = view(store.A, 1:rℓ)
        B = view(store.B, 1:rℓ)
        C = view(store.C, 1:rn, 1:rℓ)

        for j = 1:rℓ
            s = zero(T)
            for m = 1:M
                Δ = ( kn - σ[m]*kℓ ) / two_D
                s += wσ[m] * ( 1 + Δ )^(α-1) * Ψ[j,m]
            end
            A[j] = s
        end

        for j = 1:rℓ
            s = zero(T)
            for m = 1:M
                Δ = ( -kn - σ[m]*kℓ ) / two_D
                s += wσ[m] * ( 1 + Δ )^(α-1) * Ψ[j,m]
            end
            B[j] = s
        end

        for j = 1:rℓ, i = 1:rn
            outer = zero(T)
            for mτ = 1:M
                inner = zero(T)
                for mσ = 1:M
                    Δ = ( τ[mτ]*kn - σ[mσ]*kℓ ) / two_D
                    inner += wσ[mσ] * (1+Δ)^(α-1) * Ψ[j,mσ]
                end
                outer += wτ[mτ] * dΨ[i,mτ] * inner
            end
            C[i,j] = outer
        end

        D = two_D / 2
        for j = 1:rℓ
            pow = one(T)
            for i = 1:rn
                pow = -pow
                Hn[ℓbar][i,j] = c * D^(α-1) * kℓ * ( 
                                A[j] + pow * B[j] - C[i,j] )
            end
        end
    end
end

function coef_Hn_ver2!(Hn::OffsetVector{Matrix{T}}, n::Integer, rng::UnitRange,
                       t::OffsetVector{T}, M::Integer, 
                       store::Store{T}) where T <: AbstractFloat
    @argcheck rng.start ≥ 2
    rn = size(Hn[rng.start], 1)
    kn = t[n] - t[n-1]

    σ, wσ = rule(store.legendre[M])
    τ, wτ = σ, wσ
    Ψ  = view(store.Ψ,  1:store.rmax, 1:M)
    legendre_polys!(Ψ, σ)
    α = store.α
    c = - (1-α) / ( 4Γ(α) )
    for ℓbar in rng
        rℓ = size(Hn[ℓbar], 2)
        @argcheck rℓ ≤ store.rmax
        ℓ = n - ℓbar
        kℓ = t[ℓ] - t[ℓ-1]
        two_D = (t[n]+t[n-1]) - (t[ℓ]+t[ℓ-1])
        D = two_D / 2

        for j = 1:rℓ, i = 1:rn
            outer = zero(T)
            for mτ = 1:M
                inner = zero(T)
                for mσ = 1:M
                    Δ = ( τ[mτ]*kn - σ[mσ]*kℓ ) / two_D
                    inner += wσ[mσ] * (1+Δ)^(α-2) * Ψ[j,mσ]
                end
                outer += wτ[mτ] * Ψ[i,mτ] * inner
            end
            Hn[ℓbar][i,j] = c * kn * kℓ * D^(α-2) * outer
        end
    end
end

