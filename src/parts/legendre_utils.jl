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

function legendre_polys!(P::AbstractMatrix{T}, τ::AbstractVector{T}
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

function deriv_legendre_polys!(dP::AbstractMatrix{T}, τ::AbstractVector{T}
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

"""
    pcwise_t, pcwise_U = evaluate_pcwise_poly(U, t, ppI, store)
"""
function evaluate_pcwise_poly!(U::Vector{Vector{T}}, t::OffsetVector{T},
                               ppI::Integer, store::Store{T}
                              ) where { T <: AbstractFloat }
    τ = range(-one(T), stop=one(T), length=ppI) 
    pcwise_t, pcwise_U = evaluate_pcwise_poly!(U, t, τ, store) 
    return pcwise_t, pcwise_U
end

function evaluate_pcwise_poly!(U::Vector{Vector{T}}, t::OffsetVector{T},
                               τ::AbstractVector{T}, store::Store{T}
                              ) where T <: AbstractFloat 
    N = length(U)
    ppI = length(τ)
    @argcheck ppI ≤ store.ppImax
    pcwise_t = Array{T}(undef, ppI, N)
    pcwise_U = similar(pcwise_t) 
    evaluate_pcwise_poly!(pcwise_t, pcwise_U, U, t, τ, store) 
    return pcwise_t, pcwise_U
end

function evaluate_pcwise_poly!(pcwise_t::Matrix{T}, pcwise_U::Matrix{T},
                               U::Vector{Vector{T}}, t::OffsetVector{T},
                               τ::AbstractVector{T}, store::Store{T}
                              ) where { T <: AbstractFloat }
    N = length(U)
    pts_per_interval = length(τ)
    @argcheck size(pcwise_t) == (pts_per_interval, N)
    @argcheck size(pcwise_U) == (pts_per_interval, N)
    @argcheck length(t) == N+1
    rmax = store.rmax
    Ψ = view(store.Ψ, 1:rmax, 1:pts_per_interval)
    legendre_polys!(Ψ, τ)
    for n = 1:N
        rn = length(U[n])
        for m = 1:pts_per_interval
            pcwise_t[m,n] = ( (1-τ[m])*t[n-1] + (1+τ[m])*t[n] ) / 2
            s = zero(T)
            for j = 1:rn
                s += U[n][j] * Ψ[j,m]
            end
            pcwise_U[m,n] = s
        end
    end
end

function Fourier_Legendre_coefs(u::Function, r::Integer, t::OffsetArray{T},
                                store::Store{T}) where T <: AbstractFloat
    N = length(t) - 1
    rmax = store.rmax
    a = Vector{Vector{T}}(undef, N)
    M = store.Mmax
    τ, wτ = rule(store.legendre[M])
    Ψ = view(store.Ψ, 1:r, 1:M)
    legendre_polys!(Ψ, τ)
    for n = 1:N
        a[n] = Vector{T}(undef, r)
        for j = 1:r
            s = zero(T)
            for m = 1:M
                tnm = ( (1-τ[m])*t[n-1] + (1+τ[m])*t[n] ) / 2
                s += wτ[m] * u(tnm) * Ψ[j,m]
            end
            a[n][j] = (2j-1) * s / 2
        end
    end
    return a
end

function reconstruction_pts(In::Tuple{T,T}, r::Integer) where T <: AbstractFloat
    tnm1, tn = In
    tstar = reconstruction_pts(T, r)
    tstar[0] = tnm1
    for m = 1:r-1
        tstar[m] = ( ( 1 - tstar[m] ) * tnm1 + ( tstar[m] + 1 ) * tn ) / 2
    end
    tstar[r] = tn
    return tstar
end

function reconstruction_pts(::Type{T}, r::Integer) where T <: AbstractFloat
    right = GaussQuadrature.right
    τ, wτ = GaussQuadrature.legendre(T, r, right)
    return OffsetArray([ -one(0); τ ], 0:r)
end

"""
    Uhat = reconstruction(U, u0, store)
"""
function reconstruction(U::Vector{Vector{T}}, u0::T, 
                        store::Store{T}) where T <: AbstractFloat
    N = length(U)
    rmax = store.rmax
    Uhat = Vector{Vector{T}}(undef, N)
    pow = OffsetArray{T}(undef, 0:rmax)
    pow[0] = one(T)
    for n = 1:rmax
        pow[n] = -pow[n-1]
    end
    r1 = length(U[1])
    @argcheck r1+1 ≤ rmax
    U_left = zero(T)   # U_left  = U(t[0] + 0) = value at the left endpoint.
    U_right = zero(T)  # U_right = U(t[1] - 0) = value at the right endpoint.
    for j = 1:r1
        U_left += pow[j-1] * U[1][j]
        U_right += U[1][j]
    end
    jumpU0 = U_left - u0
    Uhat[1] = Vector{T}(undef, r1+1)
    Uhat[1][1:r1] .= U[1][1:r1]
    Uhat[1][r1] += pow[r1] * jumpU0 / 2
    Uhat[1][r1+1] = - pow[r1] * jumpU0 / 2
    for n = 2:N
        rn = length(U[n])
        @argcheck rn + 1 ≤ rmax
        U_left = zero(T) # U_left = U(t[n-1] + 0 )
        for j = 1:rn
            U_left += pow[j-1] * U[n][j]
        end
        jumpUnm1 = U_left - U_right
        Uhat[n] = Vector{T}(undef, rn+1)
        Uhat[n][1:rn] .= U[n][1:rn]
        Uhat[n][rn] += pow[rn] * jumpUnm1 / 2
        Uhat[n][rn+1]  = -pow[rn] * jumpUnm1 / 2
        U_right = zero(T) # U_right = U(t[n] - 0 )
        for j = 1:rn
            U_right += U[n][j]
        end
    end
    return Uhat
end

function max_order(U::Vector{Vector{T}}) where T <: AbstractFloat
    N = length(U)
    r = length(U[1])
    for n = 2:N
        rn = length(U[n])
        r = max(r, rn)
    end
    return r
end

function jumps(U::Vector{Vector{T}}, t::OffsetVector{T}, 
	u0::T) where T <: AbstractFloat
    N = length(t) - 1
    r = max_order(U)
    JU = OffsetVector{T}(undef, 0:N-1)
    U_left = u0
    U_right = zero(T)
    pow = ones(T, r)
    pow[2:2:r] .= -one(T)
    for n = 1:N
	U_left = zero(T)
	for j = 1:r
	    U_left += pow[j] * U[n][j]
	end
	JU[n-1] = U_left
    end
    JU[0] -= u0
    for n = 2:N
	U_right = sum(U[n-1])
	JU[n-1] -= U_right
    end
    return JU
end

function approx_dG_error(JU::OffsetVector{T}, t::OffsetVector{T},
        r::Integer, τ::AbstractVector{T}) where T <: AbstractFloat
    N = length(t) - 1
    pts_per_interval = length(τ)
    pcwise_t = Matrix{T}(undef, pts_per_interval, N)
    pcwise_approx_err = similar(pcwise_t)
    Ψ = Matrix{T}(undef, r+1, pts_per_interval)
    legendre_polys!(Ψ, τ)
    if r % 2 == 0
        sign = 1
    else
        sign = -1
    end
    for n = 1:N
        for m = 1:pts_per_interval
            pcwise_t[m,n] = ( (1-τ[m])*t[n-1] + (1+τ[m])*t[n] ) / 2
            pcwise_approx_err[m,n] = (sign/2) * JU[n-1] * ( Ψ[r+1,m] - Ψ[r,m] )
        end
    end
    return pcwise_t, pcwise_approx_err
end

function approx_dG_error(JU::OffsetVector{T}, t::OffsetVector{T},
        r::Integer, pts_per_interval::Integer) where T <: AbstractFloat
    τ = range(-1.0, 1.0, length=pts_per_interval)
    return approx_dG_error(JU, t, r, τ)
end
