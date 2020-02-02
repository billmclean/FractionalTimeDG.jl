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

