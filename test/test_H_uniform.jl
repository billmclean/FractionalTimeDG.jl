import QuadGK.quadgk

α = 3/4
r = 4

H0 = coef_H0(r, α)

# \int_0^\tau (\tau-\sigma)^{\alpha-1} P_{j-1}(\tau)\,d\tau
function inner_integral(j::Integer, τ::T, α::T,
                       rtol) where T <: AbstractFloat
    I, err = quadgk(-one(T), τ, rtol=rtol) do σ
        (τ-σ)^(α-1) * P(j-1, σ) 
    end
    return I
end

function analytical_inner_integral(j::Integer, τ::T, 
                                   α::T) where T <: AbstractFloat
    n = j - 1
    pow = 1 / α
    s = pow
    for k = 1:n
        pow *= -k * ( τ + 1 ) / ( 2(k+α) )
        s += binomial(n,k) * binomial(n+k,k) * pow 
    end
    return (τ+1)^α * (-1)^n * s
end

# special case α = 1
function inner_integral_1(j::Integer, τ::AbstractFloat)
    n = j - 1
    if n == 0
        return τ + 1
    else
        return ( P(n+1, τ) - P(n-1, τ) ) / ( 2n + 1 )
    end
end

function brute_force_H0(r::Integer, α::T, rtol=1e-8) where T <: AbstractFloat
    H0 = Array{T}(undef, r, r)
    c = 1 / (2^α*Γ(α))
    for j = 1:r, i = 1:r
        I, err = quadgk(-one(T), one(T), rtol=rtol) do τ
            dP(i-1, τ) * inner_integral(j, τ, α, rtol)
        end
        H0[i,j] = c * ( inner_integral(j, one(T), α, rtol) - I )
    end
    return H0
end

bf_H0 = brute_force_H0(r, α)

err = bf_H0 - H0
@test all( abs.(err) .< 1e-8 )

H1 = coef_H1(r, α, r+2)

Δ(ℓ, τ) = τ / (2ℓ)

function inner_integral_1(j::Integer, σ::T, α::T, rtol) where T <: AbstractFloat
    I, err = quadgk(zero(T), one(T), rtol=rtol) do z
        (1+z)^(α-1) * P(j-1,1-z*(1+σ))
    end
    return I
end

function inner_integral_2(i::Integer, σ::T, α::T, rtol) where T <: AbstractFloat
    I, err = quadgk(zero(T), one(T), rtol=rtol) do z
        (z+1)^(α-1) * dP(i-1,z*(1-σ)-1)
    end
    return I
end

function inner_integral_3(j::Integer, τ::T, α::T, rtol) where T <: AbstractFloat
    I, err = quadgk(-one(T), one(T), rtol=rtol) do σ 
        ( 1 + Δ(1,τ-σ) )^(α-1) * P(j-1, σ)
    end
    return I
end
function brute_force_H1(r::Integer, α::T, rtol=1e-8) where T <: AbstractFloat
    H1 = Array{T}(undef, r, r)
    c = 1 / ( 2Γ(α) )
    for j = 1:r
        Aj, err = quadgk(-one(T), one(T), rtol=rtol) do σ
            (1+Δ(1,1-σ))^(α-1) * P(j-1,σ)
        end
        Bj, err = quadgk(-one(T), one(T), rtol=rtol) do σ
            (1-Δ(1,1+σ))^(α-1) * P(j-1,σ)
        end
        pow = one(T)
        for i = 1:r
#            Cij_1, err = quadgk(-one(T), one(T), rtol=rtol) do σ
#                (1+σ)^α * dP(i-1,σ) * inner_integral_1(j, σ, α, rtol)
#            end
#            Cij_1 *= 2^(1-α)
#            Cij_2, err = quadgk(-one(T), one(T), rtol=rtol) do σ
#                (1-σ)^α * P(j-1,σ) * inner_integral_2(i, σ, α, rtol)
#            end
#            Cij_2 *= 2^(1-α)
            Cij, err = quadgk(-one(T), one(T), rtol=rtol) do τ
                dP(i-1,τ) * inner_integral_3(j, τ, α, rtol)
            end
            pow = -pow
            H1[i,j] = c * ( Aj + pow * Bj - Cij )
        end
    end
    return H1
end

bf_H1 = brute_force_H1(r, α)
err = bf_H1 - H1
@test all( abs.(err) .< 1e-7 )
