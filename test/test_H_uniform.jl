import QuadGK.quadgk

α = 3/4
r = 4

H0 = coef_H0(r, α)

# \int_0^\tau (\tau-\sigma)^{\alpha-1} P_{j-1}(\tau)\,d\tau
function inner_integral_0(j::Integer, τ::T, α::T,
                       rtol) where T <: AbstractFloat
    I, err = quadgk(-one(T), τ, rtol=rtol) do σ
        (τ-σ)^(α-1) * P(j-1, σ) 
    end
    return I
end

function analytical_inner_integral_0(j::Integer, τ::T, 
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
function inner_integral_0_1(j::Integer, τ::AbstractFloat)
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
            dP(i-1, τ) * inner_integral_0(j, τ, α, rtol)
        end
        H0[i,j] = c * ( inner_integral_0(j, one(T), α, rtol) - I )
    end
    return H0
end

bf_H0 = brute_force_H0(r, α)

err0 = bf_H0 - H0
@test all( abs.(err0) .< 1e-8 )

H = coef_H_uniform(5, r, α, r+2)

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

function inner_integral(ℓ::Integer, j::Integer, τ::T, α::T, 
                        rtol) where T <: AbstractFloat
    I, err = quadgk(-one(T), one(T), rtol=rtol) do σ 
        ( 1 + Δ(ℓ,τ-σ) )^(α-1) * P(j-1, σ)
    end
    return I
end
function brute_force_H(ℓ::Integer, r::Integer, α::T, 
                       rtol=1e-8) where T <: AbstractFloat
    Hℓ = Array{T}(undef, r, r)
    c = ℓ^(α-1) / ( 2Γ(α) )
    for j = 1:r
        Aj, err = quadgk(-one(T), one(T), rtol=rtol) do σ
            (1+Δ(ℓ,1-σ))^(α-1) * P(j-1,σ)
        end
        Bj, err = quadgk(-one(T), one(T), rtol=rtol) do σ
            (1-Δ(ℓ,1+σ))^(α-1) * P(j-1,σ)
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
                dP(i-1,τ) * inner_integral(ℓ, j, τ, α, rtol)
            end
            pow = -pow
            Hℓ[i,j] = c * ( Aj + pow * Bj - Cij )
        end
    end
    return Hℓ
end

bf_H1 = brute_force_H(1, r, α)
err1 = bf_H1 - H[:,:,1]
@test all( abs.(err1) .< 1e-8 )

bf_H4 = brute_force_H(4, r, α)
err4 = bf_H4 - H[:,:,4]
@test all( abs.(err4) .< 1e-12 )

alt_H = Array{Float64}(undef, r, r, 4)
coef_H_uniform!(alt_H, 3:4, r, α, r+2)
err3 = alt_H[:,:,3] - H[:,:,3]
err4 = alt_H[:,:,4] - H[:,:,4]
@test all( abs.(err3) .< 1e-10 )
@test all( abs.(err4) .< 1e-12 )
