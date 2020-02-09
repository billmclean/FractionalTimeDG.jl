import FractionalTimeDG.P, FractionalTimeDG.dP

α = 3/4
r = 4
store = FractionalTimeDG.setup(α, r, 2r)

function brute_force_H0(r::Integer, α::T, rtol=1e-8) where T <: AbstractFloat

    # \int_0^\tau (\tau-\sigma)^{\alpha-1} P_{j-1}(\tau)\,d\tau
    function inner_integral(j::Integer, τ::T) where T <: AbstractFloat
        I, err = quadgk(-one(T), τ, rtol=rtol) do σ
            (τ-σ)^(α-1) * P(j-1, σ) 
        end
        return I
    end

    H0 = Array{T}(undef, r, r)
    c = 1 / (2^α*Γ(α))
    for j = 1:r, i = 1:r
        I, err = quadgk(-one(T), one(T), rtol=rtol) do τ
            dP(i-1, τ) * inner_integral(j, τ)
        end
        H0[i,j] = c * ( inner_integral(j, one(T) ) - I )
    end
    return H0
end

function brute_force_H_uniform(ℓbar::Integer, r::Integer, α::T, 
                       rtol=1e-8) where T <: AbstractFloat

    function inner_integral(ℓbar::Integer, j::Integer, τ::T
                           ) where T <: AbstractFloat
        I, err = quadgk(-one(T), one(T), rtol=rtol) do σ
            Δ = (τ-σ) / ( 2ℓbar )
            return (1+Δ)^(α-1) * P(j-1, σ) 
        end
        return I
    end

    Hℓbar = Array{T}(undef, r, r)
    c = ℓbar^(α-1) / ( 2Γ(α) )
    for j = 1:r
        Aj = inner_integral(ℓbar, j, one(T))
        Bj = inner_integral(ℓbar, j, -one(T))
        pow = one(T)
        for i = 1:r
            Cij, err = quadgk(-one(T), one(T), rtol=rtol) do τ
                dP(i-1,τ) * inner_integral(ℓbar, j, τ)
            end
            pow = -pow
            Hℓbar[i,j] = c * ( Aj + pow * Bj - Cij )
        end
    end
    return Hℓbar
end

N = 4

H = coef_H_uniform!(r, N, r+2, store)

bf_H0 = brute_force_H0(r, α)

err0 = bf_H0 - H[0]
@test all( abs.(err0) .< 1e-8 )

bf_H2 = brute_force_H_uniform(2, r, α)
err2 = bf_H2- H[2]
@test all( abs.(err2) .< 1e-8 )

bf_H3 = brute_force_H_uniform(3, r, α)
err3 = bf_H3- H[3]
@test all( abs.(err3) .< 1e-8 )

