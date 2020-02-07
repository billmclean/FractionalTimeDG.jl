import FractionalTimeDG.P, FractionalTimeDG.dP

α = 3/4
r = 4


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

function brute_force_H_uniform(ℓ::Integer, r::Integer, α::T, 
                       rtol=1e-8) where T <: AbstractFloat

    function inner_integral(ℓ::Integer, j::Integer, τ::T
                           ) where T <: AbstractFloat
        I, err = quadgk(-one(T), one(T), rtol=rtol) do σ
            Δ = (τ-σ) / ( 2ℓ )
            return (1+Δ)^(α-1) * P(j-1, σ) 
        end
        return I
    end

    Hℓ = Array{T}(undef, r, r)
    c = ℓ^(α-1) / ( 2Γ(α) )
    for j = 1:r
        Aj = inner_integral(ℓ, j, one(T))
        Bj = inner_integral(ℓ, j, -one(T))
        pow = one(T)
        for i = 1:r
            Cij, err = quadgk(-one(T), one(T), rtol=rtol) do τ
                dP(i-1,τ) * inner_integral(ℓ, j, τ)
            end
            pow = -pow
            Hℓ[i,j] = c * ( Aj + pow * Bj - Cij )
        end
    end
    return Hℓ
end

H0 = coef_H0(r, α)
bf_H0 = brute_force_H0(r, α)

err0 = bf_H0 - H0
@test all( abs.(err0) .< 1e-8 )

H1 = FractionalTimeDG.coef_H1_uniform(r, α, r+2)
bf_H1 = brute_force_H_uniform(1, r, α)
err1 = bf_H1 - H[:,:,1]
err1 = bf_H1 - H1
@test all( abs.(err1) .< 1e-8 )

version = 1
H = coef_H_uniform(5, r, α, r+2, version)

bf_H4 = brute_force_H_uniform(4, r, α)
err4 = bf_H4 - H[:,:,4]
@test all( abs.(err4) .< 1e-12 )

alt_H = Array{Float64}(undef, r, r, 4)
FractionalTimeDG.coef_H_uniform!(alt_H, 3:4, r, α, r+2)
err3 = alt_H[:,:,3] - H[:,:,3]
err4 = alt_H[:,:,4] - H[:,:,4]
@test all( abs.(err3) .< 1e-10 )
@test all( abs.(err4) .< 1e-12 )
