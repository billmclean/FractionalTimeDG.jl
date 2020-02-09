import FractionalTimeDG.P, FractionalTimeDG.dP

α = 3/4
r = 4
store = FractionalTimeDG.setup(α, r, 2r)

function brute_force_Hn(r::Integer, n::Integer, ℓbar::Integer, α::T,
                        t::OffsetVector{T},
                        rtol=1e-8) where T <: AbstractFloat

    function inner(τ::T, j::Integer)
        I, err = quadgk(-one(T), one(T), rtol=rtol) do σ
            return ( 1 + Δ(n, ℓbar, τ, σ, t) )^(α-1) * P(j-1, σ)
        end
        return I
    end

    Hnℓbar = Array{T}(undef, r, r)
    ℓ = n - ℓbar
    kℓ = t[ℓ] - t[ℓ-1]
    D = (t[n]+t[n-1])/2 - (t[ℓ]+t[ℓ-1])/2
    c = D^(α-1) * kℓ / ( 2Γ(α) )
    for j = 1:r
        Aj = inner( one(T), j)
        Bj = inner(-one(T), j)
        pow = one(T)
        for i = 1:r
            Cij, err = quadgk(-one(T), one(T), rtol=rtol) do τ
                dP(i-1,τ) * inner(τ, j)
            end
            pow = -pow
            Hnℓbar[i,j] = c * ( Aj + pow * Bj - Cij )
        end
    end
    return Hnℓbar
end

N = 4
n = 3

t = OffsetVector([0.0, 0.35, 0.8, 1.3, 2.0], 0:N)

Hn = coef_Hn!(r, n, n-1, t, r+2, store)

bf_H1 = brute_force_Hn(r, n, 1, α, t)
err1 = bf_H1 - Hn[1]
@test all( abs.(err1) .< 1e-8 )

bf_H2 = brute_force_Hn(r, n, 2, α, t)
err2 = bf_H1 - Hn[2]
@test all( abs.(err1) .< 1e-8 )
