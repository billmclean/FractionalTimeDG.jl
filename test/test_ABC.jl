α = 3/4
r = 4

t = OffsetArray([0.0, 0.5, 1.0, 1.5], 0:3)

function brute_force_A(r::Integer, n::Integer, ℓ::Integer, α::T, 
                       t::OffsetArray{T}, rtol=1e-8) where T <: AbstractFloat
    A = Vector{T}(undef, r)
    for j = 1:r
        A[j], err = quadgk(-one(T), one(T), rtol=rtol) do σ
            ( 1 + Δ(n, ℓ, one(T), σ, t) )^(α-1) * P(j-1, σ)
        end
    end
    return A
end

function brute_force_B(r::Integer, n::Integer, ℓ::Integer, α::T, 
                       t::OffsetArray{T}, rtol=1e-8) where T <: AbstractFloat
    B = Vector{T}(undef, r)
    for j = 1:r
        B[j], err = quadgk(-one(T), one(T), rtol=rtol) do σ
            ( 1 + Δ(n, ℓ, -one(T), σ, t) )^(α-1) * P(j-1, σ)
        end
    end
    return B
end

function brute_force_C(r::Integer, n::Integer, ℓ::Integer, α::T, 
                       t::OffsetArray{T}, rtol=1e-8) where T <: AbstractFloat

    function inner(τ, j)
        I, err = quadgk(-one(T), one(T), rtol=rtol) do σ
            ( 1 + Δ(n, ℓ, τ, σ, t) )^(α-1) * P(j-1, σ)
        end
        return I
    end

    C = Matrix{T}(undef, r, r)
    for j = 1:r
       for i = 1:r
            C[i,j], err = quadgk(-one(T), one(T), rtol=rtol) do τ
                dP(i-1, τ) * inner(τ, j)
            end
        end
    end
    return C
end

function brute_force_C1(r::Integer, n::Integer, α::T, 
                       t::OffsetArray{T}, rtol=1e-8) where T <: AbstractFloat
    C1_first = Array{T}(undef, r, r)
    C1_second = Array{T}(undef, r, r)
    kn = t[n] - t[n-1]
    knm1 = t[n-1] - t[n-2]

    function inner_first(τ, j)
        I, err = quadgk(zero(T), one(T), rtol=rtol) do z
            ((kn+z*knm1)/(kn+knm1))^(α-1) * P(j-1, 1-z*(1+τ))
        end
        return I
    end

    function inner_second(σ, i)
        I, err = quadgk(zero(T), one(T), rtol=rtol) do z
            ((z*kn+knm1)/(kn+knm1))^(α-1) * dP(i-1, z*(1-σ)-1)
        end
        return I
    end

    for j = 1:r
        for i = 1:r
            C1_first[i,j], err = quadgk(-one(T), one(T), rtol=rtol) do τ
                (1+τ)^α * dP(i-1, τ) * inner_first(τ, j)
            end
            C1_second[i,j], err = quadgk(-one(T), one(T), rtol=rtol) do σ
                (1-σ)^α * P(j-1, σ) * inner_second(σ, i)
            end
        end
    end
    return C1_first, C1_second
end

n = 3
ℓ = 2

A1 = FractionalTimeDG.A_integral_uniform(r, n-ℓ, α, r+2)
bf_A1 = brute_force_A(r, n, ℓ, α, t)
err_A1 = A1 - bf_A1
@test all( abs.(err_A1) .< 1e-8 )

B1 = FractionalTimeDG.B1_integral_uniform(r, α)
bf_B1 = brute_force_B(r, n, ℓ, α, t)
err_B1 = B1 - bf_B1
@test all( abs.(err_B1) .< 1e-8 )

C1_first, C1_second = FractionalTimeDG.C1_integrals_uniform(r, α, r+2)
bf_C1_first, bf_C1_second = brute_force_C1(r, n, α, t)
err_C1_first = C1_first - bf_C1_first
err_C1_second = C1_second - bf_C1_second
@test all( abs.(err_C1_first) .< 1e-8 )
@test all( abs.(err_C1_second) .< 1e-8 )

C = C1_first + C1_second
bf_C = brute_force_C(r, n, ℓ, α, t)
err_C = C - bf_C
@test all( abs.(err_C) .< 1e-8 )

# Now test non-uniform time steps
t = OffsetArray([0.0, 0.3, 0.8, 1.4], 0:3)

A1 = FractionalTimeDG.A_integral(r, n, ℓ, α, t, r+2)
bf_A1 = brute_force_A(r, n, ℓ, α, t)
err_A1 = A1 - bf_A1
@test all( abs.(err_A1) .< 1e-8 )

B1 = FractionalTimeDG.B1_integral(r, n, α, t, r+2)
bf_B1 = brute_force_B(r, n, ℓ, α, t)
err_B1 = B1 - bf_B1
@test all( abs.(err_B1) .< 1e-8 )

C1_first, C1_second = FractionalTimeDG.C1_integrals(r, r, n, α, t, r+3)
bf_C1_first, bf_C1_second = brute_force_C1(r, n, α, t)
err_C1_first = C1_first - bf_C1_first
err_C1_second = C1_second - bf_C1_second
@test all( abs.(err_C1_first) .< 1e-8 )
@test all( abs.(err_C1_second) .< 1e-8 )

n = 3
ℓ = 1

bf_Cnℓ = brute_force_C(r, n, ℓ, α, t)
Cnℓ = FractionalTimeDG.C_integral(r, r, n, ℓ, α, t, r+3)
err_Cnℓ = bf_Cnℓ - Cnℓ
@test all( abs.(err_Cnℓ) .< 1e-8 )
