import FractionalTimeDG.P, FractionalTimeDG.dP

α = 3/4
r = 4
store = FractionalTimeDG.setup(α, r, 2r)

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
ℓ = n - 1

t = OffsetArray([0.0, 0.5, 1.0, 1.5], 0:3)

FractionalTimeDG.A_integral_uniform!(r, n-ℓ, r+2, store)
A1_unif = copy(store.A[1:r])
bf_A1_unif = brute_force_A(r, n, ℓ, α, t)
err_A1_unif = A1_unif - bf_A1_unif
@test all( abs.(err_A1_unif) .< 1e-8 )

FractionalTimeDG.B1_integral_uniform!(r, store)
B1_unif = copy(store.B[1:r])
bf_B1_unif = brute_force_B(r, n, ℓ, α, t)
err_B1_unif = B1_unif - bf_B1_unif
@test all( abs.(err_B1_unif) .< 1e-8 )

C1_first_unif, C1_second_unif = FractionalTimeDG.C1_integrals_uniform(r, α, r+2)
bf_C1_first_unif, bf_C1_second_unif = brute_force_C1(r, n, α, t)
err_C1_first_unif = C1_first_unif - bf_C1_first_unif
err_C1_second_unif = C1_second_unif - bf_C1_second_unif
@test all( abs.(err_C1_first_unif) .< 1e-8 )
@test all( abs.(err_C1_second_unif) .< 1e-8 )

FractionalTimeDG.C1_integral_uniform!(r, r+2, store)
C1_unif = copy(store.C[1:r,1:r])
bf_C1_unif = brute_force_C(r, n, ℓ, α, t)
err_C1_unif = C1_unif - bf_C1_unif
@test all( abs.(err_C1_unif) .< 1e-8 )

ℓ = n - 2

FractionalTimeDG.C_integral_uniform!(r, n-ℓ, r+2, store)
Cnℓ_unif = copy(store.C[1:r,1:r])
bf_Cnℓ_unif = brute_force_C(r, n, ℓ, α, t)
err_Cnℓ_unif = Cnℓ_unif - bf_Cnℓ_unif
@test all( abs.(err_Cnℓ_unif) .< 1e-8 )

# Now test non-uniform time steps
t = OffsetArray([0.0, 0.3, 0.8, 1.4], 0:3)

ℓ = n - 1

FractionalTimeDG.A_integral!(r, n, ℓ, t, r+2, store)
A1 = copy(store.A[1:r])
bf_A1 = brute_force_A(r, n, ℓ, α, t)
err_A1 = A1 - bf_A1
@test all( abs.(err_A1) .< 1e-8 )

FractionalTimeDG.B1_integral!(r, n, t, store)
B1 = copy(store.B[1:r])
bf_B1 = brute_force_B(r, n, ℓ, α, t)
err_B1 = B1 - bf_B1
@test all( abs.(err_B1) .< 1e-8 )

FractionalTimeDG.C1_integral!(r, r, n, t, r+3, store)
C1 = copy(store.C[1:r,1:r])
bf_C1_first, bf_C1_second = brute_force_C1(r, n, α, t)
bf_C1 = bf_C1_first + bf_C1_second
err_C1 = C1 - bf_C1
@test all( abs.(err_C1) .< 1e-8 )

ℓ = n - 2

bf_Cnℓ = brute_force_C(r, n, ℓ, α, t)
FractionalTimeDG.C_integral!(r, r, n, ℓ, t, r+3, store)
Cnℓ = copy(store.C[1:r,1:r])
err_Cnℓ = bf_Cnℓ - Cnℓ
@test all( abs.(err_Cnℓ) .< 1e-8 )
