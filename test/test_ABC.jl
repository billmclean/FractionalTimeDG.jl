import FractionalTimeDG.P, FractionalTimeDG.dP

α = 3/4
r = 4
store = FractionalTimeDG.Store(α, r, 2r)

function brute_force_A(r::Integer, n::Integer, ℓbar::Integer, α::T, 
                       t::OffsetArray{T}, rtol=1e-8) where T <: AbstractFloat
    A = Vector{T}(undef, r)
    for j = 1:r
        A[j], err = quadgk(-one(T), one(T), rtol=rtol) do σ
            ( 1 + Δ(n, ℓbar, one(T), σ, t) )^(α-1) * P(j-1, σ)
        end
    end
    return A
end

function brute_force_B(r::Integer, n::Integer, ℓbar::Integer, α::T, 
                       t::OffsetArray{T}, rtol=1e-8) where T <: AbstractFloat
    B = Vector{T}(undef, r)
    for j = 1:r
        B[j], err = quadgk(-one(T), one(T), rtol=rtol) do σ
            ( 1 + Δ(n, ℓbar, -one(T), σ, t) )^(α-1) * P(j-1, σ)
        end
    end
    return B
end

function brute_force_C(r::Integer, n::Integer, ℓbar::Integer, α::T, 
                       t::OffsetArray{T}, rtol=1e-8) where T <: AbstractFloat

    function inner(τ, j)
        I, err = quadgk(-one(T), one(T), rtol=rtol) do σ
            ( 1 + Δ(n, ℓbar, τ, σ, t) )^(α-1) * P(j-1, σ)
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

N = 3

H = OffsetArray{Matrix{Float64}}(undef, 0:N-1)
for ℓbar = 0:N-1
    H[ℓbar] = Array{Float64}(undef, r, r)
end

# Test uniform time steps
t = OffsetArray([0.0, 0.5, 1.0, 1.5], 0:3)

n = N
ℓbar = 1

FractionalTimeDG.coef_H1_uniform!(H[ℓbar], r+2, store)
#
A1_unif = copy(store.A[1:r])
B1_unif = copy(store.B[1:r])
C1_unif = copy(store.C[1:r,1:r])

bf_A1_unif = brute_force_A(r, n, ℓbar, α, t)
err_A1_unif = A1_unif - bf_A1_unif
@test all( abs.(err_A1_unif) .< 1e-8 )

bf_B1_unif = brute_force_B(r, n, ℓbar, α, t)
err_B1_unif = B1_unif - bf_B1_unif
@test all( abs.(err_B1_unif) .< 1e-8 )

bf_C1_unif = brute_force_C(r, n, ℓbar, α, t)
err_C1_unif = C1_unif - bf_C1_unif
@test all( abs.(err_C1_unif) .< 1e-8 )

ℓbar = 2

FractionalTimeDG.coef_H_uniform_ver1!(H, ℓbar:ℓbar, r+2, store)
A2_unif = copy(store.A[1:r])
B2_unif = copy(store.B[1:r])
C2_unif = copy(store.C[1:r,1:r])

bf_A2_unif = brute_force_A(r, n, ℓbar, α, t)
err_A2_unif = A2_unif - bf_A2_unif
@test all( abs.(err_A2_unif) .< 1e-8 )

bf_B2_unif = brute_force_B(r, n, ℓbar, α, t)
err_B2_unif = B2_unif - bf_B2_unif
@test all( abs.(err_B2_unif) .< 1e-8 )

bf_C2_unif = brute_force_C(r, n, ℓbar, α, t)
err_C2_unif = C2_unif - bf_C2_unif
@test all( abs.(err_C2_unif) .< 1e-8 )

# Test non-uniform time steps

t = OffsetArray([0.0, 0.3, 0.8, 1.4], 0:3)
Hn = copy(H)

ℓbar = 1

FractionalTimeDG.coef_Hn1!(Hn[ℓbar], n, t, r+3, store)
A1 = copy(store.A[1:r])
B1 = copy(store.B[1:r])
C1 = copy(store.C[1:r,1:r])

bf_A1 = brute_force_A(r, n, ℓbar, α, t)
err_A1 = A1 - bf_A1
@test all( abs.(err_A1) .< 1e-8 )

bf_B1 = brute_force_B(r, n, ℓbar, α, t)
err_B1 = B1 - bf_B1
@test all( abs.(err_B1) .< 1e-8 )

bf_C1 = brute_force_C(r, n, ℓbar, α, t)
err_C1 = C1 - bf_C1
@test all( abs.(err_C1) .< 1e-8 )

ℓbar = 2

FractionalTimeDG.coef_Hn_ver1!(Hn, n, ℓbar:ℓbar, t, r+2, store)
A2 = copy(store.A[1:r])
B2 = copy(store.B[1:r])
C2 = copy(store.C[1:r,1:r])

bf_A2 = brute_force_A(r, n, ℓbar, α, t)
err_A2 = A2 - bf_A2
@test all( abs.(err_A2) .< 1e-8 )

bf_B2 = brute_force_B(r, n, ℓbar, α, t)
err_B2 = B2 - bf_B2
@test all( abs.(err_B2) .< 1e-8 )
