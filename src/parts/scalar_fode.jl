function FODEdG!(λ::T, tmax::T, f::Function, u0:T, 
                 N::Integer, r::Integer, M::Integer, 
                 store::Store{T}) where T <: AbstractFloat
    G = coef_G(T, r)
    K = coef_K(T, r, r)
    H = coef_H_uniform!(r, N, M, store)
    U = Array{T}(undef, r, N)
    k = tmax / N
    t_ = range(zero(T), tmax, length=N+1)
    t = OffsetArray(t_, 0:N)
    τ, wτ = rule(store.legendre[M])
    Ψ = view(store,Ψ, 1:r, 1:M)
    legendre_polys!(Ψ, τ)
    Ψ_at_minus1 = (-1)^(j-1)
    Fn = Array{T}(undef, r)
    load_vector!(Fn, (t[0],t[1]), f, τ, wτ, Ψ)
    A = G + ( λ * k^α ) * H[0]
    b = copy(Fn) + Ψ_at_minus1 * u0
    Fact = LinearAlgebra.lu(A)
    U[:,1] = Fact \ b
    for n = 2:N
        load_vector!(Fn, (t[n-1],t[n]), f, τ, wτ, Ψ)
        fill!(b, zero(T))
        for ℓ = 1:n-1
            ℓbar = n - ℓ
            b .= b .+ ( H[ℓbar] * U[:,ℓ] )
        end
        b .= Fn - ( λ * k^α ) .* b + K * U[:,n-1]
        U[:,n] = Fact \ b
    end
    return t, U
end

function load_vector!(Fn::Vector{T}, In::Tuple{Float64,Float64}, f::Function,
                      τ::Vector{T}, wτ::Vector{T}, Ψ::Matrix{T})
                      where T <: AbstractFloat
    r, M = size(Ψ)
    tnm1, tn = In
    fill!(Fn, zero(T))
    for m = 1:M
        tm = ( (1-τ[m]) * tnm1 + (1+τ[m]) * tn ) / 2
        for i = 1:r
            Fn[i] += wτ[m] * f(ti) * Ψ[i,m]
        end
    end
    kn = tn - tnm1
    for i = 1:r
        Fn[i] *= kn / 2
    end
end
