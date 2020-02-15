
function FODEdG!(λ::T, f::Function, u0::T, t::OffsetVector{T},
                 r::Integer, M::Integer, 
                 store::Store{T}) where T <: AbstractFloat
    @argcheck r ≤ store.rmax
    @argcheck M ≤ store.Mmax
    N = length(t) - 1
    U = Vector{Vector{T}}(undef, N)
    G = coef_G(T, r)
    K = coef_K(T, r, r)
    Hn = OffsetArray{Matrix{T}}(undef, 0:N-1)
    for n = 0:N-1
        Hn[n] = Array{T}(undef, r, r)
    end
    τ, wτ = rule(store.legendre[M])
    Ψ = view(store.Ψ, 1:r, 1:M)
    legendre_polys!(Ψ, τ)
    Fn = Array{T}(undef, r)
    load_vector!(Fn, (t[0],t[1]), f, τ, wτ, Ψ)
    Ψ_at_minus1 = Vector{T}(undef, r)
    legendre_polys!(Ψ_at_minus1, -one(T))
    coef_Hn!(Hn, 1, 0, t, M, store)
    A = G + λ * Hn[0]
    b = copy(Fn) + Ψ_at_minus1 * u0
    Fact = LinearAlgebra.lu(A)
    U[1] = Fact \ b
    for n = 2:N
        coef_Hn!(Hn, n, n-1, t, M, store)
        load_vector!(Fn, (t[n-1],t[n]), f, τ, wτ, Ψ)
        fill!(b, zero(T))
        for ℓ = 1:n-1
            b .= b .+ ( Hn[n-ℓ] * U[ℓ] )
        end
        b .= Fn - λ * b + K * U[n-1]
        A = G + λ * Hn[0]
        Fact = LinearAlgebra.lu(A)
        U[n] = Fact \ b
    end
    return U
end

function FODEdG!(λ::T, tmax::T, f::Function, u0::T, 
                 N::Integer, r::Integer, M::Integer, 
                 store::Store{T}) where T <: AbstractFloat
    @argcheck r ≤ store.rmax
    @argcheck M ≤ store.Mmax
    G = coef_G(T, r)
    K = coef_K(T, r, r)
    H = coef_H_uniform!(r, N, M, store)
    U = Vector{Vector{T}}(undef, N)
    k = tmax / N
    α = store.α
    t_ = range(zero(T), tmax, length=N+1)
    t = OffsetArray(t_, 0:N)
    τ, wτ = rule(store.legendre[M])
    Ψ = view(store.Ψ, 1:r, 1:M)
    legendre_polys!(Ψ, τ)
    Fn = Array{T}(undef, r)
    load_vector!(Fn, (t[0],t[1]), f, τ, wτ, Ψ)
    Ψ_at_minus1 = Vector{T}(undef, r)
    legendre_polys!(Ψ_at_minus1, -one(T))
    A = G + ( λ * k^α ) * H[0]
    b = copy(Fn) + Ψ_at_minus1 * u0
    Fact = LinearAlgebra.lu(A)
    U[1] = Fact \ b
    for n = 2:N
        load_vector!(Fn, (t[n-1],t[n]), f, τ, wτ, Ψ)
        fill!(b, zero(T))
        for ℓ = 1:n-1
            b .= b .+ ( H[n-ℓbar] * U[ℓ] )
        end
        b .= Fn - ( λ * k^α ) .* b + K * U[n-1]
        U[n] = Fact \ b
    end
    return t, U
end

function load_vector!(Fn::Vector{T}, In::Tuple{T,T}, f::Function,
                      τ::Vector{T}, wτ::Vector{T}, Ψ::AbstractMatrix{T}
                     ) where T <: AbstractFloat
    r, M = size(Ψ)
    tnm1, tn = In
    fill!(Fn, zero(T))
    for m = 1:M
        tm = ( (1-τ[m]) * tnm1 + (1+τ[m]) * tn ) / 2
        for i = 1:r
            Fn[i] += wτ[m] * f(tm) * Ψ[i,m]
        end
    end
    kn = tn - tnm1
    for i = 1:r
        Fn[i] *= kn / 2
    end
end
