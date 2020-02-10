function Store(α::T, rmax::Integer, Mmax::Integer
              ) where T <: AbstractFloat
    if Mmax < rmax
        throw(ArgumentError("Mmax must be at least as big as rmax"))
    end
    unitlegendre = Vector{Matrix{T}}(undef, Mmax)
    legendre = Vector{Matrix{T}}(undef, Mmax)
    jacobi1 = Vector{Matrix{T}}(undef, Mmax)
    jacobi2 = Vector{Matrix{T}}(undef, Mmax)
    jacobi3 = Vector{Matrix{T}}(undef, Mmax)
    jacobi4 = Vector{Matrix{T}}(undef, Mmax)
    jacobi5 = Vector{Matrix{T}}(undef, Mmax)
    for M = 1:Mmax
        unitlegendre[M] = Array{T}(undef, M, 2)
        legendre[M] = Array{T}(undef, M, 2)
        x, w = GaussQuadrature.legendre(T, M)
        legendre[M][:,1], legendre[M][:,2] = x, w
        unitlegendre[M][:,1] = ( x .+ 1 ) / 2
        unitlegendre[M][:,2] = w / 2
        jacobi1[M] = Array{T}(undef, M, 2)
        jacobi1[M][:,1], jacobi1[M][:,2] = GaussQuadrature.jacobi(M, α-1, 
                                                                  zero(T))
        jacobi2[M] = Array{T}(undef, M, 2)
        jacobi2[M][:,1], jacobi2[M][:,2] = GaussQuadrature.jacobi(M, zero(T), 
                                                                  α-1)
        jacobi3[M] = Array{T}(undef, M, 2)
        jacobi3[M][:,1], jacobi3[M][:,2] = GaussQuadrature.jacobi(M, α, zero(T))
        jacobi4[M] = Array{T}(undef, M, 2)
        jacobi4[M][:,1], jacobi4[M][:,2] = GaussQuadrature.jacobi(M, zero(T), α)
        jacobi5[M] = Array{T}(undef, M, 2)
        jacobi5[M][:,1], jacobi5[M][:,2] = GaussQuadrature.jacobi(M, one(T), 
                                                                  α-1)
    end
    Ψ  = Array{T}(undef, rmax, Mmax)
    dΨ = Array{T}(undef, rmax, Mmax)
    A = Vector{T}(undef, rmax)
    B = Vector{T}(undef, rmax)
    C = Array{T}(undef, rmax, rmax)
    return Store(α, rmax, Mmax, unitlegendre, legendre, 
                 jacobi1, jacobi2, jacobi3, jacobi4, jacobi5,
                 Ψ, dΨ, A, B, C)
end

function rule(storerule::Matrix{T}) where T <: AbstractFloat
    return storerule[:,1], storerule[:,2]
end

function rules(storerules::Vector{Matrix{T}}) where T <: AbstractFloat
    M = length(storerules)
    pt = Vector{Vector{T}}(undef, M)
    wt = Vector{Vector{T}}(undef, M)
    for m = 1:M
        pt[m] = view(storerules[m], :, 1)
        wt[m] = view(storerules[m], :, 2)
    end
    return pt, wt
end
