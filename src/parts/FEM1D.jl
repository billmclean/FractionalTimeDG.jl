struct SpatialGrid{T <: AbstractFloat }
    x          :: OffsetVector{T}
    global_DoF :: Vector{Vector{Int64}}
    nfree      :: Int64
    nfixed     :: Int64
    max_r      :: Int64
    pt         :: Vector{T} # Quadrature points on (-1,1)
    wt         :: Vector{T} # Quadrature weights
end

"""
   Φ = shape_funcs(r, ξ)

Returns shape functions `Φ[i,j] = Φ_(i-1)(ξ[j])`.
"""
function shape_funcs(r::Int64, ξ::AbstractVector{T}
                    ) where { T <: AbstractFloat }
    J = length(ξ)
    Φ = zeros(T, r, J)
    for j = 1:J
        Φ[1,j] = ( 1 - ξ[j] ) / 2
        Φ[2,j] = ( 1 + ξ[j] ) / 2
    end
    P = Array{T}(undef, r, J)
    legendre_polys!(P, ξ) # P[i,j] = P_(i-1)(ξ[j])
    for j = 1:J, i = 3:r
        a = convert(T, 2(2i-3))
        Φ[i,j] = ( P[i,j] - P[i-2,j] ) / sqrt(a)
    end
    return Φ
end

function SpatialGrid(x::OffsetVector{T}, r::Vector{Int64},
                     bc::Tuple{String,String},
                     num_x_Gauss_pts::Int64) where { T <: AbstractFloat }
    P = length(x) - 1
    @argcheck length(r) == P DimensionMismatch
    global_DoF = Vector{Vector{Integer}}(undef, P)
    total =  0
    max_r = 0
    for p = 1:P
        @argcheck r[p] ≥ 2
        global_DoF[p] = Array{Integer}(undef, r[p])
        total += r[p]
        if r[p] ≥ max_r
            max_r = r[p]
        end
    end
    total -= P-1
    if bc == ("essential", "essential")
        nfixed = 2
        nfree = total - nfixed
        global_DoF[1][1] = nfree + 1
        m = 0
        if P > 1
            for j = 2:r[1]
                m += 1
                global_DoF[1][j] = m
            end
            for p = 2:P-1
                global_DoF[p][1] = global_DoF[p-1][2]
                for j = 2:r[p]
                    m += 1
                    global_DoF[p][j] = m
                end
            end
            global_DoF[P][1] = global_DoF[P-1][2]
            global_DoF[P][2] = nfree + 2
            for j = 3:r[P]
                m += 1
                global_DoF[P][j] = m
            end
        else
            global_DoF[1][2] = nfree + 2
            for j = 3:r[1]
                global_DoF[1][j] = j-2
            end
        end
    elseif bc == ("essential", "natural")
        nfixed = 1
        nfree = total - nfixed
        global_DoF[1][1] = nfree + 1
        m = 0
        for j = 2:r[1]
            m += 1
            global_DoF[1][j] = m
        end
        for p = 2:P
            global_DoF[p][1] = global_DoF[p-1][2]
            for j = 2:r[p]
                m += 1
                global_DoF[p][j] = m
            end
        end
    elseif bc == ("natural", "essential")
        nfixed = 1
        nfree = total - nfixed
        m = 0
        for j = 1:r[1]
            m += 1
            global_DoF[1][j] = m
        end
        for p = 2:P-1
            global_DoF[p][1] = global_DoF[p-1][2]
            for j = 2:r[p]
                m += 1
                global_DoF[p][j] = m
            end
        end
        global_DoF[P][1] = global_DoF[P-1][2]
        global_DoF[P][2] = nfree + 1
        for j = 3:r[P]
            m += 1
            global_DoF[P][j] = m
        end
    elseif bc == ("natural", "natural")
        nfixed = 0
        nfree = total - nfixed
        m = 0
        for j = 1:r[1]
            m += 1
            global_DoF[1][j] = m
        end
        for p = 2:P
            global_DoF[p][1] = global_DoF[p-1][2]
            for j = 2:r[p]
                m += 1
                global_DoF[p][j] = m
            end
        end
    else
        throw(ArgumentError("unrecognised boundary conditions"))
    end
    pt, wt = GaussQuadrature.legendre(T, num_x_Gauss_pts)
    return SpatialGrid{T}(x, global_DoF, nfree, nfixed, max_r, pt, wt)
end

function SpatialGrid(x::OffsetVector{T}, r::Int64,
                     bc::Tuple{String,String},
                     num_Gauss_pts) where { T <: AbstractFloat }
    P = length(x) - 1
    vecr = fill(r, P)
    return SpatialGrid(x, vecr, bc, num_Gauss_pts)
end

function ref_element_matrices!(M::Matrix{T}, S::Matrix{T},
                               κ::T) where { T <: AbstractFloat }
    sz = size(M, 1)
    @argcheck size(M, 2) == sz 
    @argcheck size(S) == (sz, sz) 
    @argcheck sz >= 2
    fill!(M, zero(T))
    fill!(S, zero(T))
    two = convert(T, 2)
    three = convert(T, 3)
    M[1, 1] = M[2, 2] =  2 / three
    M[2, 1] = M[1, 2] =  1 / three
    S[1, 1] = S[2, 2] =  κ / 2
    S[2, 1] = S[1, 2] = -κ / 2
    if sz == 2
        return
    end
    six = convert(T, 6)
    if sz >= 3
        M[1, 3] = M[2, 3] = -1 /sqrt(six) 
        M[3, 1] = M[1, 3]
        M[3, 2] = M[2, 3]
    end
    if sz >= 4
        ten = convert(T, 10)
        M[1, 4] = 1 / ( 3 * sqrt(ten) ) 
        M[2, 4] = -M[1, 4]            
        M[4, 1] = M[1, 4]
        M[4, 2] = M[2, 4]
    end
    for j = 3:sz
        M[j, j] = two / ((2j-5)*(2j-1))
        S[j, j] = κ
    end
    for j = 3:sz-2
        a = convert(T, (2j-3)*(2j+1))
        M[j, j+2] = M[j+2, j] = -1 / ( (2j-1) * sqrt(a) )
    end
end

function assembled_matrices(grid::SpatialGrid, 
                            κ::T) where { T <: AbstractFloat }
    x, gdof = grid.x, grid.global_DoF
    nfree, nfixed, max_r = grid.nfree, grid.nfixed, grid.max_r
    P = length(gdof)
    Mref = zeros(T, max_r, max_r)
    Sref = zeros(T, max_r, max_r)
    ref_element_matrices!(Mref, Sref, κ)
    IM, JM, VM = Int64[], Int64[], Float64[]
    IS, JS, VS = Int64[], Int64[], Float64[]
    for p = 1:P
        hp = x[p] - x[p-1]
        rp = length(gdof[p])
        for i = 1:rp
            gi = gdof[p][i]
            if gi ≤ nfree
                for j = 1:rp
                    gj = gdof[p][j]
                    push!(IM, gi)
                    push!(JM, gj)
                    push!(VM, (hp/2)*Mref[i,j])
                    push!(IS, gi)
                    push!(JS, gj)
                    push!(VS, (2/hp)*Sref[i,j])
                end
            end
        end
    end
    M = SparseArrays.sparse(IM, JM, VM, nfree, nfree+nfixed)
    S = SparseArrays.sparse(IS, JS, VS, nfree, nfree+nfixed)
    return M, S
end

function element_load_vector!(Fp::AbstractArray{T}, 
                              elt::Tuple{T,T}, f::Function, 
                              Φ::Matrix{T}, pt::Vector{T}, wt::Vector{T}
                             ) where { T <: AbstractFloat }
    r = length(Fp)
    J = length(pt)
    @argcheck size(Φ,1) ≤ r  DimensionMismatch
    @argcheck size(Φ,2) == J DimensionMismatch
    Jacobian = ( elt[2] - elt[1] ) / 2
    fill!(Fp, zero(T))
    for j = 1:J
        xj = ( ( 1 - pt[j] ) * elt[1] + ( 1 + pt[j] ) * elt[2] ) / 2
        for i = 1:r
            Fp[i] += wt[j] * f(xj) * Φ[i,j] * Jacobian
        end
    end
end

function assembled_load_vector!(F::Vector{T}, grid::SpatialGrid{T}, 
                                f::Function) where { T <: AbstractFloat }
    x, gdof, pt, wt = grid.x, grid.global_DoF, grid.pt, grid.wt
    nfree, nfixed, max_r = grid.nfree, grid.nfixed, grid.max_r
    @argcheck length(F) == nfree
    P = length(x) - 1
    Φ = shape_funcs(max_r, pt)
    Fp = zeros(T, max_r)
    fill!(F, zero(T))
    for p = 1:P
        rp = length(gdof[p])
        element_load_vector!(view(Fp, 1:rp), (x[p-1],x[p]), f, Φ, pt, wt)
        for i = 1:length(gdof[p])
            gi = gdof[p][i]
            if gi ≤ nfree
                F[gi] += Fp[i]
            end
        end
    end
end

function assembled_load_vector(grid::SpatialGrid{T}, 
                               f::Function) where { T <: AbstractFloat }
    nfree = grid.nfree
    F = zeros(T, nfree)
    assembled_load_vector!(F, grid, f)
    return F
end

function spatial_points!(xvals::Vector{T}, grid::SpatialGrid{T}, 
                         ξ::AbstractVector{T}) where { T <: AbstractFloat }
    x, gdof = grid.x, grid.global_DoF
    Nx = length(xvals)
    P = length(x) - 1
    pts_per_interval = length(ξ)
    n = 0
    if Nx == 1 + P * ( pts_per_interval - 1 )
        for p = 1:P
            for j = 1:pts_per_interval - 1
                xpj = ( ( 1 - ξ[j] ) * x[p-1] + (1 + ξ[j] ) * x[p] ) / 2
                n += 1
                xvals[n] = xpj
            end
        end
        n += 1
        xvals[n] = x[P]
    elseif Nx == P * pts_per_interval
        for p = 1:P
            for j = 1:pts_per_interval 
                xpj = ( ( 1 - ξ[j] ) * x[p-1] + (1 + ξ[j] ) * x[p] ) / 2
                n += 1
                xvals[n] = xpj
            end
        end
    else
        throw(ArgumentError("xvals length does not match grid"))
    end
end

function evaluate_fem1d_soln!(Uvals::Vector{T}, U::Vector{T}, 
                              grid::SpatialGrid{T}, 
                              ξ::AbstractVector{T}
                             ) where { T <: AbstractFloat }
    x, gdof = grid.x, grid.global_DoF
    nfree, nfixed, max_r = grid.nfree, grid.nfixed, grid.max_r
    @argcheck length(U) == nfree DimensionMismatch
    Nx = length(Uvals)
    P = length(x) - 1
    pts_per_interval = length(ξ)
    Φ = shape_funcs(max_r, ξ)
    n = 0
    if Nx == 1 + P * ( pts_per_interval - 1 ) # ξ[1] = -1, ξ[end] = +1
        for p = 1:P
            rp = length(gdof[p])
            for j = 1:pts_per_interval-1
                n += 1
                Uvals[n] = zero(T)
                for i = 1:rp
                    gpi = gdof[p][i]
                    if gpi ≤ nfree
                        Uvals[n] += U[gpi] * Φ[i,j]
                    end
                end
            end
        end
        n += 1
        gpi = gdof[P][2]
        if gpi ≤ nfree
            Uvals[n] = U[gpi] 
        end
    elseif Nx == P * pts_per_interval  # ξ[1] > -1 or ξ[end] < +1
        for p = 1:P
            rp = length(gdof[p])
            for j = 1:pts_per_interval
                n += 1
                Uvals[n] = zero(T)
                for i = 1:rp
                    gpi = gdof[p][i]
                    if gpi ≤ nfree
                        Uvals[n] += U[gpi] * Φ[i,j]
                    end
                end
            end
        end
    else
        throw(ArgumentError("Uvals length does not match grid"))
    end
end

function evaluate_fem1d_soln(U::Vector{T}, 
                             grid::SpatialGrid{T}, 
                             pts_per_interval::Int64
                            ) where { T <: AbstractFloat }
    x, nfree = grid.x, grid.nfree
    @argcheck length(U) == nfree DimensionMismatch
    P = ubound(x, 1)
    Nx = 1 + P * ( pts_per_interval - 1 )
    ξ = collect( range(-one(T), stop=one(T), length=pts_per_interval) )
    xvals = zeros(T, Nx)
    spatial_points!(xvals, grid, ξ)
    Uvals = zeros(T, Nx)
    evaluate_fem1d_soln!(Uvals, U, grid, ξ)
    return xvals, Uvals
end

function evaluate_pcwise_poly(U::Array{T}, t::OffsetVector{T}, 
                              grid::SpatialGrid{T},
                              pts_per_time_interval::Int64,
                              pts_per_space_interval::Int64
                             ) where { T <: AbstractFloat }
    τ = range(-one(T), stop=one(T), length=pts_per_time_interval)
    ξ = range(-one(T), stop=one(T), length=pts_per_space_interval)
    xvals, pcwise_t, pcwise_U = evaluate_pcwise_poly(U, t, grid, τ, ξ)
    return xvals, pcwise_t, pcwise_U
end

function evaluate_pcwise_poly(U::Array{T}, t::OffsetVector{T}, 
                              grid::SpatialGrid{T},
                              τ::AbstractVector{T}, 
                              ξ::AbstractVector{T}
                             ) where { T <: AbstractFloat }
    x = grid.x
    P = length(x) - 1
    nfree, rt, N = size(U)
    pts_per_space_interval = length(ξ)
    pts_per_time_interval = length(τ)
    @argcheck axes(t, 1) == 0:N
    if ξ[1] > -1.0 || ξ[end] < 1.0
        Nx = P * pts_per_space_interval
    else
        Nx = 1 + P * ( pts_per_space_interval - 1 )
    end
    xvals = zeros(T, Nx)
    spatial_points!(xvals, grid, ξ)
    Ψ = Array{Float64}(undef, rt, pts_per_time_interval)
    legendre_polys!(Ψ, τ)
    pcwise_t = zeros(T, pts_per_time_interval, N)
    pcwise_U = zeros(T, Nx, pts_per_time_interval, N)
    Uvals = zeros(T, Nx)
    for n = 1:N
        for i = 1:pts_per_time_interval
            pcwise_t[i,n] = ( (1-τ[i])*t[n-1] + (1+τ[i])*t[n] ) / 2
            for k = 1:rt
                evaluate_fem1d_soln!(Uvals, U[:,k,n], grid, ξ)
                pcwise_U[:,i,n] .+= Uvals * Ψ[k,i]
            end
        end
    end
    return xvals, pcwise_t, pcwise_U
end

function evaluate_pcwise_error(pcwise_U::Array{T}, u::Function,
                               pcwise_t::Array{T}, xvals::Vector{T}
                              ) where { T <: AbstractFloat }
    Nx = length(xvals)
    N = size(pcwise_t, 2)
    pts_per_time_interval = size(pcwise_t, 1)
    pcwise_err = zeros(Nx, pts_per_time_interval, N)
    for n = 1:N
        for i = 1:pts_per_time_interval
            for p = 1:Nx
                pcwise_err[p,i,n] = ( pcwise_U[p,i,n] 
                                     - u(xvals[p], pcwise_t[i,n]) )
            end
        end
    end
    return pcwise_err
end

function dynamic_load_vector!(Fn::Matrix{T}, In::Vector{Float64},
                              grid::SpatialGrid{T}, f::Function, 
                              τ::Vector{T}, w::Vector{T}, 
                              Ψ::AbstractMatrix{T}) where { T <: AbstractFloat }
    nfree, rt = size(Fn)
    @argcheck grid.nfree == nfree
    @argcheck length(In) == 2
    fill!(Fn, 0.0)
    Fni = zeros(nfree)
    Δt = In[2] - In[1]
    for i = 1:length(τ)
        ti = ( ( 1 - τ[i] ) * In[1] + ( 1 + τ[i] ) * In[2] ) / 2
        assembled_load_vector!(Fni, grid, x -> f(x,ti))
        for j = 1:rt
            Fn[:,j] += (Δt/2) * w[i] * Fni * Ψ[j,i]
        end
    end
end

function PDEDG(κ::Float64, f::Function, U0::Vector{Float64}, 
               grid::SpatialGrid{Float64}, t::OffsetVector{Float64},
               rt::Int64, Mt::Integer) where T <: AbstractFloat
    x, gdof = grid.x, grid.global_DoF
    nfree, nfixed = grid.nfree, grid.nfixed

    G = coef_G(Float64, rt)
    K = coef_K(Float64, rt, rt)
    H = zeros(rt, rt)
    for j = 1:rt
        H[j,j] = 1 / (2j-1)
    end

    N = length(t) - 1
    U = zeros(nfree*rt, N)
    Fn = zeros(nfree, rt)
    M, S = assembled_matrices(grid, κ)
    M = M[1:nfree,1:nfree]
    S = S[1:nfree,1:nfree]
    Ψ_at_minus1 = Vector{Float64}(undef, rt)
    legendre_polys!(Ψ_at_minus1, -1.0)
    τ, w = GaussQuadrature.legendre(Mt)
    Ψ = Array{Float64}(undef, rt, Mt)
    legendre_polys!(Ψ, τ)
    dynamic_load_vector!(Fn, t[0:1], grid, f, τ, w, Ψ)
    b = Fn[1:nfree*rt] + kron(Ψ_at_minus1, M*U0) 
    Δt1 = t[1] - t[0]
    G_kron_M = kron(G, M)
    H_kron_S = kron(H, S)
    A = G_kron_M + Δt1 * H_kron_S
    U[:,1] = A \ b
    K_kron_M = kron(K, M)
    for n = 2:N
        Δtn = t[n] - t[n-1]
        dynamic_load_vector!(Fn, t[n-1:n], grid, f, τ, w, Ψ)
        b = Fn[1:nfree*rt] + K_kron_M * U[:,n-1]
        A = G_kron_M + Δtn * H_kron_S
        U[:,n] = A \ b
    end
    return reshape(U, nfree, rt, N)
end

"""
    t, U = PDEDG(κ, f, U0, grid, N, rt, num_Gauss_pts)

Faster version for uniform time steps.
"""
function PDEDG(κ::Float64, f::Function, U0::Vector{Float64}, 
               grid::SpatialGrid{Float64}, max_t::Float64, N::Int64, 
               rt::Int64, Mt::Integer)
    x, gdof = grid.x, grid.global_DoF
    nfree, nfixed = grid.nfree, grid.nfixed

    G = coef_G(Float64, rt)
    K = coef_K(Float64, rt, rt)
    H = zeros(rt, rt)
    for j = 1:rt
        H[j,j] = 1 / (2j-1)
    end

    t_ = collect(range(0, max_t, length=N+1))
    t = OffsetVector(t_, 0:N)
    U = zeros(nfree*rt, N)
    Fn = zeros(nfree, rt)
    M, S = assembled_matrices(grid, κ)
    M = M[1:nfree,1:nfree]
    S = S[1:nfree,1:nfree]
    Ψ_at_minus1 = Vector{Float64}(undef, rt)
    legendre_polys!(Ψ_at_minus1, -1.0)
    τ, w = GaussQuadrature.legendre(Mt)
    Ψ = Array{Float64}(undef, rt, Mt)
    legendre_polys!(Ψ, τ)
    dynamic_load_vector!(Fn, t[0:1], grid, f, τ, w, Ψ)
    b = Fn[1:nfree*rt] + kron(Ψ_at_minus1, M*U0) 
    Δt = max_t / N
    G_kron_M = kron(G, M)
    H_kron_S = kron(H, S)
    A = G_kron_M + Δt * H_kron_S
    LU = SparseArrays.lu(A)
    U[:,1] = LU \ b
    K_kron_M = kron(K, M)
    for n = 2:N
        dynamic_load_vector!(Fn, t[n-1:n], grid, f, τ, w, Ψ)
        b = Fn[1:nfree*rt] + K_kron_M * U[:,n-1]
        U[:,n] = LU \ b
    end
    return t, reshape(U, nfree, rt, N)
end

function spatial_L2_projection(f::Function, grid::SpatialGrid{Float64})
    M, S = assembled_matrices(grid, 0.0)
    nfree = grid.nfree
    Pf = M[1:nfree,1:nfree] \ assembled_load_vector(grid, f)
    return Pf
end

"""
    err_τ = maxerr(τ, U, t, u, ξ, grid, cutoff=0.0)

Returns err_τ[i] = max |U(x,t)-u(x,t)| over x values in each spatial
interval corresponding to ξ the values in [-1,1], and over the t value 
in each time subinterval corresponding to τ[i] in [-1,1], except for time 
intervals with t[n] ≤ cutoff.
"""
function maxerr(τ:: AbstractVector{Float64}, U::Array{Float64}, 
                t::OffsetVector{Float64}, u::Function,
                ξ::AbstractVector{Float64}, grid::SpatialGrid, 
                cutoff=0.0)
    xvals, pcwise_t, pcwise_U = evaluate_pcwise_poly(U, t, grid, τ, ξ)
    pcwise_err = evaluate_pcwise_error(pcwise_U, u, pcwise_t, xvals)
    N = size(pcwise_t, 2)
    pts_per_time_interval = size(pcwise_t, 1)
    Nx = size(pcwise_err, 1)
    err_τ = zeros(pts_per_time_interval)
    for n = 1:N
        if t[n] ≤ cutoff
            continue
        end
        for i = 1:pts_per_time_interval
            for p = 1:Nx
                err_τ[i] = max(abs(pcwise_err[p,i,n]), err_τ[i])
            end
        end
    end
    return err_τ
end

"""
    U = FPDEDG(κ, f, U0, grid, t, rt, num_t_Gauss_pts)

DG solver for fractional PDE.
"""
function FPDEDG(κ::Float64, f::Function, U0::Vector{Float64}, 
               grid::SpatialGrid{Float64}, t::OffsetVector{Float64},
               rt::Int64, Mt::Int64, store::Store{Float64})
    α = store.α
    x, gdof = grid.x, grid.global_DoF
    nfree, nfixed = grid.nfree, grid.nfixed

    G = coef_G(Float64, rt)
    K = coef_K(Float64, rt, rt)
    Nt = length(t) - 1
    Hn = OffsetArray{Matrix{Float64}}(undef, 0:Nt-1)
    for n = 0:Nt-1
        Hn[n] = Array{Float64}(undef, rt, rt)
    end
    τ, wτ = rule(store.legendre[Mt])
    Ψ = view(store.Ψ, 1:rt, 1:Mt)
    legendre_polys!(Ψ, τ)

    U = zeros(nfree*rt, Nt)
    Fn = zeros(nfree, rt)
    M, S = assembled_matrices(grid, κ)
    M = M[1:nfree,1:nfree]
    S = S[1:nfree,1:nfree]
    Ψ_at_minus1 = Array{Float64}(undef, rt)
    legendre_polys!(Ψ_at_minus1, -1.0)
    dynamic_load_vector!(Fn, t[0:1], grid, f, τ, wτ, Ψ)
    b = Fn[1:nfree*rt] + kron(Ψ_at_minus1, M*U0) 
    coef_Hn!(Hn, 1, 0, t, Mt, store)
    G_kron_M = kron(G, M)
    Hnn_kron_S = kron(Hn[0], S)
    A = G_kron_M + Hnn_kron_S
    U[:,1] = A \ b
    K_kron_M = kron(K, M)
    for n = 2:Nt
        coef_Hn!(Hn, n, n-1, t, Mt, store)
        Hnn_kron_S = kron(Hn[0], S)
        A = G_kron_M + Hnn_kron_S
        fill!(b, 0.0)
        for l = 1:n-1
            Hnl_kron_S = kron(Hn[n-l], S)
            b .= b .+ Hnl_kron_S * U[:,l]
        end
        dynamic_load_vector!(Fn, t[n-1:n], grid, f, τ, wτ, Ψ)
        b .= Fn[1:nfree*rt] .+ K_kron_M * U[:,n-1] .- b
        U[:,n] = A \ b
    end
    return reshape(U, nfree, rt, Nt)
end

"""
    t, U = FPDEDG(κ, f, U0, grid, max_t, N, rt, num_t_Gauss_pts)

Specialised version for uniform time steps.
"""
function FPDEDG(κ::Float64, f::Function, U0::Vector{Float64}, 
                grid::SpatialGrid{Float64}, max_t::Float64, 
                N::Int64, rt::Int64, Mt::Int64, store::Store{Float64})
    α = store.α
    x, gdof = grid.x, grid.global_DoF
    nfree, nfixed = grid.nfree, grid.nfixed

    Δt = max_t / N
    G = coef_G(Float64, rt)
    K = coef_K(Float64, rt, rt)
    H = coef_H_uniform!(rt, N, Mt, store)

    t_ = collect(range(0, max_t, length=N+1))
    t = OffsetVector(t_, 0:N)

    U = zeros(nfree*rt, N)
    Fn = zeros(nfree, rt)
    M, S = assembled_matrices(grid, κ)
    M = M[1:nfree,1:nfree]
    S = S[1:nfree,1:nfree]
    Ψ_at_minus1 = Vector{Float64}(undef, rt)
    legendre_polys!(Ψ_at_minus1, -1.0)
    τ, w = GaussQuadrature.legendre(Mt)
    Ψ = Array{Float64}(undef, rt, Mt)
    legendre_polys!(Ψ, τ)
    dynamic_load_vector!(Fn, t[0:1], grid, f, τ, w, Ψ)
    b = Fn[1:nfree*rt] + kron(Ψ_at_minus1, M*U0) 
    G_kron_M = kron(G, M)
    Hnn_kron_S = kron(H[0], S)
    A = G_kron_M + Δt^α * Hnn_kron_S
    Fact = SparseArrays.lu(A)
    U[:,1] = Fact \ b
    K_kron_M = kron(K, M)
    for n = 2:N
        fill!(b, 0.0)
        for l = 1:n-1
            Hnl_kron_S = kron(H[n-l], S)
            b .= b .+ Δt^α * Hnl_kron_S * U[:,l]
        end
        dynamic_load_vector!(Fn, t[n-1:n], grid, f, τ, w, Ψ)
        b .= Fn[1:nfree*rt] .+ K_kron_M * U[:,n-1] .- b
        U[:,n] = Fact \ b
    end
    return t, reshape(U, nfree, rt, N)
end

function functional(pcwise_U::Array{Float64}, pcwise_t::Array{Float64},
                    func::Function, grid::SpatialGrid, 
                    ξ::Vector{Float64}, w::Vector{Float64})
    x = grid.x
    pts_per_time_interval = size(pcwise_t,1)
    N = size(pcwise_t, 2)
    P = ubound(x, 1)
    pts_per_space_interval = length(ξ)
    Fnctl = zeros(pts_per_time_interval, N)

end
