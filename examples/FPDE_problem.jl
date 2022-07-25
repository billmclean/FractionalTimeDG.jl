import GaussQuadrature.legendre
using FractionalTimeDG
using ArgCheck 
using OffsetArrays

#include("FEM1D.jl")

const L = 2.0
const max_t = 2.0
const C0 = 1.0
const Cf = 2.0
α = 0.6

u0(x) = C0 * x * (L-x)
f(x,t) = Cf * t * exp(-t)

function refsoln(x, t, α, N)
    return Bromwich_integral(t, z -> uhat(x, z, α), N)
end

function uhat(x::T, z::Complex{T}, α::T) where T <: AbstractFloat
    ω = z^(α/2)
    ρ1(x) = ( ω * x * (L-x) - 2/ω ) * cosh(ω*x) + (2x-L) * sinh(ω*x) + 2 / ω
    ρ2(x) = cosh(ω*x) - 1
    return ( (C0/ω) * ( ρ1(x) * sinh(ω*(L-x)) + ρ1(L-x) * sinh(ω*x) )
           + Cf/(z+1)^2 * ( ρ2(x) * sinh(ω*(L-x)) + ρ2(L-x) * sinh(ω*x) )
          ) / ( z * sinh(ω*L) )
end

"""
    Bromwich_integral(t, F, N)

Evaluates `1/(2πi)` times the integral of `exp(zt)F(z) dz` along a
Hankel contour, assuming `F(z)` is analytic in the cut plane
`|arg(z)|<π` and `F(conj(z)) = conj(F(z))`.
"""
function Bromwich_integral(t::Float64, F::Function, N::Integer)
    μ = 4.492075 * N / t
    ϕ = 1.172104
    h = 1.081792 / N
    z0, w0 = integration_contour(0.0, μ, ϕ)
    Σ = real( w0 * exp(z0*t) * F(z0) ) / 2
    for n = 1:N
        zn, wn = integration_contour(n*h, μ, ϕ)
        Σ += real( wn * exp(zn*t) * F(zn) )
    end
    return h * Σ
end

function integration_contour(u, μ, ϕ) 
    z = μ * ( 1 + sin(Complex(-ϕ, u)) )
    w = (μ/π) * cos(Complex(-ϕ, u))
    return z, w
end

function L2error(U::Array{Float64}, u::Function, t::OffsetArray{Float64},
                 τ::AbstractVector{Float64}, 
                 grid::SpatialGrid{Float64}, Mx::Integer)
    ξ, w = legendre(Mx)
    xvals, pcwise_t, pcwise_U = evaluate_pcwise_poly(U, t, grid, τ, ξ)
    pcwise_err = evaluate_pcwise_error(pcwise_U, u, pcwise_t, xvals)

    x = grid.x
    pts_per_time_interval = length(τ)
    Nt = length(t) - 1
    Nx = length(x) - 1
    err = Array{Float64}(undef, pts_per_time_interval, Nt)
    for nt = 1:Nt
        for i = 1:pts_per_time_interval
            m = 0
            s = 0.0
            for nx = 1:Nx
                hx = x[nx] - x[nx-1]
                for j = 1:Mx
                    m += 1
                    s += (hx/2) * w[j] * pcwise_err[m,i,nt]^2
                end
            end
            err[i,nt] = sqrt(s)
        end
    end
    return pcwise_t, err
end

function evaluate_jumps(U::Array{Float64}, U0::Vector{Float64}, 
                        t::OffsetVector{Float64}, 
                        grid::SpatialGrid{Float64}, Mx::Integer)
    x = grid.x
    Nx = length(x) - 1
    ξ, w = legendre(Mx)
    τ = [-1.0, 1.0]
    xvals, pcwise_t, pcwise_U = evaluate_pcwise_poly(U, t, grid, τ, ξ)
    U0vals = similar(xvals)
    evaluate_fem1d_soln!(U0vals, U0, grid, ξ)
    Nt = length(t) - 1
    jump = OffsetArray{Float64}(undef, 0:Nt-1)
    m = 0
    s = 0.0
    for nx = 1:Nx
        hx = x[nx] - x[nx-1]
        for j = 1:Mx
            m += 1
            s += (hx/2) * w[j] * ( pcwise_U[m,1,1] - U0vals[m] )^2
        end
    end
    jump[0] = sqrt(s)
    for nt = 1:Nt-1
        m = 0
        s = 0.0
        for nx = 1:Nx
            hx = x[nx] - x[nx-1]
            for j = 1:Mx
                m += 1
                s += (hx/2) * w[j] * ( pcwise_U[m,1,nt+1] - pcwise_U[m,2,nt] )^2
            end
        end
        jump[nt] = sqrt(s)
    end
    return jump
end

function reconstruction(U::Array{Float64}, U0::Vector{Float64}, 
                        store::Store{Float64})
    NgDoF, rt, Nt = size(U)
    Uhat = Array{Float64}(undef, NgDoF, rt+1, Nt)
    jumpU = OffsetArray{Float64}(undef, 1:NgDoF, 0:Nt-1)
    rmax = store.rmax
    pow = OffsetArray{Float64}(undef, 0:rmax)
    pow[0:2:end] .=  1.0
    pow[1:2:end] .= -1.0
    @argcheck rt+1 ≤ rmax
    U_left  = zeros(NgDoF)
    U_right = zeros(NgDoF)
    for j = 1:rt
        U_left .+= pow[j-1] * U[:,j,1]
        U_right .+= U[:,j,1]
    end
    jumpU[:,0] .= U_left - U0
    Uhat[:,1:rt,1] .= U[:,1:rt,1]
    Uhat[:,rt,1] .+= pow[rt] * jumpU[:,0] / 2
    Uhat[:,rt+1,1] .= - pow[rt] * jumpU[:,0] / 2
    for n = 2:Nt
        fill!(U_left, 0.0)
        for j = 1:rt
            U_left .+= pow[j-1] * U[:,j,n]
        end
        jumpU[:,n-1] .= U_left - U_right
        Uhat[:,1:rt,n] .= U[:,1:rt,n]
        Uhat[:,rt,n]  .+=  pow[rt] * jumpU[:,n-1] / 2
        Uhat[:,rt+1,n] .= -pow[rt] * jumpU[:,n-1] / 2
        fill!(U_right, 0.0)
        for j = 1:rt
            U_right .+= U[:,j,n]
        end
    end
    return Uhat, jumpU
end
