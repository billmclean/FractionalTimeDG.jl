import FractionalTimeDG
using OffsetArrays
using PyPlot

include("../FPDE_problem.jl")

Nx = 8
Nt = 4
rx = 4
rt = 3
Mt = 2rt
ppImax = 10
store = FractionalTimeDG.Store(α, rt, Mt, ppImax)

x = OffsetVector(range(0, L, length=Nx+1), 0:Nx)
grid = FractionalTimeDG.SpatialGrid(x, rx, ("essential", "essential"), rx)

U0 = FractionalTimeDG.spatial_L2_projection(u0, grid)
t, U = FractionalTimeDG.FPDEDG(1.0, f, U0, grid, max_t, Nt, rt, Mt, store)

#pts_per_time_interval = 5
#pts_per_space_interval = 5
#xvals, pcwise_t, pcwise_U = FractionalTimeDG.evaluate_pcwise_poly(
#    U, t, grid, pts_per_time_interval, pts_per_space_interval)
τ = [ -1.0, -0.5, 0.0, 0.5, 1.0 ]
pts_per_time_interval = length(τ)
ξ = [ -0.9, -0.5, 0.0, 0.5, 0.9 ]
xvals, pcwise_t, pcwise_U = FractionalTimeDG.evaluate_pcwise_poly(
    U, t, grid, τ, ξ)
τ = [ -1.0, -0.5, 0.0, 0.5, 1.0 ]

figure(1)
Nxvals = length(xvals)
Ntvals = pts_per_time_interval
X = Float64[ xvals[i] for i = 1:Nxvals, j = 1:Ntvals ]
for n = 1:Nt
    T = Float64[ pcwise_t[j,n] for i = 1:Nxvals, j = 1:Ntvals ]
    Z = Float64[ pcwise_U[i,j,n] for i = 1:Nxvals, j = 1:Ntvals ]
    surf(X, T, Z)
end
xlabel(L"$x$")
ylabel(L"$t$")
