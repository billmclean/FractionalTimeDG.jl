import FractionalTimeDG
using OffsetArrays
using PyPlot

include("../FPDE_problem.jl")
α = 1.0

Nx = 8
Nt = 4
rx = 4
rt = 3
Mt = 2rt

x = OffsetVector(range(0, L, length=Nx+1), 0:Nx)
grid = FractionalTimeDG.SpatialGrid(x, rx, ("essential", "essential"), rx)

U0 = FractionalTimeDG.spatial_L2_projection(u0, grid)
t, U = FractionalTimeDG.PDEDG(1.0, f, U0, grid, max_t, Nt, rt, Mt)

pts_per_time_interval = 5
pts_per_space_interval = 5
xvals, pcwise_t, pcwise_U = FractionalTimeDG.evaluate_pcwise_poly(
    U, t, grid, pts_per_time_interval, pts_per_space_interval)

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
