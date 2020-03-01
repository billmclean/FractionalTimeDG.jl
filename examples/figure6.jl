using FractionalTimeDG
import FractionalTimeDG
import FractionalTimeDG.evaluate_pcwise_poly
import FractionalTimeDG.evaluate_pcwise_error
import GaussQuadrature
using OffsetArrays
using PyPlot

include("FPDE_problem.jl")

Nx = 20
Nt = 12
Nquad = 15
u(x, t) = refsoln(x, t, α, Nquad)
rx = 4
rt = 3
Mt = 2rt
Mx = 2rx
ppImax = 15

store = Store(α, rt+1, Mt, ppImax)
x_ = collect(range(0, L, length=Nx+1))
x = OffsetVector(x_, 0:Nx)
grid = FractionalTimeDG.SpatialGrid(x, rx, ("essential", "essential"), Mx)

U0 = FractionalTimeDG.spatial_L2_projection(u0, grid)
t, U = FractionalTimeDG.FPDEDG(1.0, f, U0, grid, max_t, Nt, rt, Mt, store)
τ = range(-1, 1, length=ppImax)
pcwise_t, Uerr = L2error(U, u, t, τ, grid, Mx)

jump = evaluate_jumps(U, U0, t, grid, Mx)
Uhat, jumpU = reconstruction(U, U0, store)
pcwise_t, Uhaterr = L2error(Uhat, u, t, τ, grid, Mx)

figure(1)
line1 = semilogy(t[0:Nt-1], abs.(jump[0:Nt-1]), "C0o")
line2 = semilogy(pcwise_t, Uerr, "C1-")
line3 = semilogy(pcwise_t, Uhaterr, "k:")
legend((line1[1], line2[2], line3[1]),
       (L"jump in $U", L"error in $U$", L"error in $\hat U$"))
xlabel(L"$t$")
PyPlot.grid(true)
savefig("fig6.pdf")
