using PyPlot

include("FPDE_problem.jl")

Nx = 20
Nt = 30
Nquad = 15
x = range(0, L, length=Nx)
t = Float64[ max_t * (j/Nt)^1.5 for j = 1:Nt ]
X = Float64[ x[i] for i = 1:Nx, j = 1:Nt ]
T = Float64[ t[j] for i = 1:Nx, j = 1:Nt ]
u = Array{Float64}(undef, Nx, Nt)
u[:,1] = u0.(x)
for j = 2:Nt, i = 1:Nx
    u[i,j] = refsoln(x[i], t[j], α, Nquad)
end

figure(1)
mesh(X, T, u)
xlabel(L"$x$")
ylabel(L"$t$")
ax = gca()
ax.view_init(elev=30, azim=35)
savefig("fig5.pdf")
