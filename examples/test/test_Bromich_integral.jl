import SpecialFunctions
include("../examples/FPDE_problem.jl")

Γ = SpecialFunctions.gamma
α = 1.6
F(z) = z^(-α)
N = 14

t = range(0.1, 5.0, length=201)
y = Float64[ Bromich_integral(tval, F, N) for tval in t ]

err = maximum(abs.( y - t.^(α-1) / Γ(α) ))

