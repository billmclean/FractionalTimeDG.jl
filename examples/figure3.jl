include("sec_6_2.jl")

using PyPlot

r = 3
M = 2r
ppI = 40
store = Store(α, r, M, ppI)

N = 3
t, U = FODEdG!(λ, tmax, f, u0, N, r, M, store)

figure(3)
tt = range(0, tmax, length=201)
exact_u = [ u(t, λ, u0, f, M, store) for t in tt ]
pcwise_t, pcwise_U = evaluate_pcwise_poly!(U, t, ppI, store)
line1 = plot(tt, exact_u, "k:")
line2 = [ plot(pcwise_t[:,n], pcwise_U[:,n], "c") for n = 1:N ]
legend((line1[1], line2[1][1]), (L"$u$", L"$U$"))
grid(true)
xlabel(L"$t$", fontsize=12)
xticks((0.0, 2/3, 4/3, 2), ("0", "2/3", "4/3", "2"))
savefig("fig3.pdf")
