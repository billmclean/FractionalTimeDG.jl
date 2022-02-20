include("sec_6_2.jl")

using PyPlot

r = 3
M = 2r
ppI = 40
store = Store(α, r+1, M, ppI)

N = 5
t, U = FODEdG!(λ, tmax, f, u0, N, r, M, store)
τ = FractionalTimeDG.reconstruction_pts(T, r)
pcwise_t, pcwise_U = evaluate_pcwise_poly!(U, t, τ[0:r], store)
pcwise_u = [ u(t, λ, u0, f, 10, store) for t in pcwise_t ]
pcwise_err = pcwise_U - pcwise_u
Uhat = reconstruction(U, u0, store)
pcwise_t_fine, pcwise_Uhat = evaluate_pcwise_poly!(Uhat, t, ppI, store)
pcwise_u_fine = [ u(t, λ, u0, f, 10, store) for t in pcwise_t_fine ]
pcwise_rec_err = pcwise_Uhat - pcwise_u_fine
pcwise_rec_err[1,1] = NaN
a = FractionalTimeDG.Fourier_Legendre_coefs(r, t, store) do s
    return u(s, λ, u0, f, 10, store)
end

figure(4)
trans_pcwise_t = copy(pcwise_t')
trans_pcwise_err = copy(pcwise_err')
line0 = semilogy(pcwise_t_fine, abs.(pcwise_rec_err), "k:")
line1 = semilogy(trans_pcwise_t[:,1], abs.(trans_pcwise_err[:,1]), "o", 
                 markersize=3)
line2 = semilogy(trans_pcwise_t[:,2], abs.(trans_pcwise_err[:,2]), "v", 
                 markersize=3)
line3 = semilogy(trans_pcwise_t[:,3], abs.(trans_pcwise_err[:,3]), "+", 
                 markersize=4)
line4 = semilogy(trans_pcwise_t[:,4], abs.(trans_pcwise_err[:,4]), "x", 
                 markersize=4)
legend((line0[1], line1[1], line2[1], line3[1], line4[1]),
       (L"$\widehat E(t)$", L"$j=0$", L"j=1", L"j=2", L"j=3"))
xlabel(L"$t$", size=12)
ylabel("Error")
grid(true)
savefig("fig4.eps")
