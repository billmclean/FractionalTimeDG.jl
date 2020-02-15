import FractionalTimeDG.P, FractionalTimeDG.dP

correct_P(x) = [ 1, x, (3x^2-1)/2, (5x^3-3x)/2, (35x^4-30x^2+3)/8 ]

J = 11
τ = range(-1, 1, length=J)

Ψ = zeros(5, J)
FractionalTimeDG.legendre_polys!(Ψ, τ)
err = [ Ψ[i,j] - correct_P(τ[j])[i] for i = 1:5, j = 1:J ]
@test all( abs.(err) .< 1e-10 )

err = [ Ψ[i,j] - P(i-1, τ[j]) for i = 1:5, j = 1:J ]
@test all( abs.(err) .< 1e-10 )


correct_dP(x) = [ 0, 1, 3x, (15x^2-3)/2, (35x^3-15x)/2 ]
dΨ = zeros(5, J)
FractionalTimeDG.deriv_legendre_polys!(dΨ, τ)
err = [ dΨ[i,j] - correct_dP(τ[j])[i] for i = 1:5, j = 1:J ]
@test all( abs.(err) .< 1e-10 )

err = [ dΨ[i,j] - dP(i-1, τ[j]) for i = 1:5, j = 1:J ]
@test all( abs.(err) .< 1e-10 )

u(t) = exp(-t) * sin(2π*t)
α = 0.5
rmax = 6
Mmax = 3rmax
ppImax = 20
N = 15
store = Store(α, rmax, Mmax, ppImax)
q = parse(T, "1.5")
t = OffsetArray(T[ (n/N)^q for n = 0:N ], 0:N)
U = FractionalTimeDG.Fourier_Legendre_coefs(u, rmax, t, store)
τ = range(-1, 1, length=ppImax)
pcwise_t, pcwise_U = FractionalTimeDG.evaluate_pcwise_poly!(U, t, τ, store)
pcwise_u = [ u(t) for t in pcwise_t ]
err = pcwise_U - pcwise_u
@test all( abs.(err) .< 1e-6 )
