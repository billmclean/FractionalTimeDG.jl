correct_P(x) = [ 1, x, (3x^2-1)/2, (5x^3-3x)/2, (35x^4-30x^2+3)/8 ]

J = 11
τ = range(-1, 1, length=J)

P = zeros(5, J)
legendre_polys!(P, τ)
err = [ P[i,j] - correct_P(τ[j])[i] for i = 1:5, j = 1:J ]
@test all( abs.(err) .< 1e-10 )
correct_dP(x) = [ 0, 1, 3x, (15x^2-3)/2, (35x^3-15x)/2 ]

dP = zeros(5, J)
deriv_legendre_polys!(dP, τ)
err = [ dP[i,j] - correct_dP(τ[j])[i] for i = 1:5, j = 1:J ]
@test all( abs.(err) .< 1e-10 )
