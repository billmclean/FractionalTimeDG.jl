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

