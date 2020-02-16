import FractionalTimeDG

T = Float64
M = 11
θ = range(-T(π), T(π), length=M)
x = cos.(θ)
nmax = 5
Cheb = OffsetArray{T}(undef, 0:nmax, 1:M)
FractionalTimeDG.chebyshev_polys!(Cheb, x)

for n = 0:nmax
    err = Cheb[n,:] - cos.(n*θ)
    @test all( abs.(err) .< 1e-10 )
end
