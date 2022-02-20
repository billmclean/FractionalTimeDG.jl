using FractionalTimeDG
using PyPlot

α = 3/4
N = 1000
r = 4
M = 2r
version = 1

store = Store(α, r, M)
H = coef_H_uniform!(r, N, M, store, version)

ℓ = collect(1:N-1)

function max_antidiags(H)
    r = size(H[0], 1)
    N = length(H)
    γ = zeros(N-1, 2r-1)
    for m = 2:2r
        for ℓbar = 1:N-1
            if m ≤ r
                antidiag = [ H[ℓbar][i,m-i] for i = 1:m-1 ]
            else
                antidiag = [ H[ℓbar][i,m-i] for i = m-r:r ]
            end
#            println(antidiag)
            largest = maximum(abs.(antidiag))
            if largest ≥ eps(Float64)
                γ[ℓbar, m-1] = largest
            else
                γ[ℓbar, m-1] = NaN
            end
        end
    end
    return γ
end

γ = max_antidiags(H)

figure(2)
fmt = [":", "--", "-.", "-", ":", "--", "-."]
for m = 2:2r
    loglog(ℓ, γ[:,m-1], fmt[m-1])
end
legend([latexstring("i+j=$m") for m = 2:2r])
grid(true)
xlabel(L"$\bar\ell$", fontsize=12)
savefig("fig2.eps")
