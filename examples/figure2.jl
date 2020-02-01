using FractionalTimeDG
using PyPlot

α = 3/4
N = 1000
r = 4
M = 2r

H = coef_H_uniform(N, r, α, M)

ℓ = collect(1:N-1)

function max_antidiags(H)
    r = size(H, 1)
    N = size(H, 3) + 1
    γ = zeros(N-1, 2r-1)
    for m = 2:2r
        for ℓ = 1:N-1
            if m ≤ r
                antidiag = [ H[i,m-i,ℓ] for i = 1:m-1 ]
            else
                antidiag = [ H[i,m-i,ℓ] for i = m-r:r ]
            end
#            println(antidiag)
            largest = maximum(abs.(antidiag))
            if largest ≥ eps(Float64)
                γ[ℓ, m-1] = largest
            else
                γ[ℓ, m-1] = NaN
            end
        end
    end
    return γ
end

γ = max_antidiags(H)

figure(1)
loglog(ℓ, γ)
legend([latexstring("i+j=$m") for m = 2:2r])
grid(true)
xlabel(L"$\ell$", fontsize=12)
savefig("fig1.pdf")
