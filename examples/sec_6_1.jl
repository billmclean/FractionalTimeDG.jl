import FractionalTimeDG
using Printf
using ArgCheck

sample = [1, 2, 10, 100, 1000]
α = 3/4
rmax = 6
Mmax = 2rmax
store = FractionalTimeDG.Store(α, rmax, Mmax)
version = 2

function coef_uniform(sample, r, α, M, version)
    Ns = length(sample) 
    H_sample = Matrix{Matrix{Float64}}(undef, Ns, M)
    N = maximum(sample) + 1
    for m = 1:M
        H = FractionalTimeDG.coef_H_uniform!(r, N, m, store, version)
        for n = 1:Ns
            H_sample[n, m] = H[sample[n]][:,:]
        end
    end
    return H_sample
end

function latex_display(A::Matrix{Float64})
    m, n = size(A)
    maxsz = maximum(abs.(A))
    p = min(0, ceil(Integer, log10(maxsz)))
    @printf("10^{%0d}\n", p)
    for i = 1:m
        for j = 1:n-1
            @printf("%8.5f & ", 10.0^(-p) * A[i,j])
        end
        if i < m
            @printf("%8.5f\\\\\n", 10.0^(-p) * A[i,n])
        else
            @printf("%8.5f\n", 10.0^(-p) * A[i,n])
        end
    end
end

H0 = FractionalTimeDG.coef_H0_uniform!(rmax, store)
println("H_0:")
latex_display(H0)
Hs = coef_uniform(sample, rmax, α, Mmax, version)
for n = 1:length(sample)
    ℓ = sample[n]
    println("H_$ℓ:")
    latex_display(Hs[n,Mmax])
end

function pts_needed(Hs, sample, rmax, α, atol)
    Ns, M = size(Hs)
    @argcheck length(sample) == Ns
    first_M = fill(-1, rmax, Ns)
    err = zeros(rmax, M)
    for n = 1:Ns
        ℓ = sample[n]
        for m = 1:M-1
            ΔH = Hs[n,m] - Hs[n,M]
            for r = 1:rmax
                err[r,m] = maximum(abs.(ΔH[1:r,1:r]))
            end
        end
        for r = 1:rmax
            for m = 1:M-1
                if abs(err[r,m]) < atol
                    first_M[r,n] = m
                    break
                end
            end
        end
    end
    return first_M
end

atol = 1e-14
println("\nQuadrature points needed using version = ", version)
first_M = pts_needed(Hs, sample, rmax, α, atol)
@printf("%4s", "r")
for ℓ in sample
    @printf("  ℓ=%4d", ℓ)
end
@printf("\n\n")
for r = 1:rmax
    @printf("%3d&", r)
    for n = 1:length(sample)
        @printf("%7d&", first_M[r,n])
    end
    @printf("\n")
end
