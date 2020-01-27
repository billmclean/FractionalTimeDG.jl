using Printf

sample = [1, 2, 10, 100, 1000]
rmax = 6
α = 3/4
M = 2r

function coef_uniform(sample, r, α, M)
    Ns = length(sample) 
    H_sample = Array{Float64}(undef, r, r, Ns, M)
    N = maximum(sample) + 1
    for m = 1:M
        H = coef_H_uniform(N, r, α, m)
        for n = 1:Ns
            H_sample[:, :, n, m] = H[:,:,sample[n]]
        end
    end
    return H_sample
end

function pts_needed(sample, rmax, α, M, atol)
    Ns = length(sample)
    first_M = fill(-1, rmax, Ns)
    err = zeros(rmax, M)
    for n = 1:Ns
        ℓ = sample[n]
        H = coef_uniform(sample, rmax, α, M)
        for m = 1:M-1
            ΔH = H[:,:,n,m] - H[:,:,n,M]
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
first_M = pts_needed(sample, rmax, α, M, atol)

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

H0 = coef_H0(r, α)
println("H_0:")
latex_display(H0)
Hs = coef_uniform(sample, r, α, M)
for n = 1:length(sample)
    ℓ = sample[n]
    println("H_$ℓ:")
    latex_display(Hs[:,:,n,M])
end
