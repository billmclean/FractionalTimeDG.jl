include("sec_6_2.jl")
using Printf
using OffsetArrays

r = 3
M = 8
τ = FractionalTimeDG.reconstruction_pts(T, r)
ppI = M
store = Store(α, r+1, M, ppI)

nrows = 6
N0 = 8 # should be a multiple of 4
Nvals = Integer[ N0 * 2^i for i = 0:nrows-1 ]
err = OffsetArray{T}(undef, 0:r, 1:nrows)
for row = 1:nrows
    N = Nvals[row]
    @printf("%5d ", N)
    t, U = FODEdG!(λ, tmax, f, u0, N, r, M, store)
    pcwise_t, pcwise_U = evaluate_pcwise_poly!(U, t, τ[0:r], store)
    pcwise_u = T[ u(t, λ, u0, f, 20, store) for t in pcwise_t ]
    weight = pcwise_t.^(r-α)
    pcwise_err = OffsetArray(weight.*(pcwise_U - pcwise_u), 0:r, 1:N)
    for j = 0:r
        err[j,row] = maximum(abs.(pcwise_err[j,:]))
    end
    if row == 1
        for j = 0:r
            @printf("&%8.1e&%5s ", err[j,row], "")
        end
    else
        N_ratio = Nvals[row] / Nvals[row-1]
        for j = 0:r
            err_ratio = err[j,row-1] / err[j,row]
            rate = log(err_ratio) / log(N_ratio)
            @printf("&%8.1e&%5.2f ", err[j,row], rate)
        end
    end
    @printf("\\\\\n")
end





