include("sec_6_2.jl")
using Printf

r = 3
Mmax = ppImax = 10
M = 10
ppI = 5
store = Store(α, r+1, M, ppImax)

nrows = 6
N0 = 8
four = parse(T, "4")
qvals = T[1, 3, 5, 6]
ncols = length(qvals)
Nvals = Integer[ N0 * 2^i for i = 0:nrows-1 ]
err = Array{T}(undef, nrows, ncols)
@printf("%5s ", "N")
for q in qvals
    @printf("       q = %1d    ", q)
end
@printf("\n\n")
for row = 1:nrows
    N = Nvals[row]
    @printf("%5d ", N)
    for col = 1:ncols
        q = qvals[col]
        t = OffsetArray(T[ (n/N)^q * tmax/2 for n = 0:N ], 0:N)
        U = FODEdG!(λ, f, u0, t, r, M, store)
        Uhat = reconstruction(U, u0, store)
        pcwise_t, pcwise_Uhat = evaluate_pcwise_poly!(Uhat, t, ppI, store)
        pcwise_u = T[ u(t, λ, u0, f, 20, store) for t in pcwise_t ]
        pcwise_err = pcwise_Uhat - pcwise_u
        err[row,col] = maximum(abs.(pcwise_err))
        if row == 1
            @printf("&%8.1e&%5s ", err[row,col], "")
        else
            N_ratio = Nvals[row] / Nvals[row-1]
            err_ratio = err[row-1,col] / err[row,col]
            rate = log(err_ratio) / log(N_ratio)
            @printf("&%8.1e&%5.2f ", err[row,col], rate)
        end
    end
    @printf("\\\\\n")
end

