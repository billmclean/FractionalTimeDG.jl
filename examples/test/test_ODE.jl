import FractionalTimeDG: Store, ODEdG!, reconstruction_pts, 
                         evaluate_pcwise_poly!, jumps, dG_error_estimator

using PyPlot
#using Polynomials: polyval

T = Float64
#T = BigFloat

λ = 1 / parse(T, "2")
u0 = one(T)
ampl = one(T)
f(t) = ampl * cos(π*t)
c =  λ^2 + big(π)^2
u(t) = ( ( u0 - ampl * λ / c ) * exp(-λ*t)
        + ampl * ( λ * cos(π*t) + π * sin(π*t) ) / c )
tmax = parse(T, "2")

N = 4
r = 3
M = 4
pts_per_interval = 50
store = Store(one(T), r, M, pts_per_interval)
t, U = ODEdG!(λ, tmax, f, u0, N, r, M)
pcwise_t, pcwise_U = evaluate_pcwise_poly!(U, t, pts_per_interval, store)
pcwise_u = T[ u(t) for t in pcwise_t ]
figure(1)
line1 = plot(pcwise_t, pcwise_u, "k:")
line2 = plot(pcwise_t, pcwise_U, "c")
legend((line1[1], line2[1]), (L"$u$", L"$U$"))
xlabel(L"$t$", size=14)
ylabel(L"$U$", size=14)
xticks((0.0, 1/2, 1, 3/2, 2), ("0.0", "0.5", "1.0", "1.5", "2.0"))
grid(true)

figure(2)
JU = jumps(U, t, u0)
pcwise_tx, pcwise_approx_err = dG_error_estimator(JU, t, r, pts_per_interval)
line1 = plot(pcwise_t, pcwise_U-pcwise_u, "C0")
line2 = plot(pcwise_tx, pcwise_approx_err, "C1:")
xticks((0.0, 1/2, 1, 3/2, 2), ("0.0", "0.5", "1.0", "1.5", "2.0"))
legend((line1[1], line2[1]), (L"$U-u$", "error estimator"))
grid(true)
