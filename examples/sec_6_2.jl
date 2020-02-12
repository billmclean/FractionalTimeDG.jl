using FractionalTimeDG
using SpecialFunctions: erfcx

T = Float64
α = 1 / parse(T, "2")
r = 2
M = 2r
store = Store(α, r, M)

Ehalf(x) = erfcx(-x)

function u(t::T, λ::T, u0::T, f::Function, M::Integer, 
           store::Store{T}) where T <: AbstractFloat
    y, wy = rule(store.unitlegendre[M])
    s = zero(T)
    for m = 1:M
        s += wy[m] * Ehalf(-λ*sqrt(t)*y[m]) * f(t*(1-y[m]^2)) * y[m]
    end
    return u0 * Ehalf(-\lambda*sqrt(t)) + 2t * s
end

N = 3
t, U = FODEdG!(λ, tmax, f, u0, N, r, M, store)

