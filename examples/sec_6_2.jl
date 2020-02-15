using FractionalTimeDG
import FractionalTimeDG
using SpecialFunctions: erfcx
import GaussQuadrature

T = Float64
two = parse(T, "2")
α = 1 / two
λ = 1 / two
tmax = two

Ehalf(x) = erfcx(-x)

f(t) = cos(π*t)
u0 = one(T)

function u(t::T, λ::T, u0::T, f::Function, M::Integer, 
           store::Store{T}) where T <: AbstractFloat
    y, wy = GaussQuadrature.legendre(T, M)
    y .= ( y .+ 1 ) / 2
    wy .= wy / 2
    s = zero(T)
    for m = 1:M
        s += wy[m] * Ehalf(-λ*sqrt(t)*y[m]) * f(t*(1-y[m]^2)) * y[m]
    end
    return u0 * Ehalf(-λ*sqrt(t)) + 2t * s
end

