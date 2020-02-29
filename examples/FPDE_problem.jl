
const L = 2.0
const max_t = 2.0
const C0 = 1.0
const Cf = 2.0
α = 0.6

u0(x) = C0 * x * (L-x)
f(x,t) = Cf * t * exp(-t)

function refsoln(x, t, α, N)
    return Bromich_integral(t, z -> uhat(x, z, α), N)
end

function uhat(x::T, z::Complex{T}, α::T) where T <: AbstractFloat
    ω = z^(α/2)
    ρ1(x) = ( ω * x * (L-x) - 2/ω ) * cosh(ω*x) + (2x-L) * sinh(ω*x) + 2 / ω
    ρ2(x) = cosh(ω*x) - 1
    return ( (C0/ω) * ( ρ1(x) * sinh(ω*(L-x)) + ρ1(L-x) * sinh(ω*x) )
           + Cf/(z+1)^2 * ( ρ2(x) * sinh(ω*(L-x)) + ρ2(L-x) * sinh(ω*x) )
          ) / ( z * sinh(ω*L) )
end

"""
    Bromich_integral(t, F, N)

Evaluates `1/(2πi)` times the integral of `exp(zt)F(z) dz` along a
Hankel contour, assuming `F(z)` is analytic in the cut plane
`|arg(z)|<π` and `F(conj(z)) = conj(F(z))`.
"""
function Bromich_integral(t::Float64, F::Function, N::Integer)
    μ = 4.492075 * N / t
    ϕ = 1.172104
    h = 1.081792 / N
    z0, w0 = integration_contour(0.0, μ, ϕ)
    Σ = real( w0 * exp(z0*t) * F(z0) ) / 2
    for n = 1:N
        zn, wn = integration_contour(n*h, μ, ϕ)
        Σ += real( wn * exp(zn*t) * F(zn) )
    end
    return h * Σ
end

function integration_contour(u, μ, ϕ) 
    z = μ * ( 1 + sin(Complex(-ϕ, u)) )
    w = (μ/π) * cos(Complex(-ϕ, u))
    return z, w
end
