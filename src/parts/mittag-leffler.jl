"""
    MLneg_power_series(α, x, nmax, tol)

Sum the first `nmax` terms of the Taylor series expansion of `E_α(-x)`,
or stop when the next term is smaller than `tol`.
"""
function MLneg_power_series(α::T, x::T, nmax::Integer,
                            tol::T) where T <: AbstractFloat
    Eα = one(T)
    powx = one(T)
    for n = 1:nmax
        powx *= -x
        term = powx / Γ(1+n*α)
        Eα += term
        if abs(term) ≤ tol
            break
        end
    end
    return Eα
end

"""
    MLneg_asymptotic_series(α, x, tol, nmax)

Sum the first `nmax` terms of the asymptotic expansion of `E_α(-x)`,
or stop when the next term is smaller than `tol`.
"""
function MLneg_asymptotic_series(α::T, x::T, nmax::Integer
                            ) where T <: AbstractFloat
    recipx = one(T) / x
    powx = recipx
    Eα = powx * sinpi(α) * Γ(α) / π
    for n = 2:nmax
        powx *= -recipx
        term = powx * sinpi(n*α) * Γ(n*α) / π
        Eα += term
    end
    return Eα
end

function MLneg_integral(α::T, λ::T, t::T, nmax::Integer
                        ) where T <: AbstractFloat
    F(z) = 1 / ( z + λ * z^(1-α) )
    ϕ = parse(T, "1.1721")
    h = parse(T, "1.0818") / nmax
    μ = parse(T, "4.4921") * nmax / t
    z0 = μ * ( 1 - sin(ϕ) )
    s = exp(z0*t) * F(z0) * cos(ϕ) / 2
    for n = 1:nmax
        iunmϕ = Complex(-ϕ, n*h)
        zn = μ * ( 1 + sin(iunmϕ) )
        s += real( exp(zn*t) * F(zn) * cos(iunmϕ) )
    end
    return μ * h * s / π
end

function chebyshev_polys!(Cheb::OffsetArray{T}, x::T
                         ) where T <: AbstractFloat
    nmax = length(Cheb) - 1
    Cheb[0] = one(T)
    if nmax ≥ 1
        Cheb[1] = x
    end
    if nmax ≥ 2
        for n = 1:nmax-1
            Cheb[n+1] = 2x * Cheb[n] - Cheb[n-1]
        end
    end
end

function chebyshev_polys!(Cheb::OffsetArray{T}, x::AbstractVector{T}
                         ) where T <: AbstractFloat
    nmax, M = size(Cheb)
    nmax -= 1
    @argcheck Cheb.offsets == (-1, 0)
    @argcheck length(x) == M
    for m = 1:M
        Cheb[0,m] = one(T)
    end
    if nmax ≥ 1
        for m = 1:M
            Cheb[1,m] = x[m]
        end
    end
    if nmax ≥ 2
        for m = 1:M, n = 1:nmax-1
            Cheb[n+1,m] = 2x[m] * Cheb[n,m] - Cheb[n-1,m]
        end
    end
end

function chebyshev_coefs!(a::OffsetArray{T}, f::Function, M::Integer
                         ) where T <: AbstractFloat
    nmax = length(a) - 1
    @argcheck a.offsets == (-1,)
    x, w = GaussQuadrature.chebyshev(T, M)
    Cheb = OffsetArray{T}(undef, 0:nmax, 1:M)
    chebyshev_polys!(Cheb, x)
    for n = 0:nmax
        s = zero(T)
        for m = 1:M
            s += w[m] * f(x[m]) * Cheb[n,m]
        end
        a[n] = (2/π) * s
    end
end

function chebyshev_sum(a::OffsetArray{T}, x::T) where T <: AbstractFloat
    nmax = length(a) - 1
    bn = bnp1 = bnp2 = zero(T)
    for n = nmax:-1:1
        bn = a[n] + 2x*bnp1 - bnp2
        bnp2 = bnp1
        bnp1 = bn
    end
    bn = a[0] + 2x*bnp1 - bnp2
    return ( bn - bnp2 ) / 2
end
