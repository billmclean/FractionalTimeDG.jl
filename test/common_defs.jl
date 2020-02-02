import FractionalTimeDG.Γ, 
       FractionalTimeDG.P, 
       FractionalTimeDG.dP, 
       FractionalTimeDG.legendre_polys!,
       FractionalTimeDG.deriv_legendre_polys!,
       QuadGK.quadgk
       OffsetArrays.OffsetArray

function Δ(n::Integer, ℓ::Integer, τ::T, σ::T, 
           t::OffsetArray{T}) where T <: AbstractFloat
    kn = t[n] - t[n-1]
    kℓ = t[ℓ] - t[ℓ-1]
    two_Dnℓ = (t[n]+t[n-1]) - (t[ℓ]+t[ℓ-1])
    return ( τ * kn - σ * kℓ ) / two_Dnℓ
end

