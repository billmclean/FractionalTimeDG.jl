function Δ(n::Integer, ℓbar::Integer, τ::T, σ::T, 
           t::OffsetArray{T}) where T <: AbstractFloat
    ℓ = n - ℓbar
    kn = t[n] - t[n-1]
    kℓ = t[ℓ] - t[ℓ-1]
    two_D = (t[n]+t[n-1]) - (t[ℓ]+t[ℓ-1])
    return ( τ * kn - σ * kℓ ) / two_D
end

