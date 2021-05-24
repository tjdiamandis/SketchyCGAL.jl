abstract type SCGALResults end
struct SCGALSolution{T<:Real} <: SCGALResults
    UT::AbstractMatrix{T}
    ΛT::AbstractMatrix{T}
    ST::AbstractMatrix{T}
    Ω::AbstractMatrix{T}
    yT::AbstractVector{T}
    zT::AbstractVector{T}
    log
end
