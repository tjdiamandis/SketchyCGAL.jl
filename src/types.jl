abstract type SCGALResults end
abstract type OptimizationLog end

struct SCGALSolution{T<:Real} <: SCGALResults
    UT::AbstractMatrix{T}
    ΛT::AbstractMatrix{T}
    ST::AbstractMatrix{T}
    Ω::AbstractMatrix{T}
    yT::AbstractVector{T}
    zT::AbstractVector{T}
    log::Union{OptimizationLog, Nothing}
end


struct SCGALLog{T<:Real} <: OptimizationLog
    dual_gap::AbstractVector{T}
    obj_val::AbstractVector{T}
    primal_infeas::AbstractVector{T}
    time_sec::AbstractVector{T}
    obj_val_Xhat::Union{AbstractVector{T}, Nothing}
    primal_infeas_Xhat::Union{AbstractVector{T}, Nothing}
end
