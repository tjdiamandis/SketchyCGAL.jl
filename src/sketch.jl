function init_sketch(n, R; rseed=0)
    # Random.seed!(rseed)
    return randn(n, R), zeros(n, R)
end


# cache should have length R
function rank_one_update!(S, Ω, v, η; cache::AbstractVector=nothing)
    @assert length(cache) == size(S, 2)
    S .= (1 - η) .* S
    if isnothing(cache)
        tmp = Ω'*v
        BLAS.ger!(η, v, tmp, S)
    else
        mul!(cache, Ω', v)
        BLAS.ger!(η, v, cache, S)
    end
end


# TODO: maybe should make this unallocating for experiments?
# Numerical stable version of eq (5.3)
#  X̂ = S*pinv(Ω'*S)*S' = U*Λ*U'
function reconstruct(Ω, S; correction=false)
    n = size(S, 1)
    σ = sqrt(n)*eps()*norm(S)
    Sσ = S + σ .* Ω
    M = Ω'*S
    M .= 0.5 .* (M .+ M')
    F = nothing

    # Some error handling -- PosDefException on cholesky for first few iters
    # TODO: figure out why this occurs
    try
        F = cholesky(M)
    catch e
        !isa(e, PosDefException) && error("Sketch reconstruction error")
        M[diagind(M)] .+= minimum(eigvals(M)) .+ σ
        F = cholesky(M)
    end
    U, Σ, _ = svd(Sσ / F.U)
    Λ = Diagonal(max.(0, Σ.^2 .- σ))

    # Implements trace correction (Remark 6.1)
    # Note: correction doubles the error at worse; usually works better in practice
    if correction
        R = size(S, 2)
        correction_factor = (1.0 - tr(Λ))/R
        Λ.diag .= Λ.diag .+ correction_factor
    end

    return U, Λ
end


function construct_Xhat(soln::SCGALResults)
    return soln.UT*soln.ΛT*soln.UT'
end


# Avoids overhead of constructing Xhat fully
# cache is a n x R matrix
function compute_objective(C, U, Λ; cache=nothing)
    n = size(C, 1)
    if isnothing(cache)
        cache = C*U
    else
        mul!(cache, C, U)
    end

    obj_val = 0
    @views for i in 1:size(Λ, 1)
        obj_val += Λ.diag[i] * dot(U[:,i], cache[:,i])
    end

    return obj_val
end


# Avoids overhead of constructing Xhat fully
# cache is a n x R matrix
function compute_primal_infeas_mc(U, Λ; cache=nothing)
    n, R = size(U)
    @assert size(cache) == size(U)
    if isnothing(cache)
        cache = U*Λ
    else
        mul!(cache, U, Λ)
    end

    infeas_val = 0
    @views for i in 1:n
        infeas_val += (dot(cache[i,:], U[i,:]) - 1/n)^2
    end
    infeas_val = sqrt(infeas_val)
    return infeas_val / (1 + sqrt(1/n))
end


# MAXCUT objective with GW randomized rounding
# C = -0.25*(Diagonal(G*ones(n)) - G)
function max_cut_val(C, U, Λ)
    n, R = size(U)
    hp = randn(R)
    x = sign.(U*sqrt.(Λ)*hp)
    return -dot(x, C*x)
end
