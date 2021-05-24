function init_sketch(n, R)
    return randn(n, R), zeros(n, R)
end


# cache should have length R
function rank_one_update!(S, Ω, v, η; cache=nothing)
    S .= (1 - η) .* S
    if isnothing(cache)
        tmp = Ω'*v
        BLAS.ger!(η, v, tmp, S)
    else
        mul!(cache, Ω', v)
        BLAS.ger!(η, v, cache, S)
    end
end


# Numerical stable version of eq (5.3)
#  X̂ = S*pinv(Ω'*S)*S' = U*Λ*U'
function reconstruct(Ω, S; correction=false)
    n = size(S, 1)
    σ = sqrt(n)*eps()*norm(S)
    Sσ = S + σ .* Ω
    M = Ω'*S
    M .= 0.5 .* (M .+ M')
    F = cholesky(M)
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
