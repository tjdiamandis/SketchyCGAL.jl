function init_sketch(n, R)
    return randn(n, R), zeros(n, R)
end


# cache should have length R
function rank_one_update!(S, Ω, v, η; cache=nothing)
    S .-= η .* S
    if isnothing(cache)
        tmp = v'*Ω
        BLAS.ger!(η, v, tmp, S)
    else
        mul!(cache, Ω', v)
        BLAS.ger!(η, v, cache, S)
    end
end


# Numerical stable version of eq (5.3)
#  X̂ = S*pinv(Ω'*S)*S' = U*Λ*U'
function reconstruct(Ω, S)
    n = size(S, 1)
    σ = sqrt(n)*eps()*norm(S)
    S .+= σ.*Ω
    M = Ω'*S
    M .= 0.5 .* (M .+ M')
    F = cholesky(M)
    U, Σ, _ = svd(Sσ / F.U)
    Λ = Diagonal(max.(0, Σ.^2 .- σ))
    return U, Λ
end
