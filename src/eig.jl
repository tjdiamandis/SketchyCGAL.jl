function init_cache_evec(n, max_iters)
    return (
        p = zeros(max_iters),
        w = zeros(max_iters),
        V = zeros(n, max_iters+1),
        tmp1 = Vector{Float64}(undef, n),
        tmp2 = Vector{Float64}(undef, n)
    )
end



function approx_min_evec!(v::Vector{T}, M::AbstractMatrix; n::Int, q::Int, cache=nothing, tol=1e-12) where {T}
    max_iters = min(q, n-1)
    if isnothing(cache)
        cache = init_cache_evec(n, max_iters)
    end

    #initialize
    #QUESTION: Should this be warm started?
    @views randn!(cache.V[:,1])
    @views cache.V[:,1] .= cache.V[:,1] ./ norm(cache.V[:,1])

    i = 1
    while i <= max_iters
        vi = @view(cache.V[:,i])
        vi1 = @view(cache.V[:,i+1])

        # quad form
        mul!(vi1, M, vi)
        cache.w[i] = dot(vi, vi1)

        vi1 .-= cache.w[i] .* vi
        if i > 1
            @views vi1 .-= cache.p[i-1] .* cache.V[:,i-1]
        end

        cache.p[i] = norm(vi1)
        cache.p[i] < tol && break

        vi1 ./= cache.p[i]
        i += 1

    end
    i -= 1

    # LAPACK call for SymTridiagonal eigenvectors
    # https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.LAPACK.stegr!
    @views cache.tmp1[1:i] .= LAPACK.stegr!('V', 'I', cache.w[1:i], cache.p[1:i-1], 0.0, 0.0, 1, 1)[2][:,1]
    @views mul!(v, cache.V[:,1:i], cache.tmp1[1:i])
    mul!(cache.tmp1, M, v)
    ξ = dot(v, cache.tmp1)
    return ξ

end



# Directly uses two primatives
function approx_min_evec!(v, C_u!, Aadj_zu!, AX; n::Int, q::Int, cache=nothing, tol=1e-12)
    max_iters = min(q, n-1)
    if isnothing(cache)
        cache = init_cache_evec(n, max_iters)
    end

    #initialize
    @views randn!(cache.V[:,1])
    @views cache.V[:,1] .= cache.V[:,1] ./ norm(cache.V[:,1])

    i = 1
    while i <= max_iters
        vi = @view(cache.V[:,i])
        vi1 = @view(cache.V[:,i+1])

        C_u!(cache.tmp1, vi)
        Aadj_zu!(vi1, AX, vi)
        vi1 .+= cache.tmp1
        cache.w[i] = dot(vi, vi1)

        vi1 .-= cache.w[i] .* vi
        if i > 1
            @views vi1 .-= cache.p[i-1] .* cache.V[:,i-1]
        end

        cache.p[i] = norm(vi1)
        cache.p[i] < tol && break

        vi1 ./= cache.p[i]
        i += 1

    end
    i -= 1

    # LAPACK call for SymTridiagonal eigenvectors
    # https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.LAPACK.stegr!
    @views cache.tmp1[1:i] .= LAPACK.stegr!('V', 'I', cache.w[1:i], cache.p[1:i-1], 0.0, 0.0, 1, 1)[2][:,1]
    @views mul!(v, cache.V[:,1:i], cache.tmp1[1:i])

    C_u!(cache.tmp1, v)
    Aadj_zu!(cache.tmp2, AX, v)
    cache.tmp1 .+= cache.tmp2

    M!(cache.tmp1, v)
    ξ = dot(v, cache.tmp1)
    return ξ

end
