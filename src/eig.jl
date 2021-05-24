function init_cache_evec(n, max_iters)
    return (
        p = zeros(max_iters),
        w = zeros(max_iters),
        V = zeros(n, max_iters+1),
        tmp1 = Vector{Float64}(undef, n),
        tmp2 = Vector{Float64}(undef, n)
    )
end


function approx_min_evec(M; n::Int, q::Int, cache=nothing, tol=1e-12)
    max_iters = min(q, n-1)
    if isnothing(cache)
        cache = init_cache_evec(n, max_iters)
    end

    # cache.V = zeros(n, max_iters+1)
    cache.V[:,1] .= randn(n)
    cache.V[:,1] .= cache.V[:,1] ./ norm(cache.V[:,1])

    # cache.p = zeros(max_iters)
    # cache.w = zeros(max_iters)


    i = 1
    while i <= max_iters
        vi = @view(cache.V[:,i])
        vi1 = @view(cache.V[:,i+1])


        # M!(vi1, vi)
        vi1 .= M*vi
        cache.w[i] = dot(vi, vi1)

        vi1 .-= cache.w[i] .* vi
        if i > 1
            vi1 .-= cache.p[i-1] .* cache.V[:,i-1]
        end

        cache.p[i] = norm(vi1)
        cache.p[i] < tol && break

        vi1 ./= cache.p[i]
        i += 1

    end
    i -= 1

    @views T = SymTridiagonal(cache.w[1:i], cache.p[1:i-1])
    # try
    #     u = eigvecs(T)[:,1]
    #     v = V[:,1:i]*u
    #     mul!(tmp, M, v)
    #     ξ = dot(v, tmp)
    #     return ξ, v
    # catch
    #     @show T
    #     error("LAPACK error")
    # end

    # LAPACK ZLARRV for tridiagonal eigenvalues
    cache.tmp1[1:i] .= eigvecs(T)[:,1]
    mul!(cache.tmp2, @view(cache.V[:,1:i]), @view(cache.tmp1[1:i]))
    mul!(cache.tmp1, M, cache.tmp2)
    ξ = dot(v, cache.tmp1)
    return ξ, cache.tmp2

end


# Less allocating, M is a matrix
function approx_min_evec!(v, M::AbstractMatrix; n::Int, q::Int, cache=nothing, tol=1e-12)
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


        # M!(vi1, vi)
        # vi1 .= M*vi
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

    @views T = SymTridiagonal(cache.w[1:i], cache.p[1:i-1])
    # LAPACK ZLARRV for tridiagonal eigenvalues
    # TODO: make this line unallocating
    cache.tmp1[1:i] .= eigvecs(T)[:,1]
    @views mul!(v, cache.V[:,1:i], cache.tmp1[1:i])
    mul!(cache.tmp1, M, v)
    ξ = dot(v, cache.tmp1)
    return ξ

end



# Less allocating, M is a function M(ret, x): ret = M*x
function approx_min_evec!(v, M!; n::Int, q::Int, cache=nothing, tol=1e-12)
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


        M!(vi1, vi)
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

    @views T = SymTridiagonal(cache.w[1:i], cache.p[1:i-1])
    # LAPACK ZLARRV for tridiagonal eigenvalues
    # TODO: make this line unallocating
    cache.tmp1[1:i] .= eigvecs(T)[:,1]
    @views mul!(v, cache.V[:,1:i], cache.tmp1[1:i])
    M!(cache.tmp1, v)
    ξ = dot(v, cache.tmp1)
    return ξ

end
