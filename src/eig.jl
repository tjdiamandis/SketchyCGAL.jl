function approx_min_evec(M; n::Int, q::Int, cache=nothing, tol=1e-12)
    max_iters = min(q, n-1)

    V = zeros(n, max_iters+1)
    V[:,1] .= randn(n)
    V[:,1] .= V[:,1] ./ norm(V[:,1])

    p = zeros(max_iters)
    w = zeros(max_iters)

    tmp = Vector{Float64}(undef, n)
    i = 1
    while i <= max_iters
        vi = @view(V[:,i])
        vi1 = @view(V[:,i+1])


        # M!(vi1, vi)
        vi1 .= M*vi
        w[i] = dot(vi, vi1)

        vi1 .-= w[i] .* vi
        if i > 1
            vi1 .-= p[i-1] .* V[:,i-1]
        end

        p[i] = norm(vi1)
        p[i] < tol && break

        vi1 ./= p[i]
        i += 1

    end
    i -= 1

    @views T = SymTridiagonal(w[1:i], p[1:i-1])
    u = eigvecs(T)[:,1]
    v = V[:,1:i]*u
    mul!(tmp, M, v)
    ξ = dot(v, tmp)
    return ξ, v
end
