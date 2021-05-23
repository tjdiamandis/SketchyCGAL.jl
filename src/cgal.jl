function cgal(
    C,
    A!,
    A_adj!,
    b,
    α,
    n,
    d,
    max_iters=10_000,
    tol = 1e-5
)

    # Scaling variables
    C ./= norm(C)


    t = 1
    β0 = 1
    K = Inf
    dual_gap = Inf

    Xt = spzeros(n, n)
    yt = zeros(d)

    cache = (
        A_X = zeros(d),
        tmp_gap = zeros(d)
    )
    Dt = similar(Xt)

    while t <= max_iters && dual_gap >= max(tol, eps())
        βt = β0 * sqrt(t + 1)
        η = 2/(t + 1)

        A!(cache.A_X, X)
        cache.A_X .-= b
        cache.A_X .*= βt
        cache.A_X .+= yt
        A_adj!(Dt, cache.A_X)

        Dt .+= C
        ξ, v = eigs(Dt, nev=1, ncv=10, which=:LM)

        @. Xt = (1-η)*Xt + η*α*v*v'

        A!(cache.A_X, X)
        γ = min(β0, 4*α^2*βt*η^2 / norm(cache.A_X))
        @. yt += γ*(cache.A_X - b)
        obj_val = sum(C.*Xt)

        @. cache.tmp_gap = yt + βt*(cache.A_X - b)
        dual_gap = obj_val + dot(cache.tmp_gap, cache.A_X) - ξ[1]

        t % 10 == 0 && @info "Iter $t. Duality gap: $dual_gap"
        t += 1
    end

    return Xt, yt

end


function cgal_dense(
    C,
    A!,
    A_adj!,
    b,
    α,
    n,
    d,
    max_iters=1_000,
    tol = 1e-5
)

    print_iter = 10
    function print_header(data)
        @printf(
            "\n─────────────────────────────────────────────────────────────────────────────────────────────────\n"
        )
        @printf(
            "%13s %14s %14s %14s %14s %14s\n",
            data[1],
            data[2],
            data[3],
            data[4],
            data[5],
            data[6]
        )
        @printf(
            "─────────────────────────────────────────────────────────────────────────────────────────────────\n"
        )
    end

    function print_footer()
        @printf(
            "─────────────────────────────────────────────────────────────────────────────────────────────────\n\n"
        )
    end

    function print_iter_func(data)
        @printf(
            "%13s %14e %14e %14e %14e %13.1f\n",
            data[1],
            Float64(data[2]),
            Float64(data[3]),
            Float64(data[4]),
            Float64(data[5]),
            data[6]
        )
    end

    headers = ["Iteration", "Primal", "Dual", "Dual Gap", "Infeas", "Time"]
    print_header(headers)

    α = 1

    t = 1
    β0 = 1
    K = Inf
    dual_gap = Inf

    Xt = zeros(n, n)
    yt = zeros(d)

    cache = (
        A_X = zeros(d),
        dual_update = zeros(d)
    )
    Dt = similar(Xt)

    time_start = time_ns()
    while t <= max_iters && dual_gap >= max(tol, eps())
        βt = β0 * sqrt(t + 1)
        η = 2/(t + 1)

        A!(cache.A_X, Xt)
        cache.A_X .-= b
        cache.A_X .*= βt
        cache.A_X .+= yt

        Dt .= Diagonal(cache.A_X) .+ C
        # ξ, v = eigen(Dt)
        q = Int(ceil(10 * t^0.25 * log(n)))
        ξ, v = approx_min_evec(Dt, n=n, q=q)
        # F = partialschur(-Dt, nev=1, which=LM(), tol=sqrt(n)*eps())
        # ξ, v = partialeigen(F[1])
        # ξ = -ξ
        @. Xt = (1-η)*Xt + η*α*v*v'

        A!(cache.A_X, Xt)
        cache.dual_update .= cache.A_X .- b
        primal_infeas = norm(cache.dual_update)^2
        γ = min(β0, 4*α^2*βt*η^2 / primal_infeas)
        @. yt += γ*cache.dual_update
        obj_val = sum(C.*Xt)

        @. cache.dual_update = yt + βt*cache.dual_update
        dual_gap = obj_val + dot(cache.dual_update, cache.A_X) - ξ[1]

        if t == 1 || t % print_iter == 0
            print_iter_func((
                string(t),
                obj_val,
                dot(b, yt),
                dual_gap,
                primal_infeas,
                (time_ns() - time_start) / 1e9
            ))
        end
        t += 1
    end

    return Xt, yt

end
