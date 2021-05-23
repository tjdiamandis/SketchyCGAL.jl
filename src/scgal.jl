# No primatives
function scgal_dense(
        C_input,
        b_input,
        A!,
        A_adj!;
        n,
        d,
        R = 30,
        scale_X = 1,
        scale_C = 1,
        max_iters = 1_000,
        tol = 1e-10,
        print_iter = 10,
)

    # Copy variables
    C = copy(C_input) .* scale_C
    b = copy(b_input) .* scale_X
    norm_b = norm(b_input)

    α = 1
    t = 1
    β0 = 1
    K = Inf
    dual_gap = Inf


    obj_val = 0

    Ω, St = init_sketch(n, R)
    yt = zeros(d)
    zt = zeros(d)

    cache = (
        A_X = zeros(d),
        dual_update = zeros(d),
        sketch_update = zeros(R)
    )
    Dt = copy(C)
    v = zeros(n)

    # Keep track of things (credit: https://github.com/ZIB-IOL/FrankWolfe.jl)
    headers = ["Iteration", "Primal", "Dual", "Dual Gap", "Infeas", "Time"]
    print_header(headers)
    time_start = time_ns()
    while t <= max_iters #&& dual_gap >= max(tol, eps())
        # --- Parameter updates (from Yurtsever et al Alg 6.1) ---
        βt = β0 * sqrt(t + 1)
        η = 2/(t + 1)

        # --- Gradient computation ---
        # * Non-allocating *
        cache.A_X .= zt .- b
        cache.A_X .*= βt
        cache.A_X .+= yt
        # C entries are already in Dt -- just update diagonal
        for i in 1:n
            Dt[i,i] = C[i,i] + cache.A_X[i]
        end
        # * Allocating (testing convergence) *
        # Dt = C + Diagonal(yt + βt*(diag(Xt) - b))

        # --- Eigenvector computation ---
        # * Custom Method *
        q = Int(ceil(10 * t^0.25 * log(n)))
        ξ, v = approx_min_evec(Dt, n=n, q=q)
        # * ArnoldiMethod.jl (more stable than Lanzcos) *
        # F = partialschur(Dt, nev=1, which=SR(), tol=sqrt(n)*eps())
        # !F[2].converged && error("Eigvec computation did not converge")
        # ξ, vv = partialeigen(F[1])
        # v .= vv[:,1]
        # * Standard Julia function (dense matrix) *
        # d, V = eigen(Dt)
        # ξ = d[1]
        # v = V[:,1]


        # --- Primal update ---
        zt .-= η.*zt
        # TODO: use primative (3)
        zt .+= η .* v .* v


        # --- Sketch update ---
        rank_one_update!(St, Ω, v, η; cache=cache.sketch_update)

        # --- Dual update ---
        # A!(cache.A_X, Xt)
        cache.dual_update .= zt .- b
        # cache.dual_update .= diag(Xt) - b
        primal_infeas = norm(cache.dual_update) * 1/scale_X / (1 + norm_b)
        γ = min(β0, 4*α^2*βt*η^2 / primal_infeas^2)
        @. yt += γ*cache.dual_update

        obj_val = (1-η)*obj_val + η*v'*C*v
        dual_val = dot(b, yt) * 1/scale_X

        @. cache.dual_update = yt + βt*cache.dual_update
        dual_gap = obj_val + dot(cache.dual_update, cache.A_X) - ξ[1]

        if t == 1 || t % print_iter == 0
            print_iter_func((
                string(t),
                obj_val * 1/scale_C * 1/scale_X,
                dual_val,
                dual_gap,
                primal_infeas,
                (time_ns() - time_start) / 1e9
            ))
        end
        t += 1
    end

    return St, Ω, yt, zt

end
