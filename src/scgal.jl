# No primatives
function scgal_full(
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
        logging=false,              # stores obj_val, dual_gap, infeas
        logging_primal=false,       # computes Xhat & stores true obj, infeas
        rseed=0,
        ηt::Function=t->2.0/(t + 1.0),
        δt::Function=t->1.0,
)
    Random.seed!(0)

    # --- copy variables & scale---
    C = copy(C_input) .* scale_C
    b = copy(b_input) .* scale_X
    norm_b = norm(b_input)

    # --- parameters ---
    # Note: assume scaling factors are such that α = 1
    # α = 1
    t = 1
    β0 = 1
    K = Inf

    # --- init sketch, primal state (z), dual, values ---
    Ω, St = init_sketch(n, R)
    yt = zeros(d)
    zt = zeros(d)
    dual_gap = Inf
    primal_infeas = Inf
    obj_val = 0

    # --- cache & other memory allocations ---
    qmax = min(n-1, Int(ceil(max_iters^0.25 * log(n))))
    cache = (
        A_X = zeros(d),
        dual_update = zeros(d),
        sketch_update = zeros(R),
        evec=init_cache_evec(n, qmax),
        tmp = zeros(n)
    )
    Dt = copy(C)
    v = zeros(n)
    if logging_primal
        log_cache = zeros(n, R)
    end

    # --- logging ---
    if logging
        dual_gap_log = zeros(max_iters)
        obj_val_log = zeros(max_iters)
        primal_infeas_log = zeros(max_iters)
        time_sec_log = zeros(max_iters)
    end
    # These require a (pseudo)reconstruction of the sketch
    if logging_primal
        obj_val_Xhat_log = zeros(max_iters)
        primal_infeas_Xhat_log = zeros(max_iters)
    end

    # --- Keep track of things ---
    # (credit: https://github.com/ZIB-IOL/FrankWolfe.jl)
    headers = ["Iteration", "Primal", "Dual Gap", "Infeas", "Time"]
    print_header(headers)


    # --------------------------------------------------------------------------
    # --------------------- ITERATIONS -----------------------------------------
    # --------------------------------------------------------------------------
    time_start = time_ns()
    while t <= max_iters #&& dual_gap >= max(tol, eps())
        # --- Parameter updates (from Yurtsever et al Alg 6.1) ---
        βt = β0 * sqrt(t + 1)
        η = ηt(t)
        δ = δt(t)


        # --- Gradient computation ---
        # * Non-allocating *
        cache.A_X .= zt .- b
        cache.A_X .*= βt
        cache.A_X .+= yt
        # C entries are already in Dt -- just update diagonal
        # NOTE: This only works for MAXCUT as implemented
        for i in 1:n
            Dt[i,i] = (1.0 - δ)*Dt[i,i] + δ*(C[i,i] + cache.A_X[i])
        end


        # --- Eigenvector computation ---
        # * Custom Method *
        q = Int(ceil(t^0.25 * log(n)))
        ξ = approx_min_evec!(v, Dt, n=n, q=q, cache=cache.evec)
        # * ArnoldiMethod.jl (more stable than Lanzcos) *
        # F = partialschur(Dt, nev=1, which=SR(), tol=sqrt(n)*eps())
        # !F[2].converged && error("Eigvec computation did not converge")
        # ξ, vv = partialeigen(F[1])
        # v .= vv[:,1]


        # --- "Primal" update ---
        zt .-= η.*zt
        # TODO: use primative (3)
        #   NOTE: This only works for MAXCUT as implemented (sorry -- was lazy)
        zt .+= η .* v .* v


        # --- Sketch update ---
        # St = (1-η)*St + η*v*v'*Ω
        rank_one_update!(St, Ω, v, η; cache=cache.sketch_update)


        # --- Dual update (& primal infeasibility) ---
        cache.dual_update .= zt .- b
        # eq (6.4)
        primal_infeas = norm(cache.dual_update)
        γ = min(β0, 4β0*sqrt(t+1)/(t+1)^2 / primal_infeas^2)
        primal_infeas /= (scale_X * (1 + norm_b))
        @. yt += γ*cache.dual_update


        # --- compute objective values for output ---
        mul!(cache.tmp, C, v)
        obj_val = (1-η)*obj_val + η*dot(v, cache.tmp)


        # --- duality gap ---
        @. cache.dual_update = yt + βt*cache.dual_update
        dual_gap = obj_val + dot(cache.dual_update, cache.A_X) - ξ[1]


        # --- logging ---
        time_sec = (time_ns() - time_start) / 1e9
        if logging
            obj_val_log[t] = obj_val
            dual_gap_log[t] = dual_gap
            primal_infeas_log[t] = primal_infeas
            time_sec_log[t] = time_sec
        end
        if logging_primal && t > 15
            U, Λ = reconstruct(Ω, St, correction=true)
            obj_val_Xhat_log[t] = compute_objective(C, U, Λ; cache=log_cache)
            primal_infeas_Xhat_log[t] = compute_primal_infeas_mc(U, Λ; cache=log_cache)
        end


        # --- printing ---
        if t == 1 || t % print_iter == 0
            print_iter_func((
                string(t),
                obj_val * 1/scale_C * 1/scale_X,
                dual_gap,
                primal_infeas,
                time_sec
            ))
        end

        t += 1
    end

    # --- Prepare Output ---
    solve_time = (time_ns() - time_start) / 1e9
    @printf("\nSolved in %6.3fs\n", solve_time)
    # Reconstruct Xhat = U*Λ*U'
    U, Λ = reconstruct(Ω, St, correction=true)

    print_footer()

    # Constructs log & returns solution object
    if logging && logging_primal
        scgal_log = SCGALLog(
            dual_gap_log,
            obj_val_log,
            primal_infeas_log,
            time_sec_log,
            obj_val_Xhat_log,
            primal_infeas_Xhat_log
        )
    elseif logging
        scgal_log = SCGALLog(
            dual_gap_log,
            obj_val_log,
            primal_infeas_log,
            time_sec_log,
            nothing,
            nothing
        )
    else
        scgal_log = nothing
    end

    return SCGALSolution(U, Λ, St, Ω, yt, zt, scgal_log)

end




# TODO: better to add this as an option for the above
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Uses primatives defined in (2.4) in the paper
function scgal(
        C_u!,           # u     -> C*u
        Aadj_zu!,       # (u,z) -> A_adj(z)*u
        A_uu!,          # u     -> A(u*u')
        b_input,
        n,
        d,
        R = 30,
        scale_X = 1,
        scale_C = 1,
        max_iters = 1_000,
        tol = 1e-10,
        print_iter = 10,
)

    # Copy input
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

    # TODO: maybe better to rename to indicate size?
    qmax = min(n-1, Int(ceil(max_iters^0.25 * log(n))))
    cache = (
        A_X = zeros(d),
        dual_update = zeros(d),
        sketch_update = zeros(R),
        evec = init_cache_evec(n, qmax),
        tmp = zeros(n)
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
        # for i in 1:n
        #     Dt[i,i] = C[i,i] + cache.A_X[i]
        # end


        # --- Eigenvector computation ---
        # * Custom Method *
        q = Int(ceil(10 * t^0.25 * log(n)))
        ξ = approx_min_evec!(v, C_u!, Aadj_zu!, cache.AX, n=n, q=q, cache=cache.evec)
        # * ArnoldiMethod.jl (more stable than Lanzcos) *
        # F = partialschur(Dt, nev=1, which=SR(), tol=sqrt(n)*eps())
        # !F[2].converged && error("Eigvec computation did not converge")
        # ξ, vv = partialeigen(F[1])
        # v .= vv[:,1]


        # --- "Primal" update ---
        zt .-= η.*zt
        A_uu!(cache.dual_update, v)
        zt .+= η .* cache.dual_update


        # --- Sketch update ---
        rank_one_update!(St, Ω, v, η; cache=cache.sketch_update)


        # --- Dual update (& primal infeasibility) ---
        cache.dual_update .= zt .- b
        primal_infeas = norm(cache.dual_update) * 1/scale_X / (1 + norm_b)
        γ = min(β0, βt*η^2 / primal_infeas^2)
        @. yt += γ*cache.dual_update


        # --- compute objective values for output ---
        # TODO: make this efficient using primative 1
        C_u!(cache.tmp, v)
        obj_update = dot(v, cache.tmp)
        obj_val = (1-η)*obj_val + η*obj_update
        dual_val = dot(b, yt) * 1/scale_X


        # --- duality gap ---
        @. cache.dual_update = yt + βt*cache.dual_update
        dual_gap = obj_val + dot(cache.dual_update, cache.A_X) - ξ[1]

        # --- printing ---
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

    # --- Prepare Output ---
    solve_time = (time_ns() - time_start) / 1e9
    # Reconstruct Xhat = U*Λ*U'
    U, Λ = reconstruct(Ω, St)
    @. Λ.diag += (1.0 - tr(Λ))*ones(R) / R

    # TODO: return a soution object
    # primal var (U, Λ), dual var, AX
    return U, Λ, yt, zt

end
