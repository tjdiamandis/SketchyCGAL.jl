using COSMO
using SketchyCGAL
using LinearAlgebra, SparseArrays
using BSON

include("../utils.jl")

blas_threads = BLAS.get_num_threads()
# Run experiments single threaded
blas_threads = BLAS.set_num_threads(1)

## Data Load
G = graph_from_file(joinpath(dirname(@__DIR__), "data/gset/G67"))
# n = size(G, 1)
# C = -0.25*(Diagonal(G*ones(n)) - G)


## COSMO (try to get true opt)
cosmo_tt = @timed solve_with_COSMO(C; settings = COSMO.Settings(
    verbose = true,
    eps_abs = 1e-5,
    eps_rel = 1e-5,
    decompose = true,
    merge_strategy = COSMO.CliqueGraphMerge,
    max_iter = 5_000,
    # rho = 1e-5,
    # alpha = 1.6,
    kkt_solver = COSMO.QdldlKKTSolver,
    # kkt_solver = with_options(COSMO.MKLPardisoKKTSolver, msg_level_on = true),
    verbose_timing = true,
))
cosmo_results = tt.value
cosmo_time = tt.time
cosmo_alloc = tt.bytes/1e6 #(MB)


## SCGAL (run for awhile to get "true" opt)
## ---------- Problem setup ----------
function setup_maxcut_scgal(G::SparseMatrixCSC)
    # Parameters
    n = size(G, 1)
    d = n

    # Data
    # objective = min -1/4( ∑_ij w_ij*(1-yi*yj) ) = -1/4⟨diag(∑ᵢ w_ij) - W), Y⟩
    # note that this reformulation uses the fact that Tr(Y) = 1
    C = -0.25*(Diagonal(G*ones(n)) - G)
    b = ones(n)

    # Scaling variables -- so trace is bounded by 1
    scale_C = 1 / norm(C)
    scale_X = 1 / n
    # b .= b .* scale_X
    # const C_const = C * scale_C

    # Linear map
    # AX = diag(X)
    function A!(y, X)
        n = size(X, 1)
        for i in 1:n
            y[i] = X[i,i]
        end
        return nothing
    end

    # Adjoint
    # A*z = Diagonal(z)
    function A_adj!(S::SparseMatrixCSC, z)
        for (i, j, v) ∈ zip(findnz(S)...)
            S[i,j] = 0.0
        end
        S[diagind(S)] .= z
        return nothing
     end

    # primative 1: u -> C*u
    function C_u!(v, u)
        mul!(v, C_const, u)
    end

    #primative 2: (u, z) -> (A*z)u
    function Aadj_zu!(v, u, z)
        v .= u .* z
    end

    #primative 3: u -> A(uu^T)
    function A_uu!(z, u)
        z .= u.*u
    end

    return C, b, A!, A_adj!, n, d, scale_X, scale_C
end


## Long Run
function maxcut_scgal_long(G)
    C, b, A!, A_adj!, n, d, scale_X, scale_C = setup_maxcut_scgal(G)
    R = 10
    ηt(t) = 2.0/(t + 1.0)
    δt(t) = 1.0
    tt = @timed scgal_full(
        C, b, A!, A_adj!; n=n, d=d, scale_X=scale_X, scale_C=scale_C,
        max_iters=100_000,
        print_iter=1_000,
        R=R,
        # logging=true,
        # logging_primal=true,
        ηt=ηt,
        δt=δt
    )

    # soln = tt.value
    # trial_time = tt.time
    # trial_alloc = tt.bytes/1e6 #(MB)

    return tt
end

BLAS.set_num_threads(1)
long_run = maxcut_scgal_long(G)
BSON.bson(joinpath(@__DIR__, "output/long_run.bson"), Dict("data" => long_run))
# 100000  -7.743749e+03   5.253496e-01   7.271706e-05      1067.732


## SCGAL Trials for R, Fig 7.1 Recreation
Rs = [10; 25; 50; 100]
function run_trials_R(G, Rs)
    trial_data = Dict("time" => Dict(), "iter" => Dict())
    C, b, A!, A_adj!, n, d, scale_X, scale_C = setup_maxcut_scgal(G)
    ηt(t) = 2.0/(t + 1.0)
    δt(t) = 1.0

    for R in Rs
        tt = @timed scgal_full(
            C, b, A!, A_adj!; n=n, d=d, scale_X=scale_X, scale_C=scale_C,
            max_iters=10_000,
            print_iter=1_000,
            R=R,
            logging=true,
            # logging_primal=true,
            ηt=ηt,
            δt=δt
        )
        trial_data["time"][R] = tt

        tt2 = @timed scgal_full(
            C, b, A!, A_adj!; n=n, d=d, scale_X=scale_X, scale_C=scale_C,
            max_iters=10_000,
            print_iter=1_000,
            R=R,
            logging=true,
            logging_primal=true,
            ηt=ηt,
            δt=δt
        )
        trial_data["iter"][R] = tt2

    end
    return trial_data
end

trial_data = run_trials_R(G, Rs)
BSON.bson(joinpath(@__DIR__, "output/trials_R.bson"), trial_data)


## SCGAL trials for weight (fix R = 50)
function run_trials_weights(G; R=10, max_iters=10_000, print_iter=1000, rseed=0)
    trial_data = Dict()
    C, b, A!, A_adj!, n, d, scale_X, scale_C = setup_maxcut_scgal(G)

    tt = @timed scgal_full(
        C, b, A!, A_adj!; n=n, d=d, scale_X=scale_X, scale_C=scale_C,
        max_iters=max_iters,
        print_iter=print_iter,
        R=R,
        logging=true,
        # logging_primal=true,
        ηt=t->2.0/(t + 1.0),
        δt=t->1.0,
        rseed=rseed
    )
    trial_data["std"] = tt

    tt_sa = @timed scgal_full(
        C, b, A!, A_adj!; n=n, d=d, scale_X=scale_X, scale_C=scale_C,
        max_iters=max_iters,
        print_iter=print_iter,
        R=R,
        logging=true,
        # logging_primal=true,
        ηt=t->2.0/(t + 1.0),
        δt=t->1.0t/(1.0t + 1),
        rseed=rseed
    )
    trial_data["sa"] = tt_sa

    tt_ea = @timed scgal_full(
        C, b, A!, A_adj!; n=n, d=d, scale_X=scale_X, scale_C=scale_C,
        max_iters=max_iters,
        print_iter=print_iter,
        R=R,
        logging=true,
        # logging_primal=true,
        ηt=t->2.0/(t + 1.0),
        δt=t->0.8,
        rseed=rseed
    )
    trial_data["ea"] = tt_ea


    tt_ea2 = @timed scgal_full(
        C, b, A!, A_adj!; n=n, d=d, scale_X=scale_X, scale_C=scale_C,
        max_iters=max_iters,
        print_iter=print_iter,
        R=R,
        logging=true,
        # logging_primal=true,
        ηt=t->2.0/(t + 1.0),
        δt=t->0.5,
        rseed=rseed
    )
    trial_data["ea2"] = tt_ea2

    tt_ea2 = @timed scgal_full(
        C, b, A!, A_adj!; n=n, d=d, scale_X=scale_X, scale_C=scale_C,
        max_iters=max_iters,
        print_iter=print_iter,
        R=R,
        logging=true,
        # logging_primal=true,
        ηt=t->2.0/(t + 1.0),
        δt=t->0.3,
        rseed=rseed
    )
    trial_data["ea3"] = tt_ea2


    tt_la = @timed scgal_full(
        C, b, A!, A_adj!; n=n, d=d, scale_X=scale_X, scale_C=scale_C,
        max_iters=max_iters,
        print_iter=print_iter,
        R=R,
        logging=true,
        # logging_primal=true,
        ηt=t->2.0/(t + 1.0),
        δt=t->2.0/(t + 1.0),
        rseed=rseed
    )
    trial_data["la"] = tt_la

    tt_ua = @timed scgal_full(
        C, b, A!, A_adj!; n=n, d=d, scale_X=scale_X, scale_C=scale_C,
        max_iters=max_iters,
        print_iter=print_iter,
        R=R,
        logging=true,
        # logging_primal=true,
        ηt=t->2.0/(t + 1.0),
        δt=t->1.0/t,
        rseed=rseed
    )
    trial_data["ua"] = tt_ua

    return trial_data
end

trial_data_weights = run_trials_weights(G; max_iters=10_000, print_iter=1_000, rseed=1)
BSON.bson(joinpath(@__DIR__, "output/trials_weights_short.bson"), trial_data_weights)
