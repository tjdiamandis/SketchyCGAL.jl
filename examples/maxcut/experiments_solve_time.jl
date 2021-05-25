using COSMO
using SketchyCGAL
using LinearAlgebra, SparseArrays
using BSON
using Printf

include("utils.jl")

# Run experiments single threaded
blas_threads = BLAS.set_num_threads(1)

## Data Load
function setup_maxcut_scgal(G)
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

function maxcut_scgal(G, filename, trial_data; R=10)
    trial_data[filename] = Dict()
    C, b, A!, A_adj!, n, d, scale_X, scale_C = setup_maxcut_scgal(G)

    ηt(t) = 2.0/(t + 1.0)
    δt(t) = 1.0
    tt = @timed scgal_full(
        C, b, A!, A_adj!; n=n, d=d, scale_X=scale_X, scale_C=scale_C,
        max_iters=2_000,
        print_iter=1_000,
        R=R,
        logging=true,
        # logging_primal=true,
        # ηt=ηt,
        # δt=δt
    )

    trial_data[filename]["dim"] = (
        n=n,
        nonzeros=nnz(G)
    )
    trial_data[filename]["stats"] = tt

    return nothing
end

function run_trials_size()
    filenames = readdir(joinpath(@__DIR__, "data/gset"))
    trial_data = Dict()

    for filename in filenames
        G = graph_from_file(joinpath(@__DIR__, "data/gset/$filename"))
        @printf("On %3s: %8s %12s\n", filename, size(G,1), nnz(G))
        maxcut_scgal(G, filename, trial_data)
    end
    return trial_data
end

trial_data = run_trials_size()
BSON.bson(joinpath(@__DIR__, "output/trials_size.bson"), trial_data)
