using SketchyCGAL
using LinearAlgebra, SparseArrays
using BSON

include("../utils.jl")
G = graph_from_file(joinpath(dirname(@__DIR__), "data/dimacs10/luxembourg_osm"); dimacs=true)

# Run experiments single threaded
blas_threads = BLAS.set_num_threads(1)

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

    return C, b, A!, A_adj!, n, d, scale_X, scale_C
end


C, b, A!, A_adj!, n, d, scale_X, scale_C = setup_maxcut_scgal(G)
R = 10
ηt(t) = 2.0/(t + 1.0)
δt(t) = 1.0
tt = @timed scgal_full(
    C, b, A!, A_adj!; n=n, d=d, scale_X=scale_X, scale_C=scale_C,
    max_iters=5_000,
    print_iter=1_000,
    R=R,
    logging=true,
    # logging_primal=true,
    # compute_cut=true,
    ηt=ηt,
    δt=δt
)


tt2 = @timed scgal_full(
    C, b, A!, A_adj!; n=n, d=d, scale_X=scale_X, scale_C=scale_C,
    max_iters=5_000,
    print_iter=1_000,
    R=R,
    logging=true,
    logging_primal=true,
    compute_cut=true,
    ηt=ηt,
    δt=δt
)
trial_data = Dict(
    "time" => tt,
    "iter" => tt2
)
BSON.bson(joinpath(@__DIR__, "output/big_sdp.bson"), trial_data)
