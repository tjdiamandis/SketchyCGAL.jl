using SparseArrays

# Read GSET files
function graph_from_file(filename; dimacs=false)
    file_vec = readlines(filename)
    if !dimacs
        n, m = parse.(Int, split(strip(file_vec[1]), ' '))
    else
        n, _, m = parse.(Int, split(strip(file_vec[1]), ' '))
    end

    G = spzeros(Float64, n, n)
    if !dimacs
        for line in 2:length(file_vec)
            i, j, v = parse.(Int, split(strip(file_vec[line]), ' '))
            G[i, j] = G[j, i] = v
        end
    else
        for line in 2:length(file_vec)
            i, j = parse.(Int, split(strip(file_vec[line]), ' '))
            G[i, j] = G[j, i] = 1.0
        end
    end

    return G
end


# MAXCUT cost function (true -- not used in optimization)
function cost(G, y)
    return 0.25*(sum(G) - dot(y, G*y))
end


# COSMO native solve
function solve_with_COSMO(C; settings = nothing)
    n = size(C, 1)

    sdp = COSMO.Model()
    Cvec = vec_symm(C)
    M = spzeros(Int(n*(n+1)/2), n)
    q = -ones(n)
    for i in 1:n
        M[COSMO.mat_to_svec_ind(i,i), i] = -1.0
    end
    constraints = [COSMO.Constraint(M, Cvec, COSMO.PsdConeTriangle)]
    if isnothing(settings)
        settings = COSMO.Settings(
            verbose = true,
            eps_abs = 1e-5,
            eps_rel = 1e-5,
            decompose = true,
            merge_strategy = COSMO.CliqueGraphMerge,
            max_iter = 1000,
            # rho = 1e-5,
            # alpha = 1.6,
            kkt_solver = COSMO.QdldlKKTSolver,
            # kkt_solver = with_options(COSMO.MKLPardisoKKTSolver, msg_level_on = true),
            verbose_timing = true,
        )
    end
    BLAS.set_num_threads(1)
    COSMO.assemble!(
        sdp,
        spzeros(n, n),
        q,
        constraints,
        settings = settings,
        # x0 = ones(n),
    )
    results = COSMO.optimize!(sdp)
    return results
end


# Random
unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))
vec_symm(X) = X[LinearAlgebra.tril(trues(size(X)))']
