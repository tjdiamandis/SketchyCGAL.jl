using JuMP, MosekTools, COSMO
using SketchyCGAL
using LinearAlgebra, SparseArrays
using BSON

include("utils.jl")

G = graph_from_file(joinpath(@__DIR__, "data/gset/G1"))
n = size(G, 1)

function solve_with_jump(C; optimizer=Mosek.Optimizer)
    sdp = Model(optimizer)
    @variable(sdp, X[1:n, 1:n] in PSDCone())
    @constraint(sdp, diag(X) .== 1)
    @objective(sdp, Min, sum(C.*X))
    optimize!(sdp)
    Xopt = value.(X)

    # Optionally, we can apply a correction to enforce feasibility
    # Xopt[diagind(X)] .= ones(n)
    return Xopt
end

# Builds the LMI of the dual problem
function build_affine_exp(C, y)
    n = size(C, 1)
    aff_exp = spzeros(GenericAffExpr{Float64, VariableRef}, n, n)

    # This is a bit of a hack
    for (j, k, v) ∈ zip(findnz(C)...)
        aff_exp[j,k] = 1
    end

    for i in 1:n
        add_to_expression!(aff_exp[i,i], -y[i])
    end
    for (j, k, v) ∈ zip(findnz(C)...)
        add_to_expression!(aff_exp[j,k], C[j,k])
    end

    # hack part 2
    for (j, k, v) ∈ zip(findnz(C)...)
        aff_exp[j,k] -= 1
    end

    return aff_exp
end

function solve_dual_with_jump(C; optimizer=Mosek.Optimizer)
    sdp = Model(optimizer)
    @variable(sdp, y[1:n])
    @objective(sdp, Max, sum(y))
    A = Symmetric(build_affine_exp(C, y))
    psd_constraint = @constraint(sdp, A in PSDCone())
    optimize!(sdp)
    Xopt = value.(dual(psd_constraint))
    return Xopt
end



## ---------- Problem setup ----------
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
    @views S[diagind(S)] .+= z
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



## Solve with JuMP + COSMO
Xopt = solve_with_jump(C)

# We can also try the dual version of the problem + chordal decomposition,
#   but in this case, this approach seems much slower
# Xopt = solve_dual_with_jump(C;
#     optimizer=with_optimizer(
#         COSMO.Optimizer,
#         decompose = true,
#         merge_strategy = COSMO.CliqueGraphMerge
# ))

# Uncomment the line below to save the optimal soln
# BSON.bson("G67_Xopt", Dict("Xopt" => Xopt))

pstar_jump = sum(C .* Xopt)
@show pstar_jump

## Solve with CGAL
@time XT, yT = SketchyCGAL.cgal_full(
    C, b, A!, A_adj!; n=n, d=d, scale_X=scale_X, scale_C=scale_C,
    max_iters=250,
    print_iter=25
)

pstar_cgal = sum(C .* XT * 1/scale_X)
@show pstar_cgal

## Solve with SketchyCGAL
R = 20
ηt(t) = 2.0/(t + 1.0)
δt(t) = 1.0
tt = @timed scgal_full(
    C, b, A!, A_adj!, A_uu!; n=n, d=d, scale_X=scale_X, scale_C=scale_C,
    max_iters=1_000,
    print_iter=100,
    R=R,
    # logging=true,
    # logging_primal=true,
    ηt=ηt,
    # δt=δt
)

soln = tt.value
trial_time = tt.time
trial_alloc = tt.bytes/1e6 #(MB)

Xhat = SketchyCGAL.construct_Xhat(soln)
pstar_scgal = sum(C .* Xhat) * 1/scale_X
@show pstar_scgal

## Compute true objective (MAXCUT after rounding scheme)
cache = zeros(n, R)
@time true_obj_scgal = SketchyCGAL.compute_objective(C, soln.UT, soln.ΛT, cache=cache)# * 1/scale_X
@show true_obj_scgal

## Compute primal infeasibility
@time p_infeas_scgal = SketchyCGAL.compute_primal_infeas_mc(soln.UT, soln.ΛT, cache=cache)
@show p_infeas_scgal
