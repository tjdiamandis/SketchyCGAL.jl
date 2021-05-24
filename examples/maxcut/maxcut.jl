using JuMP, MosekTools
using SketchyCGAL
using LinearAlgebra, SparseArrays

include("utils.jl")

G = graph_from_file(joinpath(@__DIR__, "data/G1"))
n = size(G, 1)

function solve_with_jump(C)
    sdp = Model(Mosek.Optimizer)
    @variable(sdp, X[1:n, 1:n] in PSDCone())
    @constraint(sdp, diag(X) .== 1)
    @objective(sdp, Min, sum(C.*X))
    optimize!(sdp)
    Xopt = value.(X)
    # Xopt[diagind(X)] .= ones(n)
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
# b .= b .* scale_X
const C_const = C * scale_C

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


## Solve JuMP
Xopt = solve_with_jump(C)
sum(C .* Xopt)

## Solve CGAL
@time XT, yT = SketchyCGAL.cgal_full(
    C, b, A!, A_adj!; n=n, d=d, scale_X=scale_X, scale_C=scale_C,
    max_iters=200,
    print_iter=25
)

sum(C .* XT * 1/scale_X)

##
R = 20
ηt(t) = 2.0/(t + 1.0)
δt(t) = 0.5
@time soln = scgal_full(
    C, b, A!, A_adj!; n=n, d=d, scale_X=scale_X, scale_C=scale_C,
    max_iters=1_000,
    print_iter=100,
    R=R,
    # logging=true,
    # logging_primal=true,
    ηt=ηt,
    δt=δt
)

Xhat = SketchyCGAL.construct_Xhat(soln)
# Xhat = S*pinv(Ω'*S)*S'
sum(C .* Xhat) * 1/scale_X

cache = zeros(n, R)
@time SketchyCGAL.compute_objective(C, soln.UT, soln.ΛT, cache=cache)# * 1/scale_X

@time SketchyCGAL.compute_primal_infeas_mc(soln.UT, soln.ΛT, cache=cache)

@time SketchyCGAL.reconstruct(soln.Ω, soln.ST; correction=true)

soln.log.obj_val_Xhat[1:10:end] * 1/scale_C * 1/scale_X

##
norm(diag(Xhat) - b) / (1 + norm(b))
norm(zT*1/scale_X - b) / (1 + norm(b))


## primatives
@time XT, yT = cgal_dense(
    C_u!, b, A!, A_adj!; n=n, d=d, scale_X=scale_X, scale_C=scale_C,
    max_iters=100,
    print_iter=25
)

sum(C .* XT * 1/scale_X)


##
cgal_dense(
    Matrix(C), b, A!, A_adj!; n=n, d=d, scale_X=scale_X, scale_C=scale_C,
    max_iters=400,
    print_iter=25
)

@time BLAS.ger!(η, v, v, Xt)

@btime begin
    Xt .-= η.*Xt
end
@time (@. Xt - η*Xt)
