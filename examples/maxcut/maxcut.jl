include("utils.jl")
using JuMP, MosekTools

G = graph_from_file(joinpath(@__DIR__, "data/G1"))
n = size(G, 1)
C = -0.25*(Diagonal(G*ones(n)) - G)

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



α = 1
n = size(G, 1)
d = n
C = G
b = ones(n)

# Scaling variables
C = copy(C) / norm(C)
b = copy(b) * 1/n


# Linear map
# AX = diag(X)
function A!(y, X)
    n = size(X, 1)
    for i in 1:n
        y[i] = X[i,i] / n
    end
end

# Adjoint
# A*z = Diagonal(z)
function A_adj!(S::SparseMatrixCSC, z)
    S[diagind(S)] .= z ./ size(S, 1)
 end

# primative 1: u -> C*u
function p1!(v, C, u)
    mul!(v, C, u)
end

#primative 2: (u, z) -> (A*z)u
function p2!(v, u, z)
    v .= u .* z
end

#primative 3: u -> A(uu^T)
function p3!(z, u)
    z .= u.*u
end

Xopt = solve_with_jump(C)
cgal_dense(C, A!, A_adj!, b, α, n, d)
