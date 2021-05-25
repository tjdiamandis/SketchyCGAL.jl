using COSMO
using SketchyCGAL
using LinearAlgebra, SparseArrays

include("utils.jl")


## Data Load
G = graph_from_file(joinpath(@__DIR__, "data/gset/G67"))
n = size(G, 1)
C = -0.25*(Diagonal(G*ones(n)) - G)


##
tt = @timed = solve_with_COSMO(C)
results = tt.value
cosmo_time = tt.time
COSMO_alloc = tt.bytes/1e6 #(MB)
