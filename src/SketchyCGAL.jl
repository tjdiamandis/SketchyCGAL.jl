module SketchyCGAL

using LinearAlgebra
using SparseArrays
# using Arpack
# using ArnoldiMethod
using Random
using Printf

include("utils.jl")
include("types.jl")
include("sketch.jl")
include("eig.jl")
include("cgal.jl")
include("scgal.jl")
include("scgal-maxcut.jl")

export scgal_full
export construct_Xhat

end
