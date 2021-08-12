# SketchyCGAL

[![Build Status](https://github.com/tjdiamandis/SketchyCGAL.jl/workflows/CI/badge.svg)](https://github.com/tjdiamandis/SketchyCGAL.jl/actions)
[![Coverage](https://codecov.io/gh/tjdiamandis/SketchyCGAL.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/tjdiamandis/SketchyCGAL.jl)

A high-performance implementation of [`SketchyCGAL`](https://arxiv.org/abs/1912.02949) with weighted average gradients.

**Currently a work-in-progress**

## Use
This package solves trace-constrained SDPs of the form
```math
min     tr(CX)
s.t.    A(X) = b
        tr(X) ≦ 1
        X ⪰ 0,
```
where `A(X)` is a linear map. To define a problem, we must specify
- `C`, the cost matrix in `R^{n x n}`
- `b`, the vector in `R^d` on the RHS of the equality constraints
- `A(X)`, the linear map from `S^{n x n}` to `R^d`.
- `A_adj(y)`, the adjoint of the linear map (`R^d` to `S^{n x n}`).

We specify the maps as the non-allocating functions
```julia
function A!(y, X)
        ...
end

function A_adj!(S::SparseMatrixCSC, z)
        ...
end
```
where the first argument is modified. Note the adjoint should _add to_ instead of replace `S` (see example).

### Function Calls
**Scaling**
- `scale_X = 1/tr(X)`, so that `scale_X * tr(X) = 1`
- `scale_C = 1 / norm(C)` for numerical stability

**Optional Parameters**
- `R` is the sketch sized used in SketchyCGAL (10 or 20 works reasonably well)
- `logging = true` causes the solver to return a log of the duality gap (estimated), objective value, primal infeasibility, and time
- `logging_primal = true` causes the solver to return a log with the SDP objective after reconstruction of `X` (instead of just the implicit objective) and the reconstruction's primal infeasibility
- `compute_cut = true` causes the solver to add the MAXCUT objective (after rounding) to the primal log
- `ηt` is the step size. Defaults to `ηt=t->2.0/(t + 1.0)`
- `print_iter` controls how often the solver prints an update

We can call the **CGAL** method as follows
```julia
XT, yT = SketchyCGAL.cgal_full(
    C, b, A!, A_adj!; n=n, d=d, scale_X=scale_X, scale_C=scale_C,
    max_iters=250,
    print_iter=25
)
```

Similarly, we can call **SketchyCGAL** as follows
```julia
scgal_full(
    C, b, A!, A_adj!, A_uu!; n=n, d=d, scale_X=scale_X, scale_C=scale_C,
    max_iters=1_000,
    print_iter=100,
    R=R,
    logging=true,
    logging_primal=false,
    ηt=ηt,
)

```


### Example: MAXCUT
See the [MAXCUT example](TODO) in the `\examples` folder for full code. Assume we have an adjacency matrix `G`. The maxcut objective is

```math
min -1/4( ∑_ij G_ij*(1-xi*xj) ) = -1/4⟨diag(∑ᵢ G_ij) - G), X⟩
```
with constraints
```math
diag(X) = 1
X ⪰ 0.
```

We can construct the problem as follows
```julia
C = -0.25*(Diagonal(G*ones(n)) - G)
b = ones(n)

# Scaling variables -- so trace is bounded by 1
scale_C = 1 / norm(C)
scale_X = 1 / n

# Linear map
# AX = diag(X)
function A!(y, X)
    @views y .= X[diagind(X)]
    return nothing
end

# Adjoint
# A*z = Diagonal(z)
function A_adj!(S::SparseMatrixCSC, z)
    @views S[diagind(S)] .+= z
    return nothing
 end

```

Finally, we can call SketchyCGAL (in this case, with no logging)
```julia
soln = scgal_full(
    C, b, A!, A_adj!, A_uu!; n=n, d=d, scale_X=scale_X, scale_C=scale_C,
    max_iters=1_000,
    print_iter=100,
    R=20,
)
```


## References
- Yurtsever, A., Tropp, J. A., Fercoq, O., Udell, M., Cevher, V.,
Scalable Semidefinite Programming, SIAM Journal on Mathematics of Data Science, 3(1):171-200, 2021 ([link](https://epubs.siam.org/doi/abs/10.1137/19M1305045?mobileUi=0))
- Zhang, Y., Li, B., Giannakis, G., Accelerating Frank-Wolfe with Weighted Average Gradients, ICASSP 2021 ([link](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9414485&tag=1))


## TODOs
- [ ] Finish implementation that only uses primitives defined in the paper
- [ ] Add test cases
- [ ] Add examples
- [ ] Add example documentation
