using SketchyCGAL
using LinearAlgebra, SparseArrays
using BSON
using Plots

include("utils.jl")

## Timing Figure
#load data
trials_size = BSON.load(joinpath(@__DIR__, "output/trials_size.bson"))
num_trials = length(trials_size)

trial_names = Vector{String}(undef, num_trials)
ns = zeros(num_trials)
num_nonzeros = zeros(num_trials)
time_1000_iter = zeros(num_trials)
storage = zeros(num_trials)

for (ind, trial) in enumerate(trials_size)
    trial_names[ind] = trial[1]
    ns[ind] = trial[2]["dim"].n
    num_nonzeros[ind] = trial[2]["dim"].nonzeros
    time_1000_iter[ind] = trial[2]["stats"].value.log.time_sec[end] / 2
    storage[ind] = trial[2]["stats"].bytes
end


size_plt_time_edge = plot(
    num_nonzeros,
    time_1000_iter,
    seriestype = :scatter,
    title="Iteration Time vs. Graph Edge Count",
    ylabel="Time per 1,000 Iterations (s)",
    xlabel="Number of Graph Edges",
    legend=false,
    dpi=300,
    yscale=:log,
    xscale=:log,
    markersize=6,
    xlims=(10^3.5, 10^5)
    # xticks=[10^3, 10^4, 10^5, 10^6]
)

size_plt_time_vert = plot(
    ns,
    time_1000_iter,
    seriestype = :scatter,
    title="Iteration Time vs. Graph Vertex Count",
    ylabel="Time per 1,000 Iterations (s)",
    xlabel="Number of Graph Vertices (side dimension)",
    legend=false,
    dpi=300,
    yscale=:log,
    xscale=:log,
    xlims=(10^2.5, 10^4.5),
    markersize=6
    # xticks=[10^3, 10^4, 10^5, 10^6]
)

size_plt_storage_edge = plot(
    num_nonzeros,
    storage/1e6,
    seriestype = :scatter,
    title="Storage vs. Graph Edge Count",
    ylabel="Total Storage (MB)",
    xlabel="Number of Graph Edges",
    legend=false,
    dpi=300,
    yscale=:log,
    xscale=:log,
    markersize=6,
    xlims=(10^3.5, 10^5),
    yticks=([16, 32, 64, 128], ["16MB", "32MB", "64MB", "128MB"]),
    ylim=(16, 64)
)

size_plt_storage_vert = plot(
    ns,
    storage/1e6,
    seriestype = :scatter,
    title="Storage vs. Graph Vertex Count",
    ylabel="Total Storage (MB)",
    xlabel="Number of Graph Vertices (side dimension)",
    legend=false,
    dpi=300,
    yscale=:log,
    xscale=:log,
    xlims=(10^2.5, 10^4.5),
    markersize=6,
    yticks=([16, 32, 64, 128], ["16MB", "32MB", "64MB", "128MB"]),
    ylim=(16, 64)
    # xticks=[10^3, 10^4, 10^5, 10^6]
)

savefig(size_plt_time_edge, joinpath(@__DIR__, "figures/size_time_edge"))
savefig(size_plt_time_vert, joinpath(@__DIR__, "figures/size_time_vert"))
savefig(size_plt_storage_edge, joinpath(@__DIR__, "figures/size_storage_edge"))
savefig(size_plt_storage_vert, joinpath(@__DIR__, "figures/size_storage_vert"))


## Setup data
G = graph_from_file(joinpath(@__DIR__, "data/gset/G72"))
n = size(G, 1)
C = -0.25*(Diagonal(G*ones(n)) - G)
b = ones(n)
# Scaling variables -- so trace is bounded by 1
scale_C = 1 / norm(C)
scale_X = 1 / n
# pstar = -7744.42783 * scale_X * scale_C #G67
pstar = -7808.53427 * scale_X * scale_C #G72


## R parameter figure
trials_R = BSON.load(joinpath(@__DIR__, "output/trials_R_G72.bson"))
Rs = [10, 25, 50, 100]

T = 10_000
obj_vals_iter = zeros(T, length(Rs) + 1)
p_infeas_iter = zeros(T, length(Rs) + 1)

# trials_R["iter"][10].value.log

for (ind, R) in enumerate(Rs)
    obj_vals_iter[:,ind] .= trials_R["iter"][R].value.log.obj_val_Xhat
    p_infeas_iter[:, ind] .= trials_R["iter"][R].value.log.primal_infeas_Xhat
end
obj_vals_iter[:,end] .= trials_R["iter"][10].value.log.obj_val
p_infeas_iter[:,end] .= trials_R["iter"][10].value.log.primal_infeas

labels = ["R = 10" "R = 25" "R = 50" "R = 100" "Full (implicit)"]

obj_vals_iter[obj_vals_iter .== 0] .= NaN
subopt = abs.(obj_vals_iter .- pstar) / abs(pstar)
subopt_plt = plot(
    1:T,
    subopt,
    xaxis=:log,
    yaxis=:log,
    labels=labels,
    lw=2,
    title="Suboptimality vs. Iteration",
    ylabel="Suboptimality",
    xlabel="Iteration",
    legend=:bottomleft,
    dpi=300
)

p_infeas_iter[p_infeas_iter .== 0] .= NaN
p_infeas_iter[:,end] .*= scale_X
p_infeas_plt = plot(
    1:T,
    p_infeas_iter,
    xaxis=:log,
    yaxis=:log,
    labels=["R=10" "R=25" "R=50" "R=100" "Full (implicit)"],
    lw=2,
    title="Primal Infeasibility vs. Iteration",
    ylabel="Primal Infeasibility",
    xlabel="Iteration",
    legend=:bottomleft,
    dpi=300
    # xlim=(15, T)
)


timestamp = zeros(T, length(Rs))
p_infeas_time = zeros(T, length(Rs))
for (ind, R) in enumerate(Rs)
    timestamp[:,ind] .= trials_R["time"][R].value.log.time_sec
    p_infeas_time[:, ind] .= trials_R["iter"][R].value.log.primal_infeas_Xhat
end

p_infeas_time[p_infeas_time .== 0] .= NaN
p_infeas_time_plt = plot(
    timestamp,
    p_infeas_iter,
    xaxis=:log,
    yaxis=:log,
    labels=["R=10" "R=25" "R=50" "R=100"],
    lw=2,
    title="Primal Infeasibility vs. Time",
    ylabel="Primal Infeasibility",
    xlabel="Time (s)",
    legend=:bottomleft,
    dpi=300
    # xlim=(minimum(timestamp[16:end,:]), 100)
)


subopt_time_plt = plot(
    timestamp,
    subopt,
    xaxis=:log,
    yaxis=:log,
    labels=labels,
    lw=2,
    title="Suboptimality vs. Time",
    ylabel="Suboptimality",
    xlabel="Time",
    legend=:bottomleft,
    dpi=300
    # xlim=(minimum(timestamp[16:end,:]), 100)
)

savefig(p_infeas_time_plt, joinpath(@__DIR__, "figures/infeas_R_time_g72"))
savefig(p_infeas_plt, joinpath(@__DIR__, "figures/infeas_R_iter_g72"))
savefig(subopt_time_plt, joinpath(@__DIR__, "figures/subopt_R_time_g72"))
savefig(subopt_plt, joinpath(@__DIR__, "figures/subopt_R_iter_g72"))

## Weighted gradient figures

trials_weights = BSON.load(joinpath(@__DIR__, "output/trials_weights_G72.bson"))
# long_run = BSON.load(joinpath(@__DIR__, "output/long_run.bson"))
# long_run["data"].value

δ_keys = [
    "std",
    "sa",
    "ea_0.01",
    "ea_0.05",
    "ea_0.1",
    # "ea_0.3",
    "ea_0.5",
    # "ea_0.8",
    "ea_0.9",
    # "ea_0.95",
    # "ea_0.99",
]

T = 10_000
num_trials = length(δ_keys)
obj_val = zeros(T, num_trials)
p_infeas = zeros(T, num_trials)

for (ind, k) in enumerate(δ_keys)
    obj_val[:,ind] .= trials_weights[k].value.log.obj_val
    p_infeas[:,ind] .= trials_weights[k].value.log.primal_infeas * scale_X
    ind += 1
end

# labels = [
#     "Standard",
#     "SA",
#     "EA δ = 0.01",
#     "EA δ = 0.05",
#     "EA δ = 0.1",
#     "EA δ = 0.3",
#     "EA δ = 0.5",
#     "EA δ = 0.8",
#     "EA δ = 0.9",
#     "EA δ = 0.95",
#     "EA δ = 0.99",
# ]
# labels = ["Standard" "SA" "EA δ = 0.01" "EA δ = 0.05" "EA δ = 0.1" "EA δ = 0.3" "EA δ = 0.5" "EA δ = 0.8" "EA δ = 0.9" "EA δ = 0.95" "EA δ = 0.99"]##
labels = ["Standard" "SA" "EA δ = 0.01" "EA δ = 0.05" "EA δ = 0.1" "EA δ = 0.5" "EA δ = 0.9"]##


subopt = abs.(obj_val .- pstar)/abs(pstar)
# labels = ["Standard" "EA" "EA2" "EA3" "SA" "LA" "UA"]
# linestyle = [:dash :dash :dash :dash :dash :dash]
subopt_plt = plot(
    1:T,
    subopt,
    xaxis=:log,
    yaxis=:log,
    labels=labels,
    lw=2,
    title="Suboptimality vs. Iteration",
    ylabel="Suboptimality",
    xlabel="Iteration",
    # linestyle=linestyle,
    legend=:bottomleft,
    dpi=300
)

infeas_plt = plot(
    1:T,
    p_infeas,
    xaxis=:log,
    yaxis=:log,
    labels=labels,
    lw=2,
    title="Primal Infeasibility vs. Iteration",
    ylabel="Primal Infeasibility",
    xlabel="Iteration",
    # linestyle=linestyle,
    legend=:bottomleft,
    dpi=300
)


savefig(infeas_plt, joinpath(@__DIR__, "figures/infeas_weights_g72"))
savefig(subopt_plt, joinpath(@__DIR__, "figures/subopt_weights_g72"))


##
trial_big = BSON.load(joinpath(@__DIR__, "output/big_sdp.bson"))

p_infeas_iter = trial_big["iter"].value.log.primal_infeas
p_infeas_xhat_iter = trial_big["iter"].value.log.primal_infeas_Xhat
obj_val_iter = trial_big["iter"].value.log.obj_val
cut_val_iter = trial_big["iter"].value.log.cut_value
timestamp = trial_big["time"].value.log.time_sec
obj_val_iter_xhat = trial_big["iter"].value.log.obj_val_Xhat

T = length(timestamp)
cut_val_iter[cut_val_iter.==0] .= NaN
obj_val_iter_xhat[obj_val_iter_xhat.==0] .= NaN
max_cut_plt = plot(
    1:T,
    [
        cut_val_iter,
        -obj_val_iter*1/scale_X,
        -obj_val_iter_xhat*1/scale_X
    ],
    xaxis=:log,
    # yaxis=:log,
    lw=2,
    labels=["MAXCUT" "SDP Objective Full (implicit)" "SDP Objective R = 10"],
    title="MAXCUT Value vs. Iteration",
    ylabel="MAXCUT Value (Randomized Rounding)",
    xlabel="Iteration",
    legend=:topright,
    dpi=300
)

max_cut_time_plt = plot(
    timestamp,
    [
        cut_val_iter,
        -obj_val_iter*1/scale_X,
        -obj_val_iter_xhat*1/scale_X
    ],
    xaxis=:log,
    # yaxis=:log,
    lw=2,
    labels=["MAXCUT" "SDP Objective Full (implicit)" "SDP Objective R = 10"],
    title="MAXCUT Value vs. Time",
    ylabel="MAXCUT Value (Randomized Rounding)",
    xlabel="Time (s)",
    legend=:topright,
    dpi=300
)



savefig(max_cut_plt, joinpath(@__DIR__, "figures/big_sdp_maxcut_iter"))
savefig(max_cut_time_plt, joinpath(@__DIR__, "figures/big_sdp_maxcut_time"))


##
# -obj_val_iter[end]*1/scale_X - cut_val_iter[end]
