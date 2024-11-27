using HMatrices
using LinearAlgebra
using StaticArrays
using JLD2

using CairoMakie

const FILE_PATH = joinpath(ARGS[1], "lu_results.jld2")
const PLOT_PATH = joinpath(ARGS[1], "nthreads_speedup_plot.svg")

results = load(FILE_PATH)["results"]
signequal_result = results["1"]
info = Dict{Int,Dict{Int,Float64}}()

for (nthreads, result) in results
    for (size, time) in result
        if !haskey(info, size)
            info[size] = Dict{Int,Float64}()
        end
        speedup = signequal_result[size] / time
        info[size][parse(Int64, nthreads)] = speedup
    end
end

f = Figure()
ax = Axis(
    f[1, 1];
    title = "Speedup of HLU depending on number of threads",
    xlabel = "Number of threads",
    ylabel = "Speedup",
)

for (size, speedup_info) in info
    sorted_info = sort(collect(pairs(speedup_info)))
    nthreads = [x[1] for x in sorted_info]
    speedups = [x[2] for x in sorted_info]
    lines!(ax, nthreads, speedups; label = "$size")
end

perfect_speedup = collect(keys(info[minimum(keys(info))]))  # Assuming the smallest size has all thread counts
lines!(
    ax,
    perfect_speedup,
    perfect_speedup;
    linestyle = :dash,
    linewidth = 2,
    color = :red,
    label = "Perfect speedup",
)

ax.yticks = (minimum(perfect_speedup):1:maximum(perfect_speedup))
ax.xticks = (minimum(perfect_speedup):1:maximum(perfect_speedup))

f[1, 2] = Legend(f, ax, "Size"; framevisible = false)

save(PLOT_PATH, f)

f
