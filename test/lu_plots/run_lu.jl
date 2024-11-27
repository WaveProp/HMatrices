using Test
using HMatrices
using LinearAlgebra
using Random
using StaticArrays
using BenchmarkTools
using JLD2

using HMatrices: RkMatrix

include(joinpath(HMatrices.PROJECT_ROOT, "test", "testutils.jl"))

const FILE_PATH = joinpath(ARGS[1], "lu_results.jld2")

Random.seed!(1)

# 3D Laplace on sphere
problem_size = (100, 500, 1000)
threads = Threads.nthreads() > 1

results = Dict{String,Dict{Int,Float64}}()
if isfile(FILE_PATH)
    results = load(FILE_PATH)["results"]
end

thread_key = string(Threads.nthreads())
if !haskey(results, thread_key)
    results[thread_key] = Dict{Int,Float64}()
end

for size in problem_size
    println("Problem size=$size")
    m = size
    T = Float64
    X = points_on_sphere(m)
    Y = X

    K = laplace_matrix(X, X)

    splitter = CardinalitySplitter(; nmax = 50)
    Xclt = ClusterTree(X, splitter)
    Yclt = ClusterTree(Y, splitter)
    adm = StrongAdmissibilityStd(3)
    comp = PartialACA(; atol = 1e-10)

    H = assemble_hmatrix(K, Xclt, Yclt; adm, comp, threads, distributed = false)
    time = @belapsed lu($H; atol = 1e-10, threads = $threads)
    results[thread_key][size] = time
    println("Elapsed time=$time\n")
end

JLD2.@save FILE_PATH results
