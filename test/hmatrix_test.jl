using Test
using StaticArrays
using HMatrices
using LinearAlgebra
using SparseArrays
using DataFlowTasks

include(joinpath(HMatrices.PROJECT_ROOT, "test", "testutils.jl"))

@testset "Assemble" begin
    m, n = 1_000, 1_000
    X = Y = rand(SVector{3,Float64}, m)
    splitter = CardinalitySplitter(; nmax = 20)
    Xclt = Yclt = ClusterTree(X, splitter)
    adm = StrongAdmissibilityStd(; eta = 3)
    rtol = 1e-5
    comp = PartialACA(; rtol = rtol)
    # Laplace
    for threads in (true, false)
        K = laplace_matrix(X, Y)
        H = assemble_hmatrix(K, Xclt, Yclt; adm, comp, threads)
        @test norm(Matrix(K) - Matrix(H; global_index = true)) < rtol * norm(Matrix(K))
        H = assemble_hmatrix(K; threads, distributed = false)
        @test norm(Matrix(K) - Matrix(H; global_index = true)) < rtol * norm(Matrix(K))
        adjH = adjoint(H)
        H_full = Matrix(H; global_index = true)
        adjH_full = adjoint(H_full)

        @testset "getcol" begin
            Hloc = Matrix(H; global_index = false)
            @test Hloc[:, 666] ≈ HMatrices.getcol(H, 666)
            @test adjoint(Hloc)[:, 666] ≈ HMatrices.getcol(adjH, 666)
        end
        # Elastostatic
        K = elastostatic_matrix(X, Y, 1.1, 1.2)
        H = assemble_hmatrix(K, Xclt, Yclt; adm, comp, threads)
        @test norm(Matrix(K) - Matrix(H; global_index = true)) < rtol * norm(Matrix(K))
    end
end

@testset "Sparse arrays" begin
    m = 1000
    n = 1000

    X = rand(SVector{3,Float64}, m)
    Y = X
    splitter = CardinalitySplitter(; nmax = 40)
    Xclt = ClusterTree(X, splitter)
    Yclt = ClusterTree(Y, splitter)
    adm = StrongAdmissibilityStd(; eta = 3)
    rtol = 1e-5
    comp = PartialACA(; rtol = rtol)
    K = laplace_matrix(X, Y)
    H = assemble_hmatrix(K, Xclt, Yclt; adm, comp, threads = false, distributed = false)
    H_full = Matrix(H)
    T = eltype(H)
    m, n = size(H)
    S = spdiagm(0 => rand(T, n))
    Hnew = axpy!(true, S, deepcopy(H))
    @test Matrix(Hnew) == (H_full + Matrix(S))
end

@testset "Memory overlap" begin
    m = 1000
    T = Float64

    X = points_on_sphere(m)
    Y = X
    K = laplace_matrix(X, X)

    X1 = points_on_sphere(m)
    Y1 = X1
    K1 = laplace_matrix(X1, X1)

    splitter = CardinalitySplitter(; nmax = 50)
    Xclt = ClusterTree(X, splitter)
    Yclt = ClusterTree(Y, splitter)
    X1clt = ClusterTree(X1, splitter)
    Y1clt = ClusterTree(Y1, splitter)
    adm = StrongAdmissibilityStd(3)
    comp = PartialACA(; atol = 1e-10)

    H = assemble_hmatrix(K, Xclt, Yclt; adm, comp, threads = false, distributed = false)
    H1 = assemble_hmatrix(K1, X1clt, Y1clt; adm, comp, threads = false, distributed = false)

    # Test memory overlap function
    @test DataFlowTasks.memory_overlap(H, H) == true
    @test DataFlowTasks.memory_overlap(H, H1) == false
    @test DataFlowTasks.memory_overlap(H, H.children[1]) == true
end
