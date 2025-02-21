using Test
using HMatrices
using LinearAlgebra
using Random
using StaticArrays

using HMatrices: RkMatrix
using HMatrices: ITerm
include(joinpath(HMatrices.PROJECT_ROOT, "test", "testutils.jl"))

Random.seed!(1)

m = 2000
n = 2000

X = rand(SVector{3,Float64}, m)
Y = [rand(SVector{3,Float64}) for _ in 1:n]
splitter = CardinalitySplitter(; nmax = 50)
Xclt = ClusterTree(X, splitter)
Yclt = ClusterTree(Y, splitter)
adm = StrongAdmissibilityStd(; eta = 3)
atol = 1e-5
comp = PartialACA(; atol)
K = laplace_matrix(X, Y)
H = assemble_hmatrix(K, Xclt, Yclt; adm, comp, threads = false, distributed = false)

H_full = Matrix(H; global_index = false)

R = HMatrices.RkMatrix(rand(m, 10), rand(n, 10))
P = HMatrices.RkMatrix(rand(m, 10), rand(n, 10))

α = rand() - 0.5
β = rand() - 0.5

@testset "hmul!" begin
    C = deepcopy(H)
    tmp = β * H_full + α * H_full * H_full
    HMatrices.hmul!(C, H, H, α, β, PartialACA(; atol = 1e-6))
    @test Matrix(C; global_index = false) ≈ tmp
    # adjoint
    C = deepcopy(H)
    tmp = β * H_full + α * adjoint(H_full) * H_full
    HMatrices.hmul!(C, adjoint(H), H, α, β, PartialACA(; atol = 1e-6))
    @test Matrix(C; global_index = false) ≈ tmp
end

@testset "gemv" begin
    T = eltype(H)
    m, n = size(H)
    x = rand(T, n)
    y = rand(T, m)

    @testset "serial" begin
        exact = β * y + α * H_full * x
        approx = mul!(copy(y), H, x, α, β; threads = false, global_index = false)
        @test exact ≈ approx
        exact = β * y + α * Matrix(H; global_index = true) * x
        approx = mul!(copy(y), H, x, α, β; threads = false, global_index = true)
        @test exact ≈ approx

        # multiply by adjoint
        adjH = adjoint(deepcopy(H))
        exact = β * y + α * adjoint(H_full) * x
        approx = mul!(copy(y), adjH, x, α, β; threads = false, global_index = false)
        @test exact ≈ approx
        exact = β * y + α * adjoint(Matrix(H; global_index = true)) * x
        approx = mul!(copy(y), adjH, x, α, β; threads = false, global_index = true)
        @test exact ≈ approx
    end

    @testset "threads" begin
        exact = β * y + α * H_full * x
        approx = mul!(copy(y), H, x, α, β; threads = true, global_index = false)
        @test exact ≈ approx
        exact = β * y + α * Matrix(H; global_index = true) * x
        approx = mul!(copy(y), H, x, α, β; threads = false, global_index = true)
        @test exact ≈ approx

        # multiply by adjoint
        adjH = adjoint(deepcopy(H))
        exact = β * y + α * adjoint(H_full) * x
        approx = mul!(copy(y), adjH, x, α, β; threads = true, global_index = false)
        @test exact ≈ approx
        exact = β * y + α * adjoint(Matrix(H; global_index = true)) * x
        approx = mul!(copy(y), adjH, x, α, β; threads = true, global_index = true)
        @test exact ≈ approx
    end

    @testset "exact vs inexact HMatrix-vector products" begin
        h_term_test = ITerm(H,1.0e-5)

        #tests with and without global indexes

        exact = mul!(copy(y), H, x, α, β; threads = false, global_index = false)
        #approx= mul!(copy(y), H, x, α, β,1.0e-5; threads = false, global_index = false)
        approx= mul!(copy(y), h_term_test, x, α, β; threads = false, global_index = false)
        @test isapprox(exact,approx;rtol=1.0e-5)
        exact = mul!(copy(y), H, x, α, β;threads=false)  
        #approx = mul!(copy(y), H, x, α, β,1e-5;threads=false)
        approx= mul!(copy(y), h_term_test, x, α, β; threads = false)
        @test isapprox(exact,approx;rtol=1.0e-5)

        #no mentinon to threads, so it's on by default
        exact = mul!(copy(y), H, x, α, β;  global_index = false)
        #approx= mul!(copy(y), H, x, α, β,1.0e-5; global_index = false)
        approx= mul!(copy(y), h_term_test, x, α, β;global_index = false)
        @test isapprox(exact,approx;rtol=1.0e-5)

        exact = mul!(copy(y), H, x, α, β)   
        #approx = mul!(copy(y), H, x, α, β,1e-5)
        approx= mul!(copy(y), h_term_test, x, α, β)
        @test isapprox(exact,approx;rtol=1.0e-5)

        #last test checks that if the desired tolerance is to small, we'll use the full low rank matrix instead
        exact = mul!(copy(y), H, x, α, β)
        h_term_test.rtol=0.0
        approx = mul!(copy(y), h_term_test, x, α, β)
        @test isapprox(exact,approx;rtol=1.0e-5)
    end

    @testset "hermitian" begin
        threads = false
        for threads in (true, false)
            K = laplace_matrix(X, X)
            Hsym = assemble_hmatrix(Hermitian(K), Xclt, Xclt; adm, comp, threads)
            H = assemble_hmatrix(K, Xclt, Xclt; adm, comp, threads)
            x = rand(n)
            y1 = mul!(zero(x), H, x, 1, 0; threads)
            y2 = mul!(zero(x), Hsym, x, 1, 0; threads)
            @test y1 ≈ y2
        end
    end
end

@testset "gemm" begin
    α = rand() - 0.5
    β = rand() - 0.5
    T = eltype(H)
    m, n = size(H)
    k = 10
    x = rand(T, n, k)
    y = rand(T, m, k)

    @testset "serial" begin
        exact = β * y + α * H_full * x
        approx = mul!(copy(y), H, x, α, β; threads = false, global_index = false)
        @test exact ≈ approx
        exact = β * y + α * Matrix(H; global_index = true) * x
        approx = mul!(copy(y), H, x, α, β; threads = false, global_index = true)
        @test exact ≈ approx
    end

    @testset "threads" begin
        exact = β * y + α * H_full * x
        approx = mul!(copy(y), H, x, α, β; threads = true, global_index = false)
        @test exact ≈ approx
        exact = β * y + α * Matrix(H; global_index = true) * x
        approx = mul!(copy(y), H, x, α, β; threads = false, global_index = true)
        @test exact ≈ approx
    end


end
