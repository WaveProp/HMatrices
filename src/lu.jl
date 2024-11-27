using DataFlowTasks

const NOPIVOT = VERSION >= v"1.7" ? NoPivot : Val{false}

const HLU = LU{<:Any,<:HMatrix}

function Base.getproperty(LU::HLU, s::Symbol)
    H = getfield(LU, :factors) # the underlying hierarchical matrix
    if s == :L
        return UnitLowerTriangular(H)
    elseif s == :U
        return UpperTriangular(H)
    else
        return getfield(LU, s)
    end
end

function Base.show(io::IO, ::MIME"text/plain", LU::HLU)
    H = getfield(LU, :factors) # the underlying hierarchical matrix
    return println(io, "LU factorization of $H")
end

"""
    lu!(M::HMatrix,comp)

Hierarhical LU facotrization of `M`, using `comp` to generate the compressed
blocks during the multiplication routines.
"""
function LinearAlgebra.lu!(M::HMatrix, compressor; threads = use_threads())
    # perform the lu decomposition of M in place
    T = eltype(M)
    nt = Threads.nthreads()
    chn = Channel{ACABuffer{T}}(nt)
    foreach(i -> put!(chn, ACABuffer(T)), 1:nt)
    if (threads)
        _lu_threads!(M, compressor, chn)
        # wrap the result in the LU structure
        d = @dspawn LU(@R(M), Int[], 0) label = "result"
        return fetch(d)
    else
        _lu!(M, compressor, chn)
        # wrap the result in the LU structure
        return LU(M, Int[], 0)
    end
end

"""
    lu!(M::HMatrix;atol=0,rank=typemax(Int),rtol=atol>0 ||
    rank<typemax(Int) ? 0 : sqrt(eps(Float64)))

Hierarhical LU facotrization of `M`, using the `PartialACA(;atol,rtol;rank)` compressor.
"""
function LinearAlgebra.lu!(
    M::HMatrix;
    atol = 0,
    rank = typemax(Int),
    rtol = atol > 0 || rank < typemax(Int) ? 0 : sqrt(eps(Float64)),
    kwargs...,
)
    compressor = PartialACA(atol, rank, rtol)
    return lu!(M, compressor; kwargs...)
end

"""
    LinearAlgebra.lu(M::HMatrix,args...;kwargs...)

Hierarchical LU factorization. See [`lu!`](@ref) for the available options.
"""
LinearAlgebra.lu(M::HMatrix, args...; kwargs...) = lu!(deepcopy(M), args...; kwargs...)

function _lu!(M::HMatrix, compressor, bufs = nothing)
    if isleaf(M)
        d = data(M)
        @assert d isa Matrix
        lu!(d, NOPIVOT())
    else
        @assert !hasdata(M)
        chdM = children(M)
        m, n = size(chdM)
        for i in 1:m
            _lu!(chdM[i, i], compressor, bufs)
            for j in (i+1):n
                ldiv!(UnitLowerTriangular(chdM[i, i]), chdM[i, j], compressor, bufs)
                rdiv!(chdM[j, i], UpperTriangular(chdM[i, i]), compressor, bufs)
            end
            for j in (i+1):m
                for k in (i+1):n
                    hmul!(chdM[j, k], chdM[j, i], chdM[i, k], -1, 1, compressor, bufs)
                end
            end
        end
    end
    return M
end

function _lu_threads!(M::HMatrix, compressor, bufs = nothing, level = 0, parent = (0, 0))
    if isleaf(M)
        @dspawn begin
            @RW(M)
            d = data(M)
            @assert d isa Matrix
            lu!(d, NOPIVOT())
        end label = "lu($(parent[1]),$(parent[2]))\nlevel=$(level)"
    else
        @assert !hasdata(M)
        chdM = children(M)
        m, n = size(chdM)
        for i in 1:m
            _lu_threads!(chdM[i, i], compressor, bufs, level + 1, (i, i))
            for j in (i+1):n
                @dspawn ldiv!(
                    UnitLowerTriangular(@R(chdM[i, i])),
                    @RW(chdM[i, j]),
                    compressor,
                    bufs,
                ) label = "ldiv($i,$j)\nlevel=$(level+1)"
                @dspawn rdiv!(
                    @RW(chdM[j, i]),
                    UpperTriangular(@R(chdM[i, i])),
                    compressor,
                    bufs,
                ) label = "rdiv($j,$i)\nlevel=$(level+1)"
            end
            for j in (i+1):m
                for k in (i+1):n
                    @dspawn hmul!(
                        @RW(chdM[j, k]),
                        @R(chdM[j, i]),
                        @R(chdM[i, k]),
                        -1,
                        1,
                        compressor,
                        bufs,
                    ) label = "hmul($j,$k)\nlevel=$(level+1)"
                end
            end
        end
    end
    return M
end

function LinearAlgebra.ldiv!(A::HLU, y::AbstractVector; global_index = true)
    p = A.factors # underlying data
    ctree = coltree(p)
    rtree = rowtree(p)
    # permute input
    global_index && permute!(y, loc2glob(ctree))
    L, U = A.L, A.U
    # solve LUx = y through:
    # (a) L(z) = y
    # (b) Ux   = z
    ldiv!(L, y)
    ldiv!(U, y)
    global_index && invpermute!(y, loc2glob(rtree))
    return y
end
