"""
    struct MulLinearOp{R,T} <: AbstractMatrix{T}

Abstract matrix representing the following linear operator:
```
    L = R + P + a * ∑ᵢ Aᵢ * Bᵢ
```
where `R` and `P` are of type `RkMatrix{T}`, `Aᵢ,Bᵢ` are of type
`HMatrix{R,T}` and `a` is scalar multiplier. Calling `compressor(L)` produces a low-rank approximation of
`L`, where `compressor` is an [`AbstractCompressor`](@ref).

Note: this structure is used to group the operations required when multiplying
hierarchical matrices so that they can later be executed in a way that minimizes
recompression of intermediate computations.
"""
struct MulLinearOp{R,T,S} <: AbstractMatrix{T}
    R::Union{RkMatrix{T},Nothing}
    P::Union{RkMatrixBlockView{T},Nothing}
    # P::Union{RkMatrix{T},Nothing}
    pairs::Vector{NTuple{2,HMatrix{R,T}}}
    multiplier::S
end

# AbstractMatrix interface
function Base.size(L::MulLinearOp)
    isnothing(L.R) || (return size(L.R))
    isnothing(L.P) || (return size(L.P))
    isempty(L.pairs) && (return (0, 0))
    A, B = first(L.pairs)
    return (size(A, 1), size(B, 2))
end

function Base.getindex(L::Union{MulLinearOp,Adjoint{<:Any,<:MulLinearOp}}, i::Int, j::Int)
    if ALLOW_GETINDEX[]
        getcol(L, j)[i]
    else
        error("calling `getindex(L,i,j)` of a `MulLinearOp` has been disabled")
    end
end

"""
    hmul!(C::HMatrix,A::HMatrix,B::HMatrix,a,b,compressor)

Similar to `mul!` : compute `C <-- A*B*a + B*b`, where `A,B,C` are hierarchical
matrices and `compressor` is a function/functor used in the intermediate stages
of the multiplication to avoid growing the rank of admissible blocks after
addition is performed.
"""
function hmul!(C::T, A::T, B::T, a, b, compressor) where {T<:HMatrix}
    # create a plan
    dict = Dict{T,Vector{NTuple{2,T}}}()
    _plan_dict!(dict, C, A, B)
    # execute the plan
    b == true || rmul!(C, b)
    return _hmul!(C, compressor, dict, a, true)
end

function _hmul!(C::T, compressor, dict, a, root) where {T<:HMatrix}
    pairs = get(dict, C, Tuple{T,T}[])
    # update the data in C if needed
    @dspawn begin
        @RW C
        @R parent(C)
        @R pairs
        R = _parent_data_restriction(C,root)
        if !isnothing(R) || !isempty(pairs)
            if isleaf(C) && !isadmissible(C)
                d = data(C)
                for (A, B) in pairs
                    _mul_dense!(d, A, B, a)
                end
                isnothing(R) || axpy!(true, R, d)
            else
                L = MulLinearOp(data(C), R, pairs, a)
                R = compressor(L)
                setdata!(C, R)
            end
        end
    end label="hmul"
    # move to children
    for chd in children(C)
        _hmul!(chd, compressor, dict, a, false)
    end
    isleaf(C) || @dspawn setdata!(@W(C), nothing) label="clear parent"
    return C
end

function _plan_dict!(dict, C::T, A::T, B::T) where {T<:HMatrix}
    pairs = get!(dict, C, Tuple{T,T}[])
    if isleaf(A) || isleaf(B) || isleaf(C)
        push!(pairs, (A, B))
    else
        ni, nj = blocksize(C)
        _, nk = blocksize(A)
        A_children = children(A)
        B_children = children(B)
        C_children = children(C)
        for i in 1:ni
            for j in 1:nj
                for k in 1:nk
                    _plan_dict!(dict, C_children[i, j], A_children[i, k], B_children[k, j])
                end
            end
        end
    end
    return dict
end

function _parent_data_restriction(C,root)
    P  = parent(C)
    Rp = data(P)
    if root || isnothing(Rp)
        return nothing
    else
        # compute indices of C relative to its parent
        shift = pivot(P) .- 1
        irange = rowrange(C) .- shift[1]
        jrange = colrange(C) .- shift[2]
        return view(Rp, irange, jrange)
        # return RkMatrix(Rp.A[irange, :], Rp.B[jrange, :])
    end
end

# disable `mul` of hierarchial matrices
function mul!(C::HMatrix, A::HMatrix, B::HMatrix, a::Number, b::Number)
    msg = "use `hmul!` to multiply hierarchical matrices"
    return error(msg)
end

Base.getindex(L::MulLinearOp, ::Colon, j::Int) = getcol(L, j)
Base.getindex(L::Adjoint{<:Any,<:MulLinearOp}, ::Colon, j::Int) = getcol(L, j)

function getcol!(col, L::MulLinearOp, j)
    m, n = size(L)
    T = eltype(L)
    # compute j-th column of ∑ Aᵢ Bᵢ
    for (A, B) in L.pairs
        m, k = size(A)
        k, n = size(B)
        tmp = zeros(T, k)
        jg = j + offset(B)[2] # global index on hierarchical matrix B
        getcol!(tmp, B, jg)
        _hgemv_recursive!(col, A, tmp, offset(A))
    end
    # multiply the columns by a
    a = L.multiplier
    rmul!(col, a)
    # add R and P (if they exist)
    R = L.R
    if !isnothing(R)
        getcol!(col, R, j, Val(true))
    end
    P = L.P
    if !isnothing(P)
        getcol!(col, P, j, Val(true))
    end
    return col
end

function getcol!(col, adjL::Adjoint{<:Any,<:MulLinearOp}, j)
    L = parent(adjL)
    T = eltype(L)
    # compute j-th column of ∑ adjoint(Bᵢ)*adjoint(Aᵢ)
    for (A, B) in L.pairs
        At, Bt = adjoint(A), adjoint(B)
        tmp = zeros(T, size(At, 1))
        jg = j + offset(At)[2]
        getcol!(tmp, At, jg)
        _hgemv_recursive!(col, Bt, tmp, offset(Bt))
    end
    # multiply by a
    a = L.multiplier
    rmul!(col, conj(a))
    # add the j-th column of Ct if it has data
    R = L.R
    if !isnothing(R)
        getcol!(col, adjoint(R), j, Val(true))
    end
    P = L.P
    if !isnothing(P)
        getcol!(col, adjoint(P), j, Val(true))
    end
    return col
end

#=
Multiplication when the target is a dense matrix. The numbering system in the following
`_mulxyz` methods use the following convention
1 --> Matrix (dense)
2 --> RkMatrix (sparse)
3 --> HMatrix (hierarchical)
=#

function _mul_dense!(C::Matrix, A, B, a)
    Adata = isleaf(A) ? A.data : A
    Bdata = isleaf(B) ? B.data : B
    if Adata isa HMatrix
        if Bdata isa Matrix
            _mul131!(C, Adata, Bdata, a)
        elseif Bdata isa RkMatrix
            _mul132!(C, Adata, Bdata, a)
        end
    elseif Adata isa Matrix
        if Bdata isa Matrix
            _mul111!(C, Adata, Bdata, a)
        elseif Bdata isa RkMatrix
            _mul112!(C, Adata, Bdata, a)
        elseif Bdata isa HMatrix
            _mul113!(C, Adata, Bdata, a)
        end
    elseif Adata isa RkMatrix
        if Bdata isa Matrix
            _mul121!(C, Adata, Bdata, a)
        elseif Bdata isa RkMatrix
            _mul122!(C, Adata, Bdata, a)
        elseif Bdata isa HMatrix
            _mul123!(C, Adata, Bdata, a)
        end
    end
end

function _mul111!(C::Union{Matrix,SubArray,Adjoint},
                  A::Union{Matrix,SubArray,Adjoint},
                  B::Union{Matrix,SubArray,Adjoint},
                  a::Number)
    return mul!(C, A, B, a, true)
end

function _mul112!(C::Union{Matrix,SubArray,Adjoint},
                  M::Union{Matrix,SubArray,Adjoint},
                  R::RkMatrix,
                  a::Number)
    buffer = M * R.A
    _mul111!(C, buffer, R.Bt, a)
    return C
end

function _mul113!(C::Union{Matrix,SubArray,Adjoint},
                  M::Union{Matrix,SubArray,Adjoint},
                  H::HMatrix,
                  a::Number)
    T = eltype(C)
    if hasdata(H)
        mat = data(H)
        if mat isa Matrix
            _mul111!(C, M, mat, a)
        elseif mat isa RkMatrix
            _mul112!(C, M, mat, a)
        else
            error()
        end
    end
    for child in children(H)
        shift = pivot(H) .- 1
        irange = rowrange(child) .- shift[1]
        jrange = colrange(child) .- shift[2]
        Cview = @views C[:, jrange]
        Mview = @views M[:, irange]
        _mul113!(Cview, Mview, child, a)
    end
    return C
end

function _mul121!(C::Union{Matrix,SubArray,Adjoint},
                  R::RkMatrix,
                  M::Union{Matrix,SubArray,Adjoint},
                  a::Number)
    buffer = R.Bt * M
    return _mul111!(C, R.A, buffer, a)
end

function _mul122!(C::Union{Matrix,SubArray,Adjoint}, R::RkMatrix, S::RkMatrix, a::Number)
    if rank(R) < rank(S)
        _mul111!(C, R.A, (R.Bt * S.A) * S.Bt, a)
    else
        _mul111!(C, R.A * (R.Bt * S.A), S.Bt, a)
    end
    return C
end

function _mul123!(C::Union{Matrix,SubArray,Adjoint}, R::RkMatrix, H::HMatrix, a::Number)
    T = promote_type(eltype(R), eltype(H))
    tmp = zeros(T, size(R.Bt, 1), size(H, 2))
    _mul113!(tmp, R.Bt, H, 1)
    _mul111!(C, R.A, tmp, a)
    return C
end

function _mul131!(C::Union{Matrix,SubArray,Adjoint},
                  H::HMatrix,
                  M::Union{Matrix,SubArray,Adjoint},
                  a::Number)
    if isleaf(H)
        mat = data(H)
        if mat isa Matrix
            _mul111!(C, mat, M, a)
        elseif mat isa RkMatrix
            _mul121!(C, mat, M, a)
        else
            error()
        end
    end
    for child in children(H)
        shift = pivot(H) .- 1
        irange = rowrange(child) .- shift[1]
        jrange = colrange(child) .- shift[2]
        Cview = view(C, irange, :)
        Mview = view(M, jrange, :)
        _mul131!(Cview, child, Mview, a)
    end
    return C
end

function _mul132!(C::Union{Matrix,SubArray,Adjoint}, H::HMatrix, R::RkMatrix, a::Number)
    T = promote_type(eltype(H), eltype(R))
    buffer = zeros(T, size(H, 1), size(R.A, 2))
    _mul131!(buffer, H, R.A, 1)
    _mul111!(C, buffer, R.Bt, a)
    return C
end

############################################################################################
# Specializations on gemv:
# The routines below provide specialized version of mul!(C,A,B,a,b) when `A` and
# `B` are vectors
############################################################################################

# 1.2.1
function mul!(y::AbstractVector, R::RkMatrix, x::AbstractVector, a::Number, b::Number)
    # tmp = R.Bt*x
    tmp = mul!(buffer(R), adjoint(R.B), x)
    return mul!(y, R.A, tmp, a, b)
end

# 1.2.1
function mul!(y::AbstractVector,
              adjR::Adjoint{<:Any,<:RkMatrix},
              x::AbstractVector,
              a::Number,
              b::Number)
    R = parent(adjR)
    # tmp = R.At*x
    tmp = mul!(buffer(R), adjoint(R.A), x)
    return mul!(y, R.B, tmp, a, b)
end

# 1.3.1
"""
    mul!(y::AbstractVector,H::HMatrix,x::AbstractVector,a,b[;global_index,threads])

Perform `y <-- H*x*a + y*b` in place.
"""
function mul!(y::AbstractVector,
              A::HMatrix,
              x::AbstractVector,
              a::Number=1,
              b::Number=0;
              global_index=use_global_index(),
              threads=use_threads())
    # since the HMatrix represents A = inv(Pr)*H*Pc, where Pr and Pc are row and column
    # permutations, we need first to rewrite C <-- b*C + a*(inv(Pr)*H*Pc)*B as
    # C <-- inv(Pr)*(b*Pr*C + a*H*(Pc*B)). Following this rewrite, the
    # multiplication is performed by first defining B <-- Pc*B, and C <--
    # Pr*C, doing the multiplication with the permuted entries, and then
    # permuting the result  back C <-- inv(Pr)*C at the end.
    ctree = coltree(A)
    rtree = rowtree(A)
    # permute input
    if global_index
        x = x[colperm(A)]
        y = permute!(y, rowperm(A))
        rmul!(x, a) # multiply in place since this is a new copy, so does not mutate exterior x
    elseif a != 1
        x = a * x # new copy of x since we should not mutate the external x in mul!
    end
    iszero(b) ? fill!(y, zero(eltype(y))) : rmul!(y, b)
    # offset in case A is not indexed starting at (1,1); e.g. A is not the root
    # of and HMatrix
    offset = pivot(A) .- 1
    if threads
        # if a partition of the leaves does not already exist, create one. By
        # default a `hilbert_partition` is created
        # TODO: test the various threaded implementations and chose one.
        # Currently there are two main choices:
        # 1. spawn a task per leaf, and let julia scheduler handle the tasks
        # 2. create a static partition of the leaves and try to estimate the
        #    cost, then spawn one task per block of the partition. In this case,
        #    test if the hilbert partition is really faster than col_partition
        #    or row_partition
        #    Right now the hilbert partition is chosen by default without proper
        #    testing.
        @timeit_debug "partitioning leaves" begin
            haskey(CACHED_PARTITIONS, A) ||
                hilbert_partition(A, Threads.nthreads(), _cost_gemv)
            # haskey(CACHED_PARTITIONS,A) || col_partition(A,Threads.nthreads(),_cost_gemv)
            # haskey(CACHED_PARTITIONS,A) || row_partition(A,Threads.nthreads(),_cost_gemv)
        end
        @timeit_debug "threaded multiplication" begin
            p = CACHED_PARTITIONS[A]
            _hgemv_static_partition!(y, x, p.partition, offset)
            # _hgemv_threads!(y,x,p.partition,offset)  # threaded implementation
        end
    else
        @timeit_debug "serial multiplication" begin
            _hgemv_recursive!(y, A, x, offset) # serial implementation
        end
    end
    # permute output
    global_index && invpermute!(y, rowperm(A))
    return y
end

"""
    _hgemv_recursive!(C,A,B,offset)

Internal function used to compute `C[I] <-- C[I] + A*B[J]` where `I =
rowrange(A) - offset[1]` and `J = rowrange(B) - offset[2]`.

The `offset` argument is used on the caller side to signal if the original
hierarchical matrix had a `pivot` other than `(1,1)`.
"""
function _hgemv_recursive!(C::AbstractVector, A::Union{HMatrix,Adjoint{<:Any,<:HMatrix}},
                           B::AbstractVector, offset)
    T = eltype(A)
    if isleaf(A)
        irange = rowrange(A) .- offset[1]
        jrange = colrange(A) .- offset[2]
        d = data(A)
        if T <: SMatrix
            # FIXME: there is bug with gemv and static arrays, so we convert
            # them to matrices of n × 1
            # (https://github.com/JuliaArrays/StaticArrays.jl/issues/966#issuecomment-943679214).
            mul!(view(C, irange, 1:1), d, view(B, jrange, 1:1), 1, 1)
        else
            # C and B are the "global" vectors handled by the caller, so a view
            # is needed.
            mul!(view(C, irange), d, view(B, jrange), 1, 1)
        end
    else
        for block in children(A)
            _hgemv_recursive!(C, block, B, offset)
        end
    end
    return C
end

function _hgemv_threads!(C::AbstractVector, B::AbstractVector, partition, offset)
    nt = Threads.nthreads()
    # make `nt` copies of C and run in parallel
    Cthreads = [zero(C) for _ in 1:nt]
    @sync for p in partition
        for block in p
            Threads.@spawn begin
                id = Threads.threadid()
                _hgemv_recursive!(Cthreads[id], block, B, offset)
            end
        end
    end
    # reduce
    for Ct in Cthreads
        axpy!(1, Ct, C)
    end
    return C
end

function _hgemv_static_partition!(C::AbstractVector, B::AbstractVector, partition, offset)
    # create a lock for the reduction step
    T = eltype(C)
    mutex = ReentrantLock()
    np = length(partition)
    nt = Threads.nthreads()
    Cthreads = [zero(C) for _ in 1:nt]
    @sync for n in 1:np
        Threads.@spawn begin
            id = Threads.threadid()
            leaves = partition[n]
            Cloc = Cthreads[id]
            for leaf in leaves
                irange = rowrange(leaf) .- offset[1]
                jrange = colrange(leaf) .- offset[2]
                data = leaf.data
                if T <: SVector
                    mul!(view(Cloc, irange, 1:1), data, view(B, jrange, 1:1), 1, 1)
                else
                    mul!(view(Cloc, irange), data, view(B, jrange), 1, 1)
                end
            end
            # reduction
            lock(mutex) do
                return axpy!(1, Cloc, C)
            end
        end
    end
    return C
end

function rmul!(R::RkMatrix, b::Number)
    m, n = size(R)
    if m > n
        rmul!(R.B, conj(b))
    else
        rmul!(R.A, b)
    end
    return R
end

function rmul!(H::HMatrix, b::Number)
    b == true && (return H) # short circuit. If inlined, rmul!(H,true) --> no-op
    if hasdata(H)
        rmul!(data(H), b)
    end
    for child in children(H)
        rmul!(child, b)
    end
    return H
end

"""
    _cost_gemv(A::Union{Matrix,SubArray,Adjoint})

A proxy for the computational cost of a matrix/vector product.
"""
function _cost_gemv(R::RkMatrix)
    return rank(R) * sum(size(R))
end
function _cost_gemv(M::Matrix)
    return length(M)
end
function _cost_gemv(H::HMatrix)
    acc = 0.0
    if isleaf(H)
        acc += _cost_gemv(data(H))
    else
        for c in children(H)
            acc += cost_gemv(c)
        end
    end
    return acc
end
