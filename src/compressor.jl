"""
    abstract type AbstractCompressor

Types used to compress matrices.
"""
abstract type AbstractCompressor end

"""
    struct PartialACA

Adaptive cross approximation algorithm with partial pivoting. This structure can be
used to generate an [`RkMatrix`](@ref) from a matrix-like object `M` as follows:

```jldoctest
using LinearAlgebra
rtol = 1e-6
comp = PartialACA(;rtol)
A = rand(10,2)
B = rand(10,2)
M = A*adjoint(B) # a low-rank matrix
R = comp(M, axes(M)...) # compress the entire matrix `M`
norm(Matrix(R) - M) < rtol*norm(M) # true

# output

true

```

Because it uses partial pivoting, the linear operator does not have to be
evaluated at every `i,j`. This is usually much faster than [`TSVD`](@ref), but
due to the pivoting strategy the algorithm may fail in special cases, even when
the underlying linear operator is of low rank.
"""
Base.@kwdef struct PartialACA
    atol::Float64 = 0
    rank::Int = typemax(Int)
    rtol::Float64 = atol > 0 || rank < typemax(Int) ? 0 : sqrt(eps(Float64))
end

function (paca::PartialACA)(K, rowtree::ClusterTree, coltree::ClusterTree, bufs = nothing)
    # find initial column pivot for partial ACA
    istart = _aca_partial_initial_pivot(rowtree)
    irange = index_range(rowtree)
    jrange = index_range(coltree)
    return _aca_partial(
        K,
        irange,
        jrange,
        paca.atol,
        paca.rank,
        paca.rtol,
        istart - irange.start + 1,
        bufs,
    )
end

function (paca::PartialACA)(K, irange::AbstractRange, jrange::AbstractRange, bufs = nothing)
    return _aca_partial(K, irange, jrange, paca.atol, paca.rank, paca.rtol, 1, bufs)
end

(paca::PartialACA)(K::AbstractMatrix) = paca(K, axes(K, 1), axes(K, 2))

struct ACABuffer{T}
    A::VectorOfVectors{T}
    B::VectorOfVectors{T}
    I::BitVector
    J::BitVector
end

function ACABuffer(T, m = 0, n = 0)
    return ACABuffer{T}(
        VectorOfVectors(T, m),
        VectorOfVectors(T, n),
        BitVector(undef, m),
        BitVector(undef, n),
    )
end

"""
    _aca_partial(K,irange,jrange,atol,rmax,rtol,istart=1)

Internal function implementing the adaptive cross-approximation algorithm with
partial pivoting. The returned `R::RkMatrix` provides an approximation to
`K[irange,jrange]` which has either rank `is expected to satisfy `|M - R| <
max(atol,rtol*|M|)`, but this inequality may fail to hold due to the various
errors involved in estimating the error and |M|.
"""
function _aca_partial(K, irange, jrange, atol, rmax, rtol, istart, buffer_ = nothing)
    Kadj = adjoint(K)
    T = eltype(K)
    # if irange and jrange are Colon, extract the size from `K` directly. This
    # allows for some code reuse with specializations on getindex(i,::Colon) and
    # getindex(::Colon,j) for when `K` is a `RkMatrix`
    if irange isa Colon && jrange isa Colon
        m, n = size(K)
        ishift, jshift = 0, 0
    else
        m, n = length(irange), length(jrange)
        ishift, jshift = first(irange) - 1, first(jrange) - 1
        # maps global indices to local indices
    end
    buffer = isnothing(buffer_) ? ACABuffer(T, m, n) : buffer_
    A, B, I, J = buffer.A, buffer.B, buffer.I, buffer.J # extract the individual buffers for convenience
    @assert A.k == 0 && B.k == 0 "buffers must be empty"
    A.m, B.m = m, n
    rmax = min(m, n, rmax)
    resize!(I, m)
    resize!(J, n)
    fill!(I, true)
    fill!(J, true)
    i = istart # initial pivot
    er = Inf
    est_norm = 0 # approximate norm of K[irange,jrange]
    r = 0 # current rank
    while er > max(atol, rtol * est_norm) && r < rmax
        # remove index i from allowed row
        if i < 1
            all(j -> iszero(K[first(irange), j]), jrange) &&
                all(i -> iszero(K[i, first(jrange)]), irange) ||
                @warn "aca possibly failed on $irange × $jrange"
            break
        end
        I[i] = false
        # pre-allocate row and column
        a = newcol!(A)
        A.k -= 1
        b = newcol!(B)
        B.k -= 1
        # compute next row by row <-- K[i+ishift,jrange] - R[i,:]
        getblock!(b, Kadj, jrange, i + ishift)
        for k in 1:r
            axpy!(-adjoint(A[k][i]), B[k], b)
            # for j in eachindex(row)
            #     row[j] = row[j] - B[k][j]*adjoint(A[k][i])
            # end
        end
        j = _aca_partial_pivot(b, J)
        δ = b[j]
        if svdvals(δ)[end] == 0
            @debug "zero pivot found during partial aca"
            i = findfirst(x -> x == true, I)
            if isnothing(i)
                # ran out of candidate rows. Good case: the matrix is zero. Bad
                # case: aca failed
                all(j -> iszero(K[first(irange), j]), jrange) &&
                    all(i -> iszero(K[i, first(jrange)]), irange) ||
                    @warn "aca possibly failed on $irange × $jrange"
                break
            end
        else # δ != 0
            iδ = inv(δ)
            # rdiv!(b,δ) # b <-- b/δ
            for k in eachindex(b)
                b[k] = b[k] * iδ
            end
            J[j] = false
            # compute next col by col <-- K[irange,j+jshift] - R[:,j]
            getblock!(a, K, irange, j + jshift)
            for k in 1:r
                axpy!(-adjoint(B[k][j]), A[k], a)
                # for i in eachindex(col)
                #     col[i] = col[i] - A[k][i]*adjoint(B[k][j])
                # end
            end
            # increase rank and push cross
            r += 1
            A.k += 1
            B.k += 1
            # estimate the error by || R_{k} - R_{k-1} || = ||a|| ||b||
            er = norm(a) * norm(b)
            # estimate the norm by || K || ≈ || R_k ||
            est_norm = _update_frob_norm(est_norm, A, B)
            i = _aca_partial_pivot(a, I)
            @debug r, er, m, n
        end
    end
    # copy from buffer to matrix and create Rk matrix
    R_ = RkMatrix(Matrix(A), Matrix(B))
    # # indicate that `A` and `B` can be reused
    reset!(A)
    reset!(B)
    return R_
end

"""
    _update_frob_norm(acc,A,B)

Given the Frobenius norm of `Rₖ = A[1:end-1]*adjoint(B[1:end-1])` in `acc`,
compute the Frobenius norm of `Rₖ₊₁ = A*adjoint(B)` efficiently.
"""
@inline function _update_frob_norm(cur, A, B)
    k = length(A)
    a = A[k]
    b = B[k]
    out = norm(a)^2 * norm(b)^2
    for l in 1:(k-1)
        out += 2 * real(dot(A[l], a) * (dot(b, B[l])))
    end
    return sqrt(cur^2 + out)
end

"""
    _aca_partial_pivot(v,I)

Find in the valid set `I` the index of the element `x ∈ v` maximizing its
smallest singular value. This is equivalent to minimizing the spectral norm of
the inverse of `x`.

When `x` is a scalar, this is simply the element with largest absolute value.

This general implementation should work for both scalar as well as tensor-valued
kernels; see
(https://www.sciencedirect.com/science/article/pii/S0021999117306721)[https://www.sciencedirect.com/science/article/pii/S0021999117306721]
for more details.
"""
function _aca_partial_pivot(v, J)
    idx = -1
    val = -Inf
    for n in 1:length(J)
        x = v[n]
        σ = svdvals(x)[end]
        if σ > val && J[n]
            idx = n
            val = σ
        end
    end
    return idx
end

function _aca_partial_initial_pivot(rowtree)
    # the method below is suggested in Bebendorf, but it does not seem to
    # improve the  error. The idea is that the initial pivot is the closesest
    # point to the center of the cluster
    xc = center(container(rowtree))
    d = Inf
    els = root_elements(rowtree)
    loc_idxs = index_range(rowtree)
    istart = first(loc_idxs)
    for i in loc_idxs
        x = els[i]
        if norm(x - xc) < d
            d = norm(x - xc)
            istart = i
        end
    end
    return istart
end

"""
    struct TSVD

Compression algorithm based on *a posteriori* truncation of an `SVD`. This is
the optimal approximation in Frobenius norm; however, it also tends to be very
expensive and thus should be used mostly for "small" matrices.
"""
Base.@kwdef struct TSVD
    atol::Float64 = 0
    rank::Int = typemax(Int)
    rtol::Float64 = atol > 0 || rank < typemax(Int) ? 0 : sqrt(eps(Float64))
end

function (tsvd::TSVD)(K, rowtree::ClusterTree, coltree::ClusterTree, bufs = nothing)
    irange = index_range(rowtree)
    jrange = index_range(coltree)
    return tsvd(K, irange, jrange)
end

function (tsvd::TSVD)(K, irange::UnitRange, jrange::UnitRange)
    M = K[irange, jrange]
    return compress!(M, tsvd)
end

(comp::TSVD)(K::Base.Matrix) = comp(K, 1:size(K, 1), 1:size(K, 2))

"""
    compress!(M::RkMatrix,tsvd::TSVD)

Recompress the matrix `R` using a truncated svd of `R`. The implementation uses
the `qr-svd` strategy to efficiently compute `svd(R)` when `rank(R) ≪
min(size(R))`.
"""
function compress!(R::RkMatrix, tsvd::TSVD)
    m, n = size(R)
    QA, RA = qr!(R.A)
    QB, RB = qr!(R.B)
    F = svd!(RA * adjoint(RB)) # svd of an r×r problem
    U = QA * F.U
    Vt = F.Vt * adjoint(QB)
    V = adjoint(Vt)
    sp_norm = F.S[1] # spectral norm
    r = findlast(x -> x > max(tsvd.atol, tsvd.rtol * sp_norm), F.S)
    isnothing(r) && (r = min(rank(R), m, n))
    r = min(r, tsvd.rank)
    if m < n
        A = @views U[:, 1:r] * Diagonal(F.S[1:r])
        B = V[:, 1:r]
    else
        A = U[:, 1:r]
        B = @views (V[:, 1:r]) * Diagonal(F.S[1:r])
    end
    R.A, R.B = A, B
    return R
end

"""
    compress!(M::Matrix,tsvd::TSVD)

Recompress the matrix `M` using a truncated svd and output an `RkMatrix`. The
data in `M` is invalidated in the process.
"""
function compress!(M::Base.Matrix, tsvd::TSVD)
    m, n = size(M)
    F = svd!(M)
    sp_norm = F.S[1] # spectral norm
    r = findlast(x -> x > max(tsvd.atol, tsvd.rtol * sp_norm), F.S)
    isnothing(r) && (r = min(m, n))
    r = min(r, tsvd.rank)
    if m < n
        A = @views F.U[:, 1:r] * Diagonal(F.S[1:r])
        B = F.V[:, 1:r]
    else
        A = F.U[:, 1:r]
        B = @views (F.V[:, 1:r]) * Diagonal(F.S[1:r])
    end
    return RkMatrix(A, B)
end
