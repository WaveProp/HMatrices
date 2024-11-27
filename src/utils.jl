"""
    PermutedMatrix{K,T} <: AbstractMatrix{T}

Structured used to reprensent the permutation of a matrix-like object. The
original matrix is stored in the `data::K` field, and the permutations are
stored in `rowperm` and `colperm`.
"""
struct PermutedMatrix{K,T} <: AbstractMatrix{T}
    data::K # original matrix
    rowperm::Vector{Int}
    colperm::Vector{Int}
    function PermutedMatrix(orig, rowperm, colperm)
        K = typeof(orig)
        T = eltype(orig)
        return new{K,T}(orig, rowperm, colperm)
    end
end
Base.size(M::PermutedMatrix) = size(M.data)

function Base.getindex(M::PermutedMatrix, i, j)
    ip = M.rowperm[i]
    jp = M.colperm[j]
    return M.data[ip, jp]
end

"""
    filter_tree(f,tree,isterminal=true)

Return a vector containing all the nodes of `tree` such that
`f(node)==true`.  The argument `isterminal` can be used to control whether
to continue the search on `children` of nodes for which `f(node)==true`.
"""
function filter_tree(f, tree, isterminal = true)
    T = eltype(children(tree))
    nodes = Vector{T}()
    return filter_tree!(f, nodes, tree, isterminal)
end

"""
    filter_tree!(filter,nodes,tree,[isterminal=true])

Like [`filter_tree`](@ref), but appends results to `nodes`.
"""
function filter_tree!(f, nodes, tree, isterminal = true)
    if f(tree)
        push!(nodes, tree)
        # terminate the search along this path if terminal=true
        isterminal || map(x -> filter_tree!(f, nodes, x, isterminal), children(tree))
    else
        # continue on on children
        map(x -> filter_tree!(f, nodes, x, isterminal), children(tree))
    end
    return nodes
end

"""
    leaves(tree)

Return a vector containing all the leaf nodes of `tree`.
"""
function leaves(tree)
    isterminal = true
    return filter_tree(isleaf, tree, isterminal)
end

"""
    nodes(tree)

Return a vector containing all the nodes of `tree`.
"""
function nodes(tree)
    isterminal = false
    return filter_tree(x -> true, tree, isterminal)
end

"""
    depth(tree,acc=0)

Recursive function to compute the depth of `node` in a a tree-like structure.

Overload this function if your structure has a more efficient way to compute
`depth` (e.g. if it stores it in a field).
"""
function depth(tree, acc = 0)
    if isroot(tree)
        return acc
    else
        depth(parent(tree), acc + 1)
    end
end

"""
    struct VectorOfVectors{T}

A simple structure which behaves as a `Vector{Vector{T}}` but stores the entries
in a contiguous `data::Vector{T}` field. All vectors in the `VectorOfVectors`
are assumed to be of size `m`, and there are `k` of them, meaning this structure
can be used to represent a `m × k` matrix.

Similar to a vector-of-vectors, calling `A[i]` returns a view to the `i`-th
column.

See also: [`newcol!`](@ref)
"""
mutable struct VectorOfVectors{T}
    const data::Vector{T}
    m::Int
    k::Int
end

VectorOfVectors(T, m = 0, k = 0) = VectorOfVectors{T}(Vector{T}(undef, m * k), m, k)

"""
    newcol!(A::VectorOfVectors)

Append a new (unitialized) column to `A`, and return a view of it.
"""
function newcol!(A::VectorOfVectors)
    m, k = A.m, A.k
    is = m * k + 1
    ie = m * (k + 1)
    if ie > length(A.data)
        resize!(A.data, ie)
    end
    A.k += 1
    return view(A.data, is:ie)
end

"""
    reset!(A::VectorOfVectors)

Set the number of columns of `A` to zero, and the number of rows to zero, but
does not `resize!` the underlying data vector.
"""
reset!(A::VectorOfVectors) = (A.m = 0; A.k = 0)

function Base.getindex(A::VectorOfVectors, i)
    i <= A.k || throw(BoundsError(A, i))
    return view(A.data, (i-1)*A.m+1:i*A.m)
end

function Base.Matrix(A::VectorOfVectors{T}) where {T}
    out = Matrix{T}(undef, A.m, A.k)
    return copyto!(out, 1, A.data, 1, length(out))
end

Base.length(A::VectorOfVectors) = A.k

"""
    getcol!(col, M, j)

Return the `j`-th column of `M` in `col`.
"""
function getcol!(col, M, j)
    @assert length(col) == size(M, 1)
    return copyto!(col, view(M, :, j))
end

"""
    getcol(M, j)

Return the `j`-th column of `M`.
"""
function getcol(M, j)
    n = size(M, 1)
    col = Vector{eltype(M)}(undef, n)
    return getcol!(col, M, j)
end
