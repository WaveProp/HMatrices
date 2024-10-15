"""
    mutable struct RkMatrix{T}

Representation of a rank `r` matrix `M` in outer product format `M =
A*adjoint(B)` where `A` has size `m × r` and `B` has size `n × r`.

The internal representation stores `A` and `B`, but `R.Bt` or `R.At` can be used
to get the respective adjoints.
"""
mutable struct RkMatrix{T} <: AbstractMatrix{T}
    A::Matrix{T}
    B::Matrix{T}    
    rel_er :: Vector{Float64}
    function RkMatrix(A::Matrix{T}, B::Matrix{T}, rlist::Vector{Float64}) where {T}
        @assert size(A, 2) == size(B, 2) "second dimension of `A` and `B` must match"
        m, r = size(A)
        n = size(B, 1)
        if r * (m + n) >= m * n
            @debug "Inefficient RkMatrix:" size(A) size(B)
        end
        return new{T}(A, B, rlist)
    end
end

function RkMatrix(A::Matrix{T}, B::Matrix{T}) where {T}
     rk_matrix = RkMatrix(A,B,Vector{Float64}(undef,0))

     return rk_matrix
end

function Base.setproperty!(R::RkMatrix, s::Symbol, mat::Base.Matrix)
    setfield!(R, s, mat)
    return R
end

function Base.show(io::IO, ::MIME"text/plain", R::RkMatrix)
    return print(
        io,
        size(R, 1),
        "×",
        size(R, 2),
        " RkMatrix{",
        eltype(R),
        "}",
        " of rank ",
        rank(R),
    )
end

function Base.getindex(::RkMatrix, args...)
    msg = """method `getindex(::RkMatrix,args...)` has been disabled to avoid
    performance pitfalls. Unless you made an explicit call to `getindex`, this
    error usually means that a linear algebra routine involving an
    `RkMatrix` has fallen back to a generic implementation."""
    return error(msg)
end

"""
    getcol!(col,M::AbstractMatrix,j)

Fill the entries of `col` with column `j` of `M`.
"""
function getcol!(col, R::RkMatrix, j::Int, ::Val{T} = Val(false)) where {T}
    # NOTE: using a `Val` argument to dispatch on the type of `T` is important
    # for performance
    return if T
        mul!(col, R.A, view(adjoint(R.B), :, j), true, true)
    else
        mul!(col, R.A, view(adjoint(R.B), :, j))
    end
end

const AdjRk = Adjoint{<:Any,<:RkMatrix}
const AdjRkOrRk = Union{AdjRk,RkMatrix}

function getcol!(
    col,
    Ra::Adjoint{<:Any,<:RkMatrix},
    j::Int,
    ::Val{T} = Val(false),
) where {T}
    # NOTE: using a `Val` argument to dispatch on the type of `T` is important
    # for performance
    R = parent(Ra)
    return if T
        mul!(col, R.B, view(adjoint(R.A), :, j), true, true)
    else
        mul!(col, R.B, view(adjoint(R.A), :, j))
    end
end

Base.copy(R::RkMatrix) = RkMatrix(copy(R.A), copy(R.B))

Base.eltype(::RkMatrix{T}) where {T} = T
Base.size(rmat::RkMatrix) = (size(rmat.A, 1), size(rmat.B, 1))
Base.length(rmat::RkMatrix) = prod(size(rmat))
function Base.isapprox(rmat::RkMatrix, B::AbstractArray, args...; kwargs...)
    return isapprox(Matrix(rmat), B, args...; kwargs...)
end

LinearAlgebra.rank(M::RkMatrix) = size(M.A, 2)

function Base.getproperty(R::RkMatrix, s::Symbol)
    if s == :Bt
        return adjoint(R.B)
    elseif s == :At
        return adjoint(R.A)
    else
        return getfield(R, s)
    end
end

function Base.Matrix(R::RkMatrix{<:Number})
    return R.A * R.Bt
end
function Base.Matrix(adjR::Adjoint{<:Any,<:RkMatrix})
    R = parent(adjR)
    return R.B * R.At
end
function Base.Matrix(R::RkMatrix{<:SMatrix})
    # collect must be used when we have a matrix of `SMatrix` because of this issue:
    # https://github.com/JuliaArrays/StaticArrays.jl/issues/966#issuecomment-943679214
    return R.A * collect(R.Bt)
end

# FIXME: to support scalar and tensorial problems, we currently allow for T
# to be something other than a plain number. If that is the case, we
# implement a (slow) multiplication algorithm by hand to circumvent a
# problem in LinearAlgebra for the generic mulplication mul!(C,A,B,a,b) when
# C and B are a vectors of static matrices, and A is a matrix of static
# matrices. Should eventually be removed.
function LinearAlgebra.mul!(
    C::AbstractVector,
    Rk::RkMatrix{T},
    F::AbstractVector,
    a::Number,
    b::Number,
) where {T<:SMatrix}
    m, n = size(Rk)
    r = rank(Rk)
    tmp = Rk.Bt * F
    rmul!(C, b)
    for k in 1:r
        tmp[k] *= a
        for i in 1:m
            C[i] += Rk.A[i, k] * tmp[k]
        end
    end
    return C
end

# scalar multiplication
Base.:*(a::Number, R::RkMatrix) = (A = a * R.A; B = copy(R.B); RkMatrix(A, B))
