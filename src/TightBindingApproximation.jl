module TightBindingApproximation

using Printf: @sprintf
using TimerOutputs: @timeit
using LinearAlgebra: inv, dot, Hermitian, Diagonal, eigvals, cholesky
using QuantumLattices: getcontent, expand, id, pidtype, iidtype, rcoord, idtype, plain, creation, annihilation, atol, rtol
using QuantumLattices: AbstractPID, FID, NID, Index, Internal, Fock, Phonon, AbstractLattice, Bonds, Hilbert, OIDToTuple, Table, Term, Boundary
using QuantumLattices: Hopping, Onsite, Pairing, PhononKinetic, PhononPotential, DMPhonon
using QuantumLattices: Engine, Parameters, AbstractGenerator, Generator, Action, Assignment, Algorithm

import LinearAlgebra: eigen, ishermitian
import QuantumLattices: contentnames, statistics, dimension, kind, matrix!, update!, prepare!, run!

export TBAKind, AbstractTBA, TBAMatrix, commutator, TBA, TBAEB

"""
    TBAKind{K}

The kind of a free quantum lattice system using the tight-binding approximation.
"""
struct TBAKind{K}
    TBAKind(k::Symbol) = new{k}()
end
@inline TBAKind{K}() where K = TBAKind(K)
@inline Base.promote_rule(::Type{TBAKind{K}}, ::Type{TBAKind{K}}) where K = TBAKind{K}
@inline Base.promote_rule(::Type{TBAKind{:TBA}}, ::Type{TBAKind{:BdG}}) = TBAKind{:BdG}
@inline Base.promote_rule(::Type{TBAKind{:BdG}}, ::Type{TBAKind{:TBA}}) = TBAKind{:BdG}

"""
    TBAKind(T::Type{<:Term})

Depending on the kind of a term type, get the corresponding TBA kind.
"""
@inline TBAKind(::Type{T}) where {T<:Term} = error("kind error: not defined for $(kind(T)).")
@inline TBAKind(::Type{T}) where {T<:Union{Hopping, Onsite}} = TBAKind(:TBA)
@inline TBAKind(::Type{T}) where {T<:Union{Pairing, PhononKinetic, PhononPotential, DMPhonon}} = TBAKind(:BdG)
@inline @generated function TBAKind(::Type{TS}) where {TS<:Tuple{Vararg{Term}}}
    exprs = []
    for i = 1:fieldcount(TS)
        push!(exprs, :(typeof(TBAKind(fieldtype(TS, $i)))))
    end
    return Expr(:call, Expr(:call, :reduce, :promote_type, Expr(:tuple, exprs...)))
end

"""
    OIDToTuple(::TBAKind, ::Type{I}) where {I<:Index{<:AbstractPID, <:FID}} -> OIDToTuple
    OIDToTuple(::TBAKind, ::Type{I}) where {I<:Index{<:AbstractPID, <:NID}} -> OIDToTuple

Get the oid-to-tuple metric for a free fermionic/bosonic system or a free phononic system.
"""
@inline @generated function OIDToTuple(::TBAKind, ::Type{I}) where {I<:Index{<:AbstractPID, <:FID}}
    return OIDToTuple(:nambu, fieldnames(pidtype(I))..., :orbital, :spin)
end
@inline @generated function OIDToTuple(::TBAKind{:TBA}, ::Type{I}) where {I<:Index{<:AbstractPID, <:FID}}
    return OIDToTuple(fieldnames(pidtype(I))..., :orbital, :spin)
end
@inline @generated function OIDToTuple(::TBAKind, ::Type{I}) where {I<:Index{<:AbstractPID, <:NID}}
    return OIDToTuple(:tag, fieldnames(pidtype(I))..., :dir)
end

"""
    commutator(::TBAKind, ::Type{<:Internal}, n::Integer) -> Union{AbstractMatrix, Nothing}

Get the commutation relation of the single-particle operators of a free quantum lattice system using the tight-binding approximation.
"""
@inline commutator(::TBAKind, ::Type{<:Internal}, ::Integer) = nothing
@inline commutator(::TBAKind{:BdG}, ::Type{<:Fock{:f}}, n::Integer) = nothing
@inline commutator(::TBAKind{:BdG}, ::Type{<:Fock{:b}}, n::Integer) = Diagonal(kron([1, -1], ones(Int64, n÷2)))
@inline commutator(::TBAKind{:BdG}, ::Type{<:Phonon}, n::Integer) = Hermitian(kron([0 -1im; 1im 0], Diagonal(ones(Int, n÷2))))

"""
    AbstractTBA{K, H<:AbstractGenerator, G<:Union{Nothing, AbstractMatrix}} <: Engine

Abstract type for free quantum lattice systems using the tight-binding approximation.
"""
abstract type AbstractTBA{K, H<:AbstractGenerator, G<:Union{Nothing, AbstractMatrix}} <: Engine end
@inline contentnames(::Type{<:AbstractTBA}) = (:H, :commutator)
@inline kind(tba::AbstractTBA) = kind(typeof(tba))
@inline kind(::Type{<:AbstractTBA{K}}) where K = K
@inline Base.eltype(tba::AbstractTBA) = eltype(typeof(tba))
@inline Base.eltype(::Type{<:AbstractTBA{K, H} where K}) where {H<:AbstractGenerator} = eltype(H)
@inline Base.valtype(tba::AbstractTBA) = valtype(typeof(tba))
@inline Base.valtype(::Type{<:AbstractTBA{K, H} where K}) where {H<:AbstractGenerator} = valtype(valtype(eltype(H)))
@inline Base.valtype(tba::AbstractTBA, ::Nothing) = valtype(tba)
@inline Base.valtype(tba::AbstractTBA, k) = promote_type(valtype(tba), Complex{Int})
@inline statistics(tba::AbstractTBA) = statistics(typeof(tba))
@inline statistics(::Type{<:AbstractTBA{K, H} where K}) where {H<:AbstractGenerator} = statistics(eltype(idtype(eltype(H))))
@inline dimension(tba::AbstractTBA) = length(getcontent(getcontent(tba, :H), :table))
@inline update!(tba::AbstractTBA; kwargs...) = (update!(getcontent(tba, :H); kwargs...); tba)
@inline Parameters(tba::AbstractTBA) = Parameters(getcontent(tba, :H))

"""
    TBAMatrix{T, H<:AbstractMatrix{T}, G<:Union{AbstractMatrix, Nothing}} <: AbstractMatrix{T}

Matrix representation of a free quantum lattice system using the tight-binding approximation.
"""
struct TBAMatrix{T, H<:AbstractMatrix{T}, G<:Union{AbstractMatrix, Nothing}} <: AbstractMatrix{T}
    H::H
    commutator::G
    function TBAMatrix(H::AbstractMatrix, commutator::Union{AbstractMatrix, Nothing})
        new{eltype(H), typeof(H), typeof(commutator)}(H, commutator)
    end
end
@inline Base.size(m::TBAMatrix) = size(m.H)
@inline Base.getindex(m::TBAMatrix, i::Integer, j::Integer) = m.H[i, j]
@inline ishermitian(m::TBAMatrix) = ishermitian(typeof(m))
@inline ishermitian(::Type{<:TBAMatrix}) = true

"""
    matrix!(tba::AbstractTBA; k=nothing, kwargs...) -> TBAMatrix

Get the matrix representation of a free quantum lattice system.
"""
function matrix!(tba::AbstractTBA{TBAKind(:TBA)}; k=nothing, kwargs...)
    length(kwargs)>0 && update!(tba; kwargs...)
    H = getcontent(tba, :H)
    table = getcontent(H, :table)
    result = zeros(valtype(tba, k), dimension(tba), dimension(tba))
    for op in values(expand(H))
        seq₁, seq₂ = table[id(op)[1].index'], table[id(op)[2].index]
        phase = isnothing(k) ? one(valtype(tba, k)) : convert(valtype(tba, k), exp(-1im*dot(k, rcoord(op))))
        result[seq₁, seq₂] += op.value*phase
    end
    return TBAMatrix(Hermitian(result), getcontent(tba, :commutator))
end
function matrix!(tba::AbstractTBA{TBAKind(:BdG)}; k=nothing, kwargs...)
    length(kwargs)>0 && update!(tba; kwargs...)
    H = getcontent(tba, :H)
    table = getcontent(H, :table)
    result = zeros(valtype(tba, k), dimension(tba), dimension(tba))
    for op in values(expand(H))
        seq₁, seq₂ = table[id(op)[1].index'], table[id(op)[2].index]
        phase = isnothing(k) ? one(valtype(tba, k)) : convert(valtype(tba, k), exp(-1im*dot(k, rcoord(op))))
        result[seq₁, seq₂] += op.value*phase
        if id(op)[1].index.iid.nambu==creation && id(op)[2].index.iid.nambu==annihilation
            seq₁, seq₂ = table[id(op)[1].index], table[id(op)[2].index']
            sign = statistics(tba)==:f ? -1 : +1
            result[seq₁, seq₂] += sign*op.value*phase'
        end
    end
    return TBAMatrix(Hermitian(result), getcontent(tba, :commutator))
end

"""
    eigen(m::TBAMatrix) -> Eigen

Solve the eigen problem of a free quantum lattice system.
"""
@inline eigen(m::TBAMatrix{T, H, Nothing}) where {T, H<:AbstractMatrix{T}} = eigen(m.H)
function eigen(m::TBAMatrix{T, H, G}) where {T, H<:AbstractMatrix{T}, G<:AbstractMatrix}
    W = cholesky(m.H)
    K = eigen(Hermitian(W.U*m.commutator*W.L))
    @assert length(K.values)%2==0 "eigen error: wrong dimension of matrix."
    for i = 1:(length(K.values)÷2)
        K.values[i] = -K.values[i]
    end
    V = inv(W.U)*K.vectors*sqrt(Diagonal(K.values))
    return Eigen(K.values, V)
end

"""
    TBA{K, L<:AbstractLattice, H<:AbstractGenerator, G<:Union{AbstractMatrix, Nothing}} <: AbstractTBA{K, H, G}

The usual tight binding approximation for quantum lattice systems.
"""
struct TBA{K, L<:AbstractLattice, H<:AbstractGenerator, G<:Union{AbstractMatrix, Nothing}} <: AbstractTBA{K, H, G}
    lattice::L
    H::H
    commutator::G
    function TBA{K}(lattice::AbstractLattice, H::AbstractGenerator, commutator::Union{AbstractMatrix, Nothing}) where K
        @assert isa(K, TBAKind) "TBA error: wrong kind."
        if !isnothing(commutator)
            values = eigvals(commutator)
            num₁ = count(isapprox(+1, atol=atol, rtol=rtol), values)
            num₂ = count(isapprox(-1, atol=atol, rtol=rtol), values)
            @assert num₁==num₂==length(values)//2 "TBA error: unsupported input commutator."
        end
        new{K, typeof(lattice), typeof(H), typeof(commutator)}(lattice, H, commutator)
    end
end
@inline contentnames(::Type{<:TBA}) = (:lattice, :H, :commutator)

"""
    TBA(lattice::AbstractLattice, hilbert::Hilbert, terms::Tuple{Vararg{Term}}; boundary::Boundary=plain)

Construct a tight-binding quantum lattice system.
"""
@inline function TBA(lattice::AbstractLattice, hilbert::Hilbert, terms::Tuple{Vararg{Term}}; boundary::Boundary=plain)
    tbakind = TBAKind(typeof(terms))
    table = Table(hilbert, OIDToTuple(tbakind, Index{hilbert|>keytype, hilbert|>valtype|>eltype}))
    commt = commutator(tbakind, hilbert|>valtype, length(table))
    return TBA{tbakind}(lattice, Generator(terms, Bonds(lattice), hilbert; half=false, table=table, boundary=boundary), commt)
end

"""
    TBAEB{P} <: Action

Energy bands by tight-binding-approximation for quantum lattice systems.
"""
struct TBAEB{P} <: Action
    path::P
end
@inline prepare!(eb::TBAEB, tba::AbstractTBA) = (zeros(Float64, length(eb.path)), zeros(Float64, length(eb.path), dimension(tba)))
@inline Base.nameof(tba::Algorithm{<:AbstractTBA}, eb::Assignment{<:TBAEB}) = @sprintf "%s_%s" repr(tba, ∉(keys(eb.action.path))) eb.id
function run!(tba::Algorithm{<:AbstractTBA}, eb::Assignment{<:TBAEB})
    for (i, params) in enumerate(eb.action.path)
        eb.data[1][i] = length(params)==1 && isa(first(params), Number) ? first(params) : i
        @timeit tba.timer "matrix" (m = matrix!(tba.engine; params...))
        @timeit tba.timer "eigen" (eb.data[2][i, :] = eigen(m).values)
    end
end

end # module