module Wannier90

using DelimitedFiles: readdlm
using LinearAlgebra: Hermitian, dot, norm
using QuantumLattices: CoordinatedIndex, FockIndex, Hilbert, Index, Lattice, LinearTransformation, Matrixization, Operator, OperatorIndexToTuple, OperatorPack, Operators, OperatorSet, OperatorSum, ReciprocalPath, StaticGenerator, Table
using QuantumLattices: 𝕔, 𝕔⁺
using StaticArrays: SVector
using ..TightBindingApproximation: Fermionic, Quadraticization, TBAMatrix

import QuantumLattices: Algorithm, add!, dimension, matrix
import ..TightBindingApproximation: TBA

export Operatorization, W90, W90Hoppings, W90Matrixization, choosehamiltonian, distancehamiltonian, showW90, showhamiltonian, findblock, readbands, readcenters, readhamiltonian, readlattice, readpath

"""
    findblock(name::String, content::String) -> Union{Nothing, Vector{SubString{String}}}

Find a named block in the content of the Wannier90 ".win" input file.
"""
function findblock(name::String, content::String)
    r = match(Regex("(?i)begin\\s*"*name*"\\s*\\n([\\s\\S]*)\\s*\\nend\\s*"*name), content)
    isnothing(r) && return
    return strip.(split(r[1], "\n"))
end

"""
    readlattice(path::AbstractString, seedname::AbstractString; name::Symbol=Symbol(seedname), projection::Bool=true) -> Lattice{3, Float64, 3}

Read the lattice from the Wannier90 ".win" input file with a given `name`.

Besides, `projection` specifies whether only the sites with initial projections in the Wannier90 ".win" input file are included in the constructed lattice.
"""
function readlattice(path::AbstractString, seedname::AbstractString; name::Symbol=Symbol(seedname), projection::Bool=true)
    content = open(io->read(io, String), joinpath(path, seedname*".win"), "r")
    # get translation vectors
    lines = findblock("unit_cell_cart", content)
    isnothing(lines) && error("readlattice error: unable to find \"unit_cell_cart\" block in the Wannier90 \".win\" input file.")
    @assert length(lines)∈(3, 4) "readlattice error: incorrect \"unit_cell_cart\" block."
    pref = coeff(lines)
    lines = map(i->split(lines[i]), start(lines):length(lines))
    vectors = SVector(
        SVector(parse(Float64, lines[1][1])*pref, parse(Float64, lines[1][2])*pref, parse(Float64, lines[1][3])*pref),
        SVector(parse(Float64, lines[2][1])*pref, parse(Float64, lines[2][2])*pref, parse(Float64, lines[2][3])*pref),
        SVector(parse(Float64, lines[3][1])*pref, parse(Float64, lines[3][2])*pref, parse(Float64, lines[3][3])*pref)
    )
    # get atom positions
    lines = findblock("atoms_frac", content)
    if !isnothing(lines)
        lines = map(split, lines)
        coordinates = hcat(vectors...) * [parse(Float64, lines[j][i]) for i=2:4, j=1:length(lines)]
    else
        lines = findblock("atoms_cart", content)
        isnothing(lines) && error("readlattice error: unable to find \"atoms_frac\" or \"atoms_cart\" block in the Wannier90 \".win\" input file.")
        pref = coeff(lines)
        lines = map(i->split(lines[i]), start(lines):length(lines))
        coordinates = [parse(Float64, lines[j][i])*pref for i=2:4, j=1:length(lines)]
    end
    # projection of atoms
    if projection
        atoms = [lowercase(line[1]) for line in lines]
        lines = findblock("projections", content)
        pref = coeff(lines)
        lines = map(line->split(line, ":")[1], lines)
        sites = Int[]
        for i = start(lines):length(lines)
            site = strip(split(lines[i], ":")[1])
            if occursin(r"(?i)[fc]=(.*)", site)
                # specify an atom by its position
                coordinate = map(str->parse(Float64, str), split(match(r"(?i)[fc]=(.*)", site)[1], ","))
                coordinate = occursin(r"(?i)f=", site) ? mapreduce(*, +, vectors, coordinate) : pref*coordinate
                result = findsite(coordinate, coordinates)
                @assert isa(result, Int) "readlattice error: cannot find site($site)."
                push!(sites, result)
            else
                # specify an atom by its symbol
                result = findall(==(lowercase(site)), atoms)
                @assert length(result)>0 "readlattice error: cannot find $site atoms."
                append!(sites, result)
            end
        end
        coordinates = coordinates[:, sites]
    end
    return Lattice(name, coordinates::Matrix{Float64}, vectors::SVector{3, SVector{3, Float64}})
end
@inline coeff(lines) = lowercase(lines[1])=="bohr" ? 0.5291772108 : 1.0
@inline start(lines) = lowercase(lines[1])∈("bohr", "ang", "angstrom") ? 2 : 1
function findsite(coordinate, coordinates; atol=10^-6, rtol=10^-6)
    for i = 1:size(coordinates)[2]
        isapprox(coordinate, coordinates[:, i]; atol=atol, rtol=rtol) && return i
    end
end

"""
    readlattice(path::AbstractString; name::Symbol=:lattice) -> Lattice{3, Float64, 3}

Read the lattice from the "POSCAR" file with a given `name`.
"""
function readlattice(path::AbstractString; name::Symbol=:lattice)
    lines = readlines(joinpath(path, "POSCAR"))
    scale = parse.(Float64, split(lines[2]))
    length(scale)==1 && (scale = fill(only(scale), 3))
    vector(line, scale) = SVector{3, Float64}(map(x->parse(Float64, x)*scale, split(line)))
    vectors = SVector(vector(lines[3], scale[1]), vector(lines[4], scale[2]), vector(lines[5], scale[3]))
    total = sum(parse.(Int, split(lines[7])))
    coordinates = zeros(Float64, 3, total)
    flag = lowercase(strip(lines[8]))=="cartesian"
    for i in 1:total
        tokens = split(lines[8+i])
        coord = parse.(Float64, tokens[1:3])
        coordinates[:, i] = flag ? map(*, coord, scale) : mapreduce(*, +, vectors, coord)
    end
    return Lattice(name, coordinates::Matrix{Float64}, vectors::SVector{3, SVector{3, Float64}})
end

"""
    readcenters(path::AbstractString, seedname::AbstractString) -> Matrix{Float64}

Read the centers of Wannier functions from the Wannier90 "_centres.xyz" output data file.
"""
function readcenters(path::AbstractString, seedname::AbstractString)
    content = open(io->read(io, String), joinpath(path, seedname*"_centres.xyz"), "r")
    lines = [content[index] for index in findall(r"X.*", content)]
    length(lines)==0 && error("readcenters error: Wannier centers not found.")
    result = zeros(Float64, 3, length(lines))
    for (i, line) in enumerate(lines)
        parts = split(line)
        result[1, i] = parse(Float64, parts[2])
        result[2, i] = parse(Float64, parts[3])
        result[3, i] = parse(Float64, parts[4])
    end
    return result
end

"""
    W90Hoppings <: OperatorPack{Matrix{ComplexF64}, NTuple{3, Int}}

Hopping amplitudes among the Wannier orbitals in two unitcells with a fixed relative displacement.

Here,
1) the `:value` attribute represents the hopping amplitude matrix of Wannier orbitals,
2) the `:id` attribute represents the relative displacement of the two unitcells in basis of the lattice translation vectors.
"""
struct W90Hoppings <: OperatorPack{Matrix{ComplexF64}, NTuple{3, Int}}
    value::Matrix{ComplexF64}
    id::NTuple{3, Int}
    function W90Hoppings(value::AbstractMatrix{<:Number}, id::NTuple{3, Integer})
        @assert size(value)[1]==size(value)[2] "W90Hoppings error: hopping matrix is not square."
        new(value, id)
    end
end

"""
    readhamiltonian(path::AbstractString, seedname::AbstractString) -> StaticGenerator{OperatorSum{W90Hoppings, NTuple{3, Int}}}

Read the hamiltonian from the Wannier90 "_hr.dat" output data file.
"""
function readhamiltonian(path::AbstractString, seedname::AbstractString)
    lines = open(readlines, joinpath(path, seedname*"_hr.dat"), "r")
    num_wan = parse(Int, lines[2])
    nrpts = parse(Int, lines[3])
    degeneracies = Int[]
    start = 0
    for i in 4:length(lines)
        append!(degeneracies, parse(Int, degeneracy) for degeneracy in split(lines[i]))
        if length(degeneracies) == nrpts
            start = i+1
            break
        end
        length(degeneracies)>nrpts && error("readhamiltonian error: too many degeneracies for the Wigner-Seitz points of the superlattice.")
    end
    index = 1
    ham = OperatorSum{W90Hoppings}()
    deg = Dict{NTuple{3, Int}, Int}()
    for k in start:length(lines)
        info = split(lines[k])
        point = (parse(Int, info[1]), parse(Int, info[2]), parse(Int, info[3]))
        i, j = parse(Int, info[4]), parse(Int, info[5])
        value = parse(Float64, info[6]) + 1im*parse(Float64, info[7])
        if !haskey(ham, point)
            ham.contents[point] = W90Hoppings(zeros(ComplexF64, num_wan, num_wan), point)
            deg[point] = degeneracies[index]
            index += 1
        end
        ham.contents[point].value[i, j] = value / deg[point]
    end
    for point in keys(ham.contents)
        @assert haskey(ham, map(-, point)) "readhamiltonian error: did not find the inverse of point $(point)."
    end
    return StaticGenerator(ham)
end

"""
    choosehamiltonian(
        ham::StaticGenerator{OperatorSum{W90Hoppings, NTuple{3, Int}}},
        lattice::Lattice,
        centers::Matrix{Float64};
        min_hopping_norm::Real = 0.0,
        max_hopping_norm::Union{Real, Nothing} = nothing,
        max_distance::Union{Real, Nothing} = nothing,
        min_distance::Union{Real, Nothing} = nothing
    ) -> StaticGenerator{OperatorSum{W90Hoppings, NTuple{3, Int}}}

Select Wannier90 hopping amplitudes that satisfy distance and/or hopping norm thresholds.

For each (R, i, j) triplet, the Cartesian distance is `norm(R_cart + centers[:, j] - centers[:, i])`
where `R_cart = Σ vectors × R`. Returns a new Hamiltonian containing only entries that pass all
active filters.

This is one of the diagnostic tools that can be used to help in determining
`min_hopping_norm` and `max_distance` for the tight-binding model.

# Keyword Arguments
- `min_hopping_norm::Real = 0.0`: minimum |value| to keep.
- `max_hopping_norm::Union{Real, Nothing} = nothing`: maximum |value| to keep.
- `min_distance::Union{Real, Nothing} = nothing`: minimum Cartesian distance to keep.
- `max_distance::Union{Real, Nothing} = nothing`: maximum Cartesian distance to keep.
"""
function choosehamiltonian(
    ham::StaticGenerator{OperatorSum{W90Hoppings, NTuple{3, Int}}},
    lattice::Lattice,
    centers::Matrix{Float64};
    min_hopping_norm::Real = 0.0,
    max_hopping_norm::Union{Real, Nothing} = nothing,
    max_distance::Union{Real, Nothing} = nothing,
    min_distance::Union{Real, Nothing} = nothing
)
    n_wan = size(centers, 2)
    @assert size(centers, 1) == 3 "choosehamiltonian error: centers must have 3 rows."
    @assert n_wan == size(first(ham).value, 1) "choosehamiltonian error: mismatched sizes of centers and hopping matrices."

    vectors = lattice.vectors
    filtered = OperatorSum{W90Hoppings}()

    for m in ham.operators
        R = m.id
        R_cart = mapreduce(*, +, vectors, R)

        new_value = zeros(ComplexF64, n_wan, n_wan)
        has_entry = false

        for j in 1:n_wan
            cⱼ = SVector(centers[1, j], centers[2, j], centers[3, j])
            for i in 1:n_wan
                val = m.value[i, j]
                abs_val = abs(val)

                # Hopping norm filters
                abs_val < min_hopping_norm && continue
                !isnothing(max_hopping_norm) && abs_val > max_hopping_norm && continue

                # Distance filters
                cᵢ = SVector(centers[1, i], centers[2, i], centers[3, i])
                dist = norm(R_cart + cⱼ - cᵢ)
                !isnothing(min_distance) && dist < min_distance && continue
                !isnothing(max_distance) && dist > max_distance && continue

                new_value[i, j] = val
                has_entry = true
            end
        end

        if has_entry
            filtered.contents[R] = W90Hoppings(new_value, R)
        end
    end

    return StaticGenerator(filtered)
end

"""
    distancehamiltonian(
        ham::StaticGenerator{OperatorSum{W90Hoppings, NTuple{3, Int}}},
        lattice::Lattice,
        centers::Matrix{Float64}
    ) -> (Vector{Float64}, Vector{Tuple{NTuple{3, Int}, Int, Int}}, Vector{Float64})

Return all hopping terms with their Cartesian distances and indices for diagnostic purposes.

For well localized Wannier functions, the hopping amplitude should decay exponentially with
distance. This function returns the data needed to check this behavior by plotting the
hopping amplitudes as a function of distance.

# Returns
- `distances::Vector{Float64}`: Cartesian distances between orbital `i` at origin cell and orbital `j` at cell `R`.
- `indices::Vector{Tuple{NTuple{3, Int}, Int, Int}}`: `(R, i, j)` triplets identifying each hopping.
- `values::Vector{Float64}`: `|hopping|` amplitudes (already degeneracy-divided).

# Example
```julia
using Plots
dists, idxs, vals = distancehamiltonian(ham, lattice, centers)
scatter(dists, log.(vals); xlabel="Distance (Å)", ylabel="log|hopping|", legend=false)
savefig("hopping_decay.pdf")
```
"""
function distancehamiltonian(ham::StaticGenerator{OperatorSum{W90Hoppings, NTuple{3, Int}}}, lattice::Lattice, centers::Matrix{Float64})
    n_wan = size(centers, 2)
    vectors = lattice.vectors

    total = n_wan * n_wan * length(ham.operators)
    distances = Vector{Float64}(undef, total)
    values = Vector{Float64}(undef, total)
    indices = Vector{Tuple{NTuple{3, Int}, Int, Int}}(undef, total)

    k = 1
    for m in ham.operators
        R = m.id
        R_cart = mapreduce(*, +, vectors, R)

        for j in 1:n_wan
            cⱼ = SVector(centers[1, j], centers[2, j], centers[3, j])
            for i in 1:n_wan
                cᵢ = SVector(centers[1, i], centers[2, i], centers[3, i])
                dist = norm(R_cart + cⱼ - cᵢ)

                distances[k] = dist
                values[k] = abs(m.value[i, j])
                indices[k] = (R, i, j)
                k += 1
            end
        end
    end
    return distances, indices, values
end

"""
    choosehamiltonian(
        path::AbstractString,
        seedname::AbstractString;
        name::Symbol=Symbol(seedname),
        projection::Bool=true,
        min_hopping_norm::Real=0.0,
        max_hopping_norm::Union{Real, Nothing}=nothing,
        max_distance::Union{Real, Nothing}=nothing,
        min_distance::Union{Real, Nothing}=nothing
    ) -> StaticGenerator{OperatorSum{W90Hoppings, NTuple{3, Int}}}

Convenience method: read lattice, centers, and Hamiltonian from Wannier90 output files,
then filter by hopping norm and distance thresholds.

See also: [`choosehamiltonian`](@ref) for the direct-input version.
"""
function choosehamiltonian(
    path::AbstractString,
    seedname::AbstractString;
    name::Symbol=Symbol(seedname),
    projection::Bool=true,
    min_hopping_norm::Real=0.0,
    max_hopping_norm::Union{Real, Nothing}=nothing,
    max_distance::Union{Real, Nothing}=nothing,
    min_distance::Union{Real, Nothing}=nothing
)
    lattice = readlattice(path, seedname; name=name, projection=projection)
    centers = readcenters(path, seedname)
    ham = readhamiltonian(path, seedname)
    return choosehamiltonian(ham, lattice, centers; min_hopping_norm, max_hopping_norm, max_distance, min_distance)
end

"""
    distancehamiltonian(
        path::AbstractString,
        seedname::AbstractString;
        name::Symbol=Symbol(seedname),
        projection::Bool=true
    ) -> (Vector{Float64}, Vector{Tuple{NTuple{3, Int}, Int, Int}}, Vector{Float64})

Convenience method: read lattice, centers, and Hamiltonian from Wannier90 output files,
then return all hopping terms with their Cartesian distances and indices.

See also: [`distancehamiltonian`](@ref) for the direct-input version.
"""
function distancehamiltonian(path::AbstractString, seedname::AbstractString; name::Symbol=Symbol(seedname), projection::Bool=true)
    lattice = readlattice(path, seedname; name=name, projection=projection)
    centers = readcenters(path, seedname)
    ham = readhamiltonian(path, seedname)
    return distancehamiltonian(ham, lattice, centers)
end

"""
    showhamiltonian(
        io::IO,
        ham::StaticGenerator{OperatorSum{W90Hoppings, NTuple{3, Int}}},
        lattice::Lattice,
        centers::Matrix{Float64},
        nsublattice::Int
    )

    showhamiltonian(
        ham::StaticGenerator{OperatorSum{W90Hoppings, NTuple{3, Int}}},
        lattice::Lattice,
        centers::Matrix{Float64},
        nsublattice::Int
    )

    showhamiltonian(
        filename::AbstractString,
        ham::StaticGenerator{OperatorSum{W90Hoppings, NTuple{3, Int}}},
        lattice::Lattice,
        centers::Matrix{Float64},
        nsublattice::Int
    )

Print the Wannier90 hopping Hamiltonian in a human-readable format.

For each R-vector and each sublattice pair `(i, j)`, prints the Cartesian
distance `norm(R_cart + centers[:, j₁] - centers[:, i₁])` between the first
orbitals, followed by the corresponding `norbit × norbit` hopping sub-matrix.

When `io` is omitted, prints to `stdout`.  When `filename` is provided,
writes to the specified file.  `nsublattice` is the number of sublattices;
each sublattice has `n_wan ÷ nsublattice` orbitals.
"""
function showhamiltonian(io::IO, ham::StaticGenerator{OperatorSum{W90Hoppings, NTuple{3, Int}}}, lattice::Lattice, centers::Matrix{Float64}, nsublattice::Int)
    n_wan = size(centers, 2)
    vectors = lattice.vectors
    norbit = div(n_wan, nsublattice)

    # ---------- header: lattice info ----------
    println(io, "# ================================================================================")
    println(io, "# Wannier90 Tight-Binding Hamiltonian")
    println(io, "# ================================================================================")
    println(io, "#")
    println(io, "# Explanation:")
    println(io, "#   R           = (n₁, n₂, n₃) : supercell translation vector in lattice-vector basis.")
    println(io, "#                                R_cart = n₁·a₁ + n₂·a₂ + n₃·a₃")
    println(io, "#   sublattice  = (si, sj)      : sublattice indices. Each sublattice spans $(norbit) orbital(s),")
    println(io, "#                                so sublattice s starts at orbital index (s-1)×$(norbit)+1.")
    println(io, "#   distance    = ∥R_cart + center_j - center_i∥  (Å)")
    println(io, "#                                Cartesian distance between Wannier orbital i (origin cell)")
    println(io, "#                                and orbital j (in cell R).")
    println(io, "#   orbital hoppings matrix      = ⟨w_{i,0}|H|w_{j,R}⟩  for orbitals on the two sites.")
    println(io, "#                                Each block is $(norbit)×$(norbit), i.e. hoppings among the orbitals")
    println(io, "#                                belonging to the two sublattices.")
    println(io, "#")
    println(io, "# Lattice vectors (columns, Å):")
    for (idx, a) in enumerate(vectors)
        println(io, "#   a$idx = ($(a[1]), $(a[2]), $(a[3]))")
    end
    println(io, "#")
    println(io, "# Wannier centers (orbitals × 3, Å):")
    for w in 1:n_wan
        println(io, "#   orbital $w @ ($(centers[1, w]), $(centers[2, w]), $(centers[3, w]))")
    end
    println(io, "#")
    n_R = length(ham.operators)
    println(io, "# Total Wannier orbitals = $n_wan, nsublattice = $nsublattice, norbit/sublattice = $norbit, R-vectors = $n_R")
    println(io, "# Hopping entries follow below:")
    println(io, "# ================================================================================")
    println(io)

    for m in ham.operators
        R = m.id
        R_cart = mapreduce(*, +, vectors, R)
        println(io, "# --- R = ($(R[1]), $(R[2]), $(R[3])) ---")
        for (sj, j) in enumerate(1:norbit:n_wan)
            cⱼ = SVector(centers[1, j], centers[2, j], centers[3, j])
            for (si, i) in enumerate(1:norbit:n_wan)
                cᵢ = SVector(centers[1, i], centers[2, i], centers[3, i])
                dist = norm(R_cart + cⱼ - cᵢ)
                kind = (R == (0, 0, 0) && si == sj) ? "(on-site)" : "(hopping)"
                println(io, "#   sublattice($si,$sj),  distance = $(round(dist, digits=4))  $kind")
                show(io, MIME"text/plain"(), m.value[i:i+norbit-1, j:j+norbit-1])
                println(io)
                println(io)
            end
        end
    end
end
showhamiltonian(ham::StaticGenerator{OperatorSum{W90Hoppings, NTuple{3, Int}}}, lattice::Lattice, centers::Matrix{Float64}, nsublattice::Int) = showhamiltonian(stdout, ham, lattice, centers, nsublattice)

function showhamiltonian(filename::AbstractString, ham::StaticGenerator{OperatorSum{W90Hoppings, NTuple{3, Int}}}, lattice::Lattice, centers::Matrix{Float64}, nsublattice::Int)
    open(filename, "w") do io
        showhamiltonian(io, ham, lattice, centers, nsublattice)
    end
end


"""
    W90Matrixization{V<:AbstractVector{<:Real}} <: Matrixization

Matrixization of the Hamiltonian obtained from Wannier90.
"""
struct W90Matrixization{V<:AbstractVector{<:Real}} <: Matrixization
    k::V
    vectors::SVector{3, SVector{3, Float64}}
    centers::Matrix{Float64}
    gauge::Symbol
    function W90Matrixization(k::AbstractVector{<:Real}, vectors::AbstractVector{<:AbstractVector{<:Real}}, centers::AbstractMatrix{<:Real}, gauge::Symbol)
        @assert length(k)==3 "W90Matrixization error: the length of `k` must be 3."
        @assert length(vectors)==3 && all(v->length(v)==3, vectors) "W90Matrixization error: `vectors` must be 3 length-3 vectors."
        @assert size(centers)[1]==3 "W90Matrixization error: the row number of `centers` must be 3."
        @assert gauge∈(:icoordinate, :rcoordinate) "W90Matrixization error: `gauge` must be either `:icoordinate` or `:rcoordinate`."
        new{typeof(k)}(k, vectors, centers, gauge)
    end
end
@inline Base.valtype(::Type{<:W90Matrixization}, ::Type{<:Union{W90Hoppings, OperatorSet{<:W90Hoppings}}}) = Matrix{ComplexF64}
@inline Base.zero(mr::W90Matrixization, m::Union{W90Hoppings, OperatorSet{<:W90Hoppings}}) = zeros(eltype(valtype(mr, m)), size(mr.centers)[2], size(mr.centers)[2])
@inline (mr::W90Matrixization)(m::W90Hoppings; kwargs...) = add!(zero(mr, m), mr, m; kwargs...)
function add!(dest::AbstractMatrix, mr::W90Matrixization, m::W90Hoppings; kwargs...)
    @assert size(dest)[1]==size(dest)[2]==size(mr.centers)[2]==size(m.value)[1] "add! error: mismatch occurs."
    icoordinate = mapreduce(*, +, mr.vectors, m.id)
    if mr.gauge == :icoordinate
        phase = exp(1im*dot(mr.k, icoordinate))
        for index in eachindex(dest)
            dest[index] += m.value[index] * phase
        end
    else
        for i = 1:size(mr.centers)[2]
            centerᵢ = SVector(mr.centers[1, i], mr.centers[2, i], mr.centers[3, i])
            for j = 1:size(mr.centers)[2]
                centerⱼ = SVector(mr.centers[1, j], mr.centers[2, j], mr.centers[3, j])
                phase = exp(1im*dot(mr.k, icoordinate+centerⱼ-centerᵢ))
                dest[i, j] += m.value[i, j] * phase
            end
        end
    end
    return dest
end

"""
    W90 <: TBA{Fermionic{:TBA}, StaticGenerator{OperatorSum{W90Hoppings, NTuple{3, Int}}}, Nothing}

A quantum lattice system based on the information obtained from Wannier90.
"""
struct W90 <: TBA{Fermionic{:TBA}, StaticGenerator{OperatorSum{W90Hoppings, NTuple{3, Int}}}, Nothing}
    lattice::Lattice{3, Float64, 3}
    centers::Matrix{Float64}
    H::StaticGenerator{OperatorSum{W90Hoppings, NTuple{3, Int}}}
    function W90(lattice::Lattice{3, <:Real, 3}, centers::AbstractMatrix{<:Real}, H::StaticGenerator{OperatorSum{W90Hoppings, NTuple{3, Int}}})
        @assert size(centers)[1]==3 "W90 error: the row number of `centers` must be 3."
        @assert size(centers)[2]==size(first(H).value)[1] "W90 error: mismatched size of Wannier centers and hoppings."
        new(lattice, centers, H)
    end
end
@inline dimension(wan::W90) = size(wan.centers)[2]
@inline update!(wan::W90; parameters...) = wan
@inline function matrix(wan::W90, k::AbstractVector{<:Number}=SVector(0, 0, 0); gauge=:icoordinate, kwargs...)
    m = W90Matrixization(k, wan.lattice.vectors, wan.centers, gauge)(wan.H.operators; kwargs...)
    return TBAMatrix(Hermitian(m), nothing)
end

"""
    W90(lattice::Lattice, centers::Matrix{<:Real}, H::StaticGenerator{<:OperatorSum{W90Hoppings}})
    W90(lattice::Lattice, hilbert::Hilbert, H::StaticGenerator{<:OperatorSum{W90Hoppings}})

Construct a quantum lattice system based on the information obtained from Wannier90.

In general, the Wannier centers could deviate from their corresponding atom positions.
When `centers::Matrix{<:Real}` is used, the Wannier centers are assigned directly.
When `hilbert::Hilbert` is used, the Wannier centers will be approximated by their corresponding atom positions.
"""
function W90(lattice::Lattice, hilbert::Hilbert, H::StaticGenerator{<:OperatorSum{W90Hoppings}})
    table = Table(hilbert, OperatorIndexToTuple(:site, :orbital, :spin))
    centers = zeros(Float64, 3, length(table))
    for (key, index) in pairs(table)
        centers[:, index] = lattice[key[1]]
    end
    return W90(lattice, centers, H)
end

"""
    W90(path::AbstractString, seedname::AbstractString; name::Symbol=Symbol(seedname), projection::Bool=true)

Construct a quantum lattice system based on the files obtained from Wannier90.
"""
function W90(path::AbstractString, seedname::AbstractString; name::Symbol=Symbol(seedname), projection::Bool=true)
    lattice = readlattice(path, seedname; name=name, projection=projection)
    centers = readcenters(path, seedname)
    H = readhamiltonian(path, seedname)
    return W90(lattice, centers, H)
end

"""
    showW90(io::IO, wan::W90, nsublattice::Int)

    showW90(wan::W90, nsublattice::Int)

    showW90(filename::AbstractString, wan::W90, nsublattice::Int)

Print the Wannier90 hopping Hamiltonian stored in a `W90` system in a human-readable format.

When `io` is omitted, prints to `stdout`.
When `filename` is provided, writes to the specified file.
`nsublattice` is the number of sublattices.
"""
showW90(io::IO, wan::W90, nsublattice::Int) = showhamiltonian(io, wan.H, wan.lattice, wan.centers, nsublattice)
showW90(wan::W90, nsublattice::Int) = showhamiltonian(stdout, wan.H, wan.lattice, wan.centers, nsublattice)
showW90(filename::AbstractString, wan::W90, nsublattice::Int) = showhamiltonian(filename, wan.H, wan.lattice, wan.centers, nsublattice)

"""
    readbands(path::AbstractString, seedname::AbstractString) -> Tuple{Vector{Float64}, Matrix{Float64}}

Read the energy bands from the Wannier90 "_band.dat" output data file when the "bands_plot" parameter in the Wannier90 ".win" input file is set to be true.
"""
function readbands(path::AbstractString, seedname::AbstractString)
    nk = open(joinpath(path, seedname*"_band.dat"), "r") do f
        count = 0
        while !eof(f)
            line = readline(f; keep=true)
            all(isspace, line) && return count
            count += 1
        end
        error("readbands error: mismatched file organization format.")
    end
    content = readdlm(joinpath(path, seedname*"_band.dat"))
    return content[1:nk, 1], reshape(content[:, 2], nk, :)
end

"""
    readpath(path::AbstractString, seedname::AbstractString; length=nothing) -> ReciprocalPath

Read the reciprocal path along which to plot the energy bands from the Wannier90 ".win" input file.
"""
function readpath(path::AbstractString, seedname::AbstractString; length=nothing)
    lattice = readlattice(path, seedname)
    content = open(io->read(io, String), joinpath(path, seedname*".win"), "r")
    lines = findblock("kpoint_path", content)
    isnothing(lines) && error("readpath error: unable to find \"kpoint_path\" block in the Wannier90 \".win\" input file.")
    segments, labels = Pair{NTuple{3, Float64}, NTuple{3, Float64}}[], Pair{String, String}[]
    duplicate = Dict{String, Vector{NTuple{3, Float64}}}()
    for line in lines
        tokens = split(line)
        @assert Base.length(tokens)==8 "readpath error: incorrect format."
        spoint = (parse(Float64, tokens[2]), parse(Float64, tokens[3]), parse(Float64, tokens[4]))
        epoint = (parse(Float64, tokens[6]), parse(Float64, tokens[7]), parse(Float64, tokens[8]))
        slabel = resolve_duplicate!(duplicate, tokens[1], spoint)
        elabel = resolve_duplicate!(duplicate, tokens[5], epoint)
        push!(segments, spoint=>epoint)
        push!(labels, slabel=>elabel)
    end
    if isnothing(length)
        token = match(r"bands_num_points\s*=\s*(\d+)"i, content)
        length = isnothing(token) ? 100 : parse(Int, token[1])
    end
    return ReciprocalPath(lattice, segments...; labels=labels, length=length)
end
function resolve_duplicate!(duplicate::Dict{String, Vector{NTuple{3, Float64}}}, token::AbstractString, value::NTuple{3, Float64})
    pool = get(duplicate, token, nothing)
    if isnothing(pool)
        duplicate[token] = [value]
        return token
    else
        index = findfirst(==(value), duplicate[token])
        if isnothing(index)
            push!(pool, value)
            index = length(pool)
        end
        return string(token, "′"^(index-1))
    end
end

"""
    Operatorization <: LinearTransformation

Operatorize the hopping amplitudes among Wannier orbitals.
"""
struct Operatorization <: LinearTransformation
    centers::Matrix{Float64}
    vectors::SVector{3, SVector{3, Float64}}
    table::Dict{Int, Tuple{Int, Int, Rational{Int}}}
end
@inline function Base.valtype(::Type{<:Operatorization}, ::Type{<:Union{W90Hoppings, OperatorSet{<:W90Hoppings}}})
    I = CoordinatedIndex{Index{FockIndex{:f, Int, Rational{Int}}, Int}, SVector{3, Float64}}
    return Operators{Operator{ComplexF64, Tuple{I, I}}, Tuple{I, I}}
end
@inline (operatorization::Operatorization)(m::W90Hoppings; kwargs...) = add!(zero(operatorization, m), operatorization, m; kwargs...)
function add!(dest::Operators, operatorization::Operatorization, m::W90Hoppings; tol::Real=1e-6, complement_spin::Bool=false, kwargs...)
    icoordinate = mapreduce(*, +, operatorization.vectors, m.id)
    for j in axes(m.value, 2)
        for i in axes(m.value, 1)
            v = m.value[i, j]
            if abs(v)>tol
                siteᵢ, orbitalᵢ, spinᵢ = operatorization.table[i]
                siteⱼ, orbitalⱼ, spinⱼ = operatorization.table[j]
                rcoordinateᵢ = SVector(operatorization.centers[1, i], operatorization.centers[2, i], operatorization.centers[3, i])
                rcoordinateⱼ = SVector(operatorization.centers[1, j], operatorization.centers[2, j], operatorization.centers[3, j]) + icoordinate
                if complement_spin && (spinᵢ==spinⱼ==0)
                    c⁺ = 𝕔⁺(siteᵢ, orbitalᵢ, 1//2, rcoordinateᵢ, SVector(0.0, 0.0, 0.0))
                    c = 𝕔(siteⱼ, orbitalⱼ, 1//2, rcoordinateⱼ, icoordinate)
                    add!(dest, Operator(v, c⁺, c))
                    c⁺ = 𝕔⁺(siteᵢ, orbitalᵢ, -1//2, rcoordinateᵢ, SVector(0.0, 0.0, 0.0))
                    c = 𝕔(siteⱼ, orbitalⱼ, -1//2, rcoordinateⱼ, icoordinate)
                    add!(dest, Operator(v, c⁺, c))
                else
                    c⁺ = 𝕔⁺(siteᵢ, orbitalᵢ, spinᵢ, rcoordinateᵢ, SVector(0.0, 0.0, 0.0))
                    c = 𝕔(siteⱼ, orbitalⱼ, spinⱼ, rcoordinateⱼ, icoordinate)
                    add!(dest, Operator(v, c⁺, c))
                end
            end
        end
    end
    return dest
end

"""
    Operatorization(centers::AbstractMatrix{<:Number}, vectors::AbstractVector{<:AbstractVector{<:Number}}, hilbert::Hilbert)

Construct a `Operatorization`.
"""
@inline function Operatorization(centers::AbstractMatrix{<:Number}, vectors::AbstractVector{<:AbstractVector{<:Number}}, hilbert::Hilbert)
    table = Table(hilbert, OperatorIndexToTuple(:site, :orbital, :spin))
    @assert extrema(values(table))==(1, size(centers, 2)) "Operatorization error: mismatched centers and hilbert space."
    return Operatorization(centers, vectors, Dict(index=>key for (key, index) in pairs(table)))
end

"""
    TBA(wan::W90, hilbert::Hilbert; complement_spin::Bool=false, tol::Real=1e-6)

Convert a Wannier90 tight-binding system to the operator formed one.
"""
function TBA(wan::W90, hilbert::Hilbert; complement_spin::Bool=false, tol::Real=1e-6)
    operatorization = Operatorization(wan.centers, wan.lattice.vectors, hilbert)
    indexes = Index{FockIndex{:f, Int, Rational{Int}}, Int}[]
    for (site, orbital, spin) in values(operatorization.table)
        if complement_spin && spin==0
            push!(indexes, 𝕔(site, orbital, 1//2))
            push!(indexes, 𝕔(site, orbital, -1//2))
        else
            push!(indexes, 𝕔(site, orbital, spin))
        end
    end
    table = Table(indexes, OperatorIndexToTuple(:site, :orbital, :spin))
    H = operatorization(wan.H; complement_spin, tol)
    quadraticization = Quadraticization{Fermionic{:TBA}}(table)
    return TBA{Fermionic{:TBA}}(wan.lattice, H, quadraticization)
end

"""
    Algorithm(wan::Algorithm{W90}, hilbert::Hilbert; complement_spin::Bool=false, tol::Real=1e-6)

Convert a Wannier90 tight-binding system to the operator formed one.
"""
@inline function Algorithm(wan::Algorithm{W90}, hilbert::Hilbert; complement_spin::Bool=false, tol::Real=1e-6)
    return Algorithm(wan.name, TBA(wan.frontend, hilbert; complement_spin, tol), wan.parameters, wan.map; dir=wan.dir)
end

end
