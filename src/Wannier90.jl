module Wannier90

using DelimitedFiles: readdlm
using LinearAlgebra: Hermitian, dot, norm
using QuantumLattices: CoordinatedIndex, FockIndex, Hilbert, Index, Lattice, LinearTransformation, Matrixization, Operator, OperatorIndexToTuple, OperatorPack, Operators, OperatorSet, OperatorSum, ReciprocalPath, StaticGenerator, Table, decompose
using QuantumLattices: 𝕔, 𝕔⁺
using StaticArrays: SVector
using ..TightBindingApproximation: Fermionic, Quadraticization, TBAMatrix

import QuantumLattices: Algorithm, add!, dimension, matrix
import ..TightBindingApproximation: TBA

export Operatorization, W90, W90Hoppings, W90Matrixization, choosehamiltonian, distancehamiltonian, foldcenters, showW90, showhamiltonian, findblock, readbands, readcenters, readhamiltonian, readlattice, readpath

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
    foldcenters(centers::AbstractMatrix{<:Real}, lattice::Lattice; atol::Real=1e-8) -> Matrix{Float64}

Fold each Wannier centre into the first unit cell of `lattice`.

Each column of `centers` is the Cartesian coordinate of one Wannier function. The
coordinates written by Wannier90 in "_centres.xyz" are only defined modulo a lattice
translation, so different Wannier functions of the same site may be given in different
periodic images. The inter-site distance `norm(R_cart + centers[:, j] - centers[:, i])`
is then shifted by a lattice vector, which makes the distance filters of
[`choosehamiltonian`](@ref) and the decay plot of [`distancehamiltonian`](@ref) select
the wrong `H(R)` blocks.

Given the fractional coordinates `f = decompose(c, vectors)`, the result is
`c - Σᵢ floor(fᵢ + atol) * vectorsᵢ`, so every centre ends up in the first unit cell.
The block index `R` of `H(R)` is a cell displacement and is therefore not affected: it
is the folded (physical) position of the home-cell Wannier function that enters the
distance formula.

`atol` absorbs the round-off of the Cartesian-to-fractional conversion, so a centre at a
cell boundary (`f ≈ 1`) is folded to `0` instead of staying at `1`.

See also: [`readcenters`](@ref), [`choosehamiltonian`](@ref), [`distancehamiltonian`](@ref).
"""
function foldcenters(centers::AbstractMatrix{<:Real}, lattice::Lattice; atol::Real = 1e-8)
    size(centers, 1) == 3 || error("foldcenters error: `centers` must have 3 rows, got $(size(centers, 1)).")
    vectors = lattice.vectors
    result = Matrix{Float64}(undef, 3, size(centers, 2))
    for i in 1:size(centers, 2)
        c = SVector{3, Float64}(centers[1, i], centers[2, i], centers[3, i])
        f = decompose(c, vectors)
        n = ntuple(k -> floor(f[k] + atol), Val(3))
        result[:, i] = c - sum(vectors[k] * n[k] for k in 1:3)
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
        min_distance::Union{Real, Nothing} = nothing,
        fold::Bool = true
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
- `fold::Bool = true`: fold every Wannier centre into the first unit cell with
  [`foldcenters`](@ref) before measuring the distances. The centres returned by
  [`readcenters`](@ref) are only defined modulo a lattice translation, so an unfolded
  centre would shift the distance of a hopping by a lattice vector and select the wrong
  `H(R)` blocks. Set `fold = false` only if the input centres are already folded and
  must be used exactly as given.
"""
function choosehamiltonian(
    ham::StaticGenerator{OperatorSum{W90Hoppings, NTuple{3, Int}}},
    lattice::Lattice,
    centers::Matrix{Float64};
    min_hopping_norm::Real = 0.0,
    max_hopping_norm::Union{Real, Nothing} = nothing,
    max_distance::Union{Real, Nothing} = nothing,
    min_distance::Union{Real, Nothing} = nothing,
    fold::Bool = true
)
    n_wan = size(centers, 2)
    @assert size(centers, 1) == 3 "choosehamiltonian error: centers must have 3 rows."
    @assert n_wan == size(first(ham).value, 1) "choosehamiltonian error: mismatched sizes of centers and hopping matrices."

    centers = fold ? foldcenters(centers, lattice) : centers

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
        centers::Matrix{Float64};
        fold::Bool = true
    ) -> (Vector{Float64}, Vector{Tuple{NTuple{3, Int}, Int, Int}}, Vector{Float64})

Return all hopping terms with their Cartesian distances and indices for diagnostic purposes.

For well localized Wannier functions, the hopping amplitude should decay exponentially with
distance. This function returns the data needed to check this behavior by plotting the
hopping amplitudes as a function of distance.

# Returns
- `distances::Vector{Float64}`: Cartesian distances between orbital `i` at origin cell and orbital `j` at cell `R`.
- `indices::Vector{Tuple{NTuple{3, Int}, Int, Int}}`: `(R, i, j)` triplets identifying each hopping.
- `values::Vector{Float64}`: `|hopping|` amplitudes (already degeneracy-divided).

# Keyword Arguments
- `fold::Bool = true`: fold every Wannier centre into the first unit cell with
  [`foldcenters`](@ref) before measuring the distances (see [`choosehamiltonian`](@ref)).

# Example
```julia
using Plots
dists, idxs, vals = distancehamiltonian(ham, lattice, centers)
scatter(dists, log.(vals); xlabel="Distance (Å)", ylabel="log|hopping|", legend=false)
savefig("hopping_decay.pdf")
```
"""
function distancehamiltonian(ham::StaticGenerator{OperatorSum{W90Hoppings, NTuple{3, Int}}}, lattice::Lattice, centers::Matrix{Float64}; fold::Bool = true)
    centers = fold ? foldcenters(centers, lattice) : centers
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
        min_distance::Union{Real, Nothing}=nothing,
        fold::Bool=true
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
    min_distance::Union{Real, Nothing}=nothing,
    fold::Bool=true
)
    lattice = readlattice(path, seedname; name=name, projection=projection)
    centers = readcenters(path, seedname)
    ham = readhamiltonian(path, seedname)
    return choosehamiltonian(ham, lattice, centers; min_hopping_norm, max_hopping_norm, max_distance, min_distance, fold)
end

"""
    distancehamiltonian(
        path::AbstractString,
        seedname::AbstractString;
        name::Symbol=Symbol(seedname),
        projection::Bool=true,
        fold::Bool=true
    ) -> (Vector{Float64}, Vector{Tuple{NTuple{3, Int}, Int, Int}}, Vector{Float64})

Convenience method: read lattice, centers, and Hamiltonian from Wannier90 output files,
then return all hopping terms with their Cartesian distances and indices.

See also: [`distancehamiltonian`](@ref) for the direct-input version.
"""
function distancehamiltonian(path::AbstractString, seedname::AbstractString; name::Symbol=Symbol(seedname), projection::Bool=true, fold::Bool=true)
    lattice = readlattice(path, seedname; name=name, projection=projection)
    centers = readcenters(path, seedname)
    ham = readhamiltonian(path, seedname)
    return distancehamiltonian(ham, lattice, centers; fold)
end

"""
    _elementstring(x::Number) -> String

Format one matrix element at full precision. Real numbers use `string`, which
gives the shortest representation that round-trips to the stored value, so all
the digits a `Float64` actually holds are written out and no trailing zeros are
added. Complex numbers are written in the compact `re±im·i` form, and a negative
zero is always normalized to `0.0`.
"""
function _elementstring(x::Real)
    return string(iszero(x) ? abs(x) : x)
end
function _elementstring(x::Complex)
    re = iszero(real(x)) ? abs(real(x)) : real(x)
    im_ = iszero(imag(x)) ? abs(imag(x)) : imag(x)
    return string(re) * (signbit(im_) ? "-" : "+") * string(abs(im_)) * "im"
end

"""
    _showmatrix(io::IO, block::AbstractMatrix{<:Number}, digits::Union{Nothing, Int})

Print a hopping block at full precision. Julia's own `text/plain` display of a
matrix limits the number of digits of each element according to the largest
magnitude in the same column (`4.91394713456` is shown as `4.91395`), so every
element is formatted here independently with [`_elementstring`](@ref).

With `digits === nothing` the stored values are printed as they are. Otherwise
every element is first rounded to `digits` *significant* digits, which keeps all
the digits the data actually carries (`4.91394713456` stays `4.91394713456`)
while dropping binary representation noise (`4.9139470000000003` becomes
`4.913947`). Significant digits are used instead of decimal places so that no
trailing zeros are ever added.
"""
function _showmatrix(io::IO, block::AbstractMatrix{<:Number}, digits::Union{Nothing, Int})
    isnothing(digits) || 1 <= digits <= 17 ||
        error("showhamiltonian error: `digits` must be between 1 and 17, got $digits.")
    nrow, ncol = size(block)
    strings = Matrix{String}(undef, nrow, ncol)
    widths = zeros(Int, ncol)
    for j in 1:ncol, i in 1:nrow
        z = block[i, j]
        if !isnothing(digits)
            z = complex(round(real(z); sigdigits = digits), round(imag(z); sigdigits = digits))
        end
        strings[i, j] = _elementstring(z)
        widths[j] = max(widths[j], length(strings[i, j]))
    end
    println(io, "$nrow×$ncol Matrix{$(eltype(block))}:")
    for i in 1:nrow
        print(io, " ")
        for j in 1:ncol
            j > 1 && print(io, "  ")
            print(io, lpad(strings[i, j], widths[j]))
        end
        println(io)
    end
    return nothing
end

"""
    showhamiltonian(
        io::IO,
        ham::StaticGenerator{OperatorSum{W90Hoppings, NTuple{3, Int}}},
        lattice::Lattice,
        centers::Matrix{Float64},
        nsublattice::Int;
        spinors::Bool=false,
        checkcenters::Bool=true,
        digits::Union{Nothing, Int}=nothing,
        fold::Bool=true
    )

    showhamiltonian(
        ham::StaticGenerator{OperatorSum{W90Hoppings, NTuple{3, Int}}},
        lattice::Lattice,
        centers::Matrix{Float64},
        nsublattice::Int;
        kwargs...
    )

    showhamiltonian(
        filename::AbstractString,
        ham::StaticGenerator{OperatorSum{W90Hoppings, NTuple{3, Int}}},
        lattice::Lattice,
        centers::Matrix{Float64},
        nsublattice::Int;
        kwargs...
    )

Print the Wannier90 hopping Hamiltonian in a human-readable format, both without
and with spin-orbit coupling.

For each R-vector and each sublattice pair `(si, sj)`, prints the Cartesian
distance `norm(R_cart + centers[:, j₁] - centers[:, i₁])` between the first
orbitals, followed by the corresponding hopping block: `norbit × norbit` when
`spinors = false`, and `2norbit × 2norbit` when `spinors = true`.

With `fold = true` (the default) the centres are first folded into the first unit
cell with [`foldcenters`](@ref), so the printed centres and distances are the
physical ones. The centres in "_centres.xyz" are only defined modulo a lattice
translation, and an unfolded centre would shift a distance by a whole lattice
vector, making it inconsistent with the cutoff used by [`choosehamiltonian`](@ref).
Set `fold = false` to print the centres exactly as they were passed in.

`nsublattice` is the number of sublattices. Without spin-orbit coupling each
sublattice has `norbit = num_wan ÷ nsublattice` Wannier functions. With
spin-orbit coupling (`spinors = true`) each sublattice carries `norbit` spatial
orbitals, so `num_wan = 2 * nsublattice * norbit`, and the rows and columns of
each block follow the Wannier90 spinor order `↑o₁, ↓o₁, ↑o₂, ↓o₂, …`: for each
orbital the spin-up component comes first, followed by the spin-down component
of the same orbital.

With `spinors = true` and `checkcenters = true`, a warning is issued when the
two spinor members of a pair do not share a Wannier centre, which usually means
that `spinors`, `nsublattice` or the index order of the `_hr.dat` file is
inconsistent with this convention.

Every matrix element is printed at full precision, i.e. with all the digits the
stored `Float64` actually holds, without trailing zeros or artificial padding.
(Julia's own matrix display shortens the elements according to the largest
magnitude in the same column, which would turn `4.91394713456` into `4.91395`.)

`digits` optionally limits the precision. With `digits = nothing` (the default)
the stored values are printed as they are. With an integer `digits = n`
(1 ≤ n ≤ 17) every element is first rounded to `n` *significant* digits.
`digits = 16` is the usual choice: it keeps every digit the data carries up to
the precision of `Float64` while removing the binary representation noise of
numbers read from a text file -- `4.9139470000000003` is printed as `4.913947`
and `0.30000000000000004` as `0.3`. Since significant digits are used rather
than decimal places, no trailing zeros are ever added.

When `io` is omitted, prints to `stdout`.  When `filename` is provided,
writes to the specified file.
"""
function showhamiltonian(io::IO, ham::StaticGenerator{OperatorSum{W90Hoppings, NTuple{3, Int}}}, lattice::Lattice, centers::Matrix{Float64}, nsublattice::Int; spinors::Bool = false, checkcenters::Bool = true, digits::Union{Nothing, Int} = nothing, fold::Bool = true)
    n_wan = size(centers, 2)
    vectors = lattice.vectors
    centers = fold ? foldcenters(centers, lattice) : centers
    nsublattice >= 1 ||
        error("showhamiltonian error: `nsublattice` must be positive, got $nsublattice.")
    if spinors
        n_wan % (2 * nsublattice) == 0 ||
            error("showhamiltonian error: with `spinors = true` num_wan = $n_wan must be divisible by 2 × nsublattice = $(2 * nsublattice).")
        norbit = div(n_wan, 2 * nsublattice)
        nblock = 2 * norbit
    else
        n_wan % nsublattice == 0 ||
            error("showhamiltonian error: num_wan = $n_wan must be divisible by nsublattice = $nsublattice.")
        norbit = div(n_wan, nsublattice)
        nblock = norbit
    end

    if spinors && checkcenters
        mismatched = Int[]
        for s in 1:nsublattice, k in 1:norbit
            i, j = (s - 1) * nblock + 2k - 1, (s - 1) * nblock + 2k
            scale = 1 + maximum(abs, centers[:, i])
            maximum(abs, centers[:, i] - centers[:, j]) > 1e-3 * scale &&
                push!(mismatched, j)
        end
        isempty(mismatched) ||
            @warn "showhamiltonian warning: the spinor pairs ending at the Wannier indices $(mismatched) do not share a Wannier centre; check `spinors`, `nsublattice` and the index order of `_hr.dat`."
    end

    # ---------- header: lattice info ----------
    println(io, "# ================================================================================")
    println(io, "# Wannier90 Tight-Binding Hamiltonian" * (spinors ? " (spin-orbit coupling)" : ""))
    println(io, "# ================================================================================")
    println(io, "#")
    println(io, "# Explanation:")
    println(io, "#   R           = (n₁, n₂, n₃) : supercell translation vector in lattice-vector basis.")
    println(io, "#                                R_cart = n₁·a₁ + n₂·a₂ + n₃·a₃")
    println(io, "#   sublattice  = (si, sj)      : sublattice indices. Each sublattice spans $(nblock) $(spinors ? "spinor orbital(s)" : "orbital(s)"),")
    println(io, "#                                so sublattice s starts at orbital index (s-1)×$(nblock)+1.")
    if spinors
        println(io, "#   spin order  = ↑o₁, ↓o₁, ↑o₂, ↓o₂, …")
        println(io, "#                                for each orbital the spin-up component comes first,")
        println(io, "#                                then the spin-down component of the same orbital.")
    end
    println(io, "#   distance    = ∥R_cart + center_j - center_i∥  (Å)")
    println(io, "#                                Cartesian distance between Wannier orbital i (origin cell)")
    println(io, "#                                and orbital j (in cell R).")
    println(io, "#   orbital hoppings matrix      = ⟨w_{i,0}|H|w_{j,R}⟩  for orbitals on the two sites.")
    println(io, "#                                Each block is $(nblock)×$(nblock), i.e. hoppings among the orbitals")
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
    println(io, "# Total Wannier orbitals = $n_wan, nsublattice = $nsublattice, norbit/sublattice = $norbit, " * (spinors ? "spins = 2, " : "") * "R-vectors = $n_R")
    println(io, "# Hopping entries follow below:")
    println(io, "# ================================================================================")
    println(io)

    for m in ham.operators
        R = m.id
        R_cart = mapreduce(*, +, vectors, R)
        println(io, "# --- R = ($(R[1]), $(R[2]), $(R[3])) ---")
        for (sj, j) in enumerate(1:nblock:n_wan)
            cⱼ = SVector(centers[1, j], centers[2, j], centers[3, j])
            for (si, i) in enumerate(1:nblock:n_wan)
                cᵢ = SVector(centers[1, i], centers[2, i], centers[3, i])
                dist = norm(R_cart + cⱼ - cᵢ)
                kind = (R == (0, 0, 0) && si == sj) ? "(on-site)" : "(hopping)"
                println(io, "#   sublattice($si,$sj),  distance = $(round(dist, digits=4))  $kind")
                _showmatrix(io, m.value[i:i+nblock-1, j:j+nblock-1], digits)
                println(io)
                println(io)
            end
        end
    end
end
showhamiltonian(ham::StaticGenerator{OperatorSum{W90Hoppings, NTuple{3, Int}}}, lattice::Lattice, centers::Matrix{Float64}, nsublattice::Int; kwargs...) = showhamiltonian(stdout, ham, lattice, centers, nsublattice; kwargs...)

function showhamiltonian(filename::AbstractString, ham::StaticGenerator{OperatorSum{W90Hoppings, NTuple{3, Int}}}, lattice::Lattice, centers::Matrix{Float64}, nsublattice::Int; kwargs...)
    open(filename, "w") do io
        showhamiltonian(io, ham, lattice, centers, nsublattice; kwargs...)
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
    W90(path::AbstractString, seedname::AbstractString; name::Symbol=Symbol(seedname), projection::Bool=true, fold::Bool=true)

Construct a quantum lattice system based on the files obtained from Wannier90.

With `fold = true` (the default) the centres read from "_centres.xyz" are folded into the
first unit cell with [`foldcenters`](@ref), so `wan.centers` holds physical in-cell
coordinates. The centres of the file are only defined modulo a lattice translation, and
an unfolded centre would shift the distances used by [`choosehamiltonian`](@ref),
[`distancehamiltonian`](@ref) and [`showhamiltonian`](@ref) by a whole lattice vector.
The band structure is not affected: the phase of a folded centre differs by
`exp(2πi · n)` in the `:rcoordinate` gauge. Set `fold = false` to keep the file values.
"""
function W90(path::AbstractString, seedname::AbstractString; name::Symbol=Symbol(seedname), projection::Bool=true, fold::Bool=true)
    lattice = readlattice(path, seedname; name=name, projection=projection)
    centers = readcenters(path, seedname)
    fold && (centers = foldcenters(centers, lattice))
    H = readhamiltonian(path, seedname)
    return W90(lattice, centers, H)
end

"""
    showW90(io::IO, wan::W90, nsublattice::Int; spinors::Bool=false, checkcenters::Bool=true, digits::Union{Nothing, Int}=nothing)

    showW90(wan::W90, nsublattice::Int; kwargs...)

    showW90(filename::AbstractString, wan::W90, nsublattice::Int; kwargs...)

Print the Wannier90 hopping Hamiltonian stored in a `W90` system in a human-readable format,
both without and with spin-orbit coupling.

When `spinors = false` each sublattice has `num_wan ÷ nsublattice` Wannier functions and each
`(R, si, sj)` block is `norbit × norbit`. When `spinors = true` each sublattice carries
`norbit` spatial orbitals (`num_wan = 2 * nsublattice * norbit`), each block is
`2norbit × 2norbit`, and the rows and columns follow the Wannier90 spinor order
`↑o₁, ↓o₁, ↑o₂, ↓o₂, …`. See [`showhamiltonian`](@ref) for the details.

All matrix elements are printed at full double precision, with no trailing
zeros. `digits = nothing` (default) prints the stored values as they are; an
integer `digits = n` (1 ≤ n ≤ 17) rounds every element to `n` significant digits
first, and `digits = 16` gives full double precision while removing
binary-representation noise.

When `io` is omitted, prints to `stdout`.
When `filename` is provided, writes to the specified file.
`nsublattice` is the number of sublattices.

With `fold = true` (the default) the Wannier centres are folded into the first unit cell
with [`foldcenters`](@ref) before they are printed and before the distances are computed;
`fold = false` prints the centres of `wan.centers` exactly as they are.
"""
showW90(io::IO, wan::W90, nsublattice::Int; kwargs...) = showhamiltonian(io, wan.H, wan.lattice, wan.centers, nsublattice; kwargs...)
showW90(wan::W90, nsublattice::Int; kwargs...) = showhamiltonian(stdout, wan.H, wan.lattice, wan.centers, nsublattice; kwargs...)
showW90(filename::AbstractString, wan::W90, nsublattice::Int; kwargs...) = showhamiltonian(filename, wan.H, wan.lattice, wan.centers, nsublattice; kwargs...)

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

# ---------------------------------------------------------------------------
# Wannier-gauge transformation
# ---------------------------------------------------------------------------

using LinearAlgebra: I

export gaugetransform

"""
    gaugetransform(wan::W90, U::AbstractMatrix{<:Number};
                   atol::Real=1e-8, checkunitary::Bool=true) -> W90

Return a new `W90` whose Wannier functions are rotated by the unitary `U`:

    H′(R) = U' * H(R) * U    for every R,

i.e. the basis change `w′_i = Σ_j w_j U[j, i]`, so that a Bloch matrix built with
`matrix(wan, k; gauge = :icoordinate)` transforms as `M′(k) = U' * M(k) * U`.

The **same** `U` is applied to the onsite block `H(0)` and to every hopping block
`H(R)` at once. Wannier90 blocks live in one particular Wannier gauge; rotating
the onsite and hopping blocks independently destroys it.

The input `wan` is not modified: a fresh `OperatorSum` and a copy of `centers` are
returned. The Wannier centres themselves are **kept unchanged**, not recomputed.
A general `U` mixes Wannier functions with different centres, so the returned
centres are only an approximation to the centres of the new functions. Functions
that use the centres -- `matrix(wan, k; gauge = :rcoordinate)`,
`Operatorization`/`TBA`, `choosehamiltonian`, `distancehamiltonian`, `showW90`
and the site mapping of the `Wannier90SOPT` bridges -- remain exact when `U` is
block diagonal over groups of co-centred Wannier functions (the usual per-site
case), and are approximate otherwise.

Only the one-body blocks are rotated. Interaction terms such as `Hubbard` or the
Kanamori terms are not touched: if `U` represents a physical rotation of the
orbitals, the interaction tensors must be transformed separately.

# Arguments
- `U`: an `n × n` matrix, `n = num_wan`. It is converted to `ComplexF64` once.
- `atol`: absolute tolerance of the unitarity check.
- `checkunitary`: if `true` (default), `U' * U ≈ I` is required. If `false`, no
  check is made; the result is then a general congruence `U' H U`, which is not a
  basis change and does not preserve the eigenvalues.

# Notes
- For any `U`, `H′(-R) = U' H(-R) U = (U' H(R) U)' = H′(R)'`, so the `-R`
  pairing and the Hermiticity of `H(0)` are preserved automatically and are not
  re-checked.
- With `gauge = :rcoordinate` the Fourier phase involves the centres. Since the
  centres are kept, `M′_rco = U' M_rco U` holds exactly only when `U` commutes
  with `diag(exp(im .* (k ⋅ c)))` on every rotated block.

See `gaugetransform(wan, siteindices, rotations)` for the per-site form.
"""
function gaugetransform(wan::W90, U::AbstractMatrix{<:Number};
                        atol::Real = 1e-8, checkunitary::Bool = true)
    n = dimension(wan)
    size(U) == (n, n) ||
        error("gaugetransform error: `U` must be $n × $n for $n Wannier functions, got $(size(U)).")
    Umat = Matrix{ComplexF64}(U)
    if checkunitary
        deviation = norm(Umat' * Umat - I)
        deviation <= atol ||
            error("gaugetransform error: `U` is not unitary within atol = $atol (norm(U'U - I) = $deviation).")
    end
    ham = OperatorSum{W90Hoppings}()
    for (R, hop) in wan.H.operators.contents
        size(hop.value) == (n, n) ||
            error("gaugetransform error: the block H($R) has size $(size(hop.value)), expected ($n, $n).")
        ham.contents[R] = W90Hoppings(Umat' * hop.value * Umat, R)
    end
    return W90(wan.lattice, copy(wan.centers), StaticGenerator(ham))
end

"""
    gaugetransform(wan::W90, siteindices::AbstractDict, rotations::AbstractDict;
                   atol::Real=1e-8, checkunitary::Bool=true) -> W90

Apply a **different** unitary rotation on each site and return the new `W90`.

`siteindices` maps a site to its Wannier indices in the local basis order, e.g.
`Dict(1 => [1, 2], 2 => [3, 4])`. Every Wannier index `1:num_wan` must appear
exactly once.

`rotations` maps a site to its local unitary, e.g. `Dict(1 => R₁, 2 => R₂)` where
`R₁` is `2 × 2` for `siteindices[1] = [1, 2]`. The `j`-th row and column of
`rotations[site]` refer to `siteindices[site][j]`. Sites absent from `rotations`
are left unchanged.

The rotations of different sites may differ. The resulting full transformation is
block diagonal in the site partition, so the block structure of `H(R)` is
preserved:

    H′(R)[s, t] = rotations[s]' * H(R)[s, t] * rotations[t].

All `H(R)` blocks, onsite and hopping, are transformed in one call, which keeps
the data in a single Wannier gauge. See `gaugetransform(wan, U)` for the
conventions, the treatment of the Wannier centres and the interaction terms.

If `rotations` is a function instead of a dictionary, it is called as
`rotations(site)` for every site of `siteindices`, so it must also return the
identity for the sites that should stay unchanged.
"""
function gaugetransform(wan::W90, siteindices::AbstractDict, rotations::AbstractDict;
                        atol::Real = 1e-13, checkunitary::Bool = true)
    n = dimension(wan)
    used = falses(n)
    blocks = Tuple{Vector{Int}, Matrix{ComplexF64}}[]
    for (site, value) in siteindices
        value isa AbstractSet &&
            error("gaugetransform error: the Wannier indices of site $site must be an ordered collection, not a $(typeof(value)).")
        indices = Int[]
        for w in value
            w isa Integer ||
                error("gaugetransform error: the Wannier indices of site $site must be integers, got $(repr(w)).")
            1 <= w <= n ||
                error("gaugetransform error: site $site uses the Wannier index $w, which is outside 1:$n.")
            used[w] &&
                error("gaugetransform error: the Wannier index $w is used by more than one site.")
            used[w] = true
            push!(indices, Int(w))
        end
        haskey(rotations, site) || continue
        block = rotations[site]
        block isa AbstractMatrix ||
            error("gaugetransform error: the rotation of site $site must be an AbstractMatrix, got $(typeof(block)).")
        size(block) == (length(indices), length(indices)) ||
            error("gaugetransform error: the rotation of site $site must be $(length(indices)) × $(length(indices)) for the Wannier indices $indices, got $(size(block)).")
        Rmat = Matrix{ComplexF64}(block)
        if checkunitary
            deviation = norm(Rmat' * Rmat - I)
            deviation <= atol ||
                error("gaugetransform error: the rotation of site $site is not unitary within atol = $atol (norm(R'R - I) = $deviation).")
        end
        push!(blocks, (indices, Rmat))
    end
    for site in keys(rotations)
        haskey(siteindices, site) ||
            error("gaugetransform error: a rotation was given for the unknown site $site; `siteindices` has the sites $(collect(keys(siteindices))).")
    end
    all(used) ||
        error("gaugetransform error: `siteindices` does not cover every Wannier index; missing $(findall(x -> !x, used)).")
    U = Matrix{ComplexF64}(I, n, n)
    for (indices, Rmat) in blocks
        U[indices, indices] = Rmat
    end
    return gaugetransform(wan, U; atol = atol, checkunitary = false)
end

function gaugetransform(wan::W90, siteindices::AbstractDict, rotations::Function;
                        atol::Real = 1e-8, checkunitary::Bool = true)
    table = Dict(site => rotations(site) for site in keys(siteindices))
    return gaugetransform(wan, siteindices, table; atol = atol, checkunitary = checkunitary)
end
end
