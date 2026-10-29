"""
    Wannier90SOPT

Bridge between [`Wannier90`](@ref) hopping data and the operator generators
used by `SecondOrderPerturbationTheory.SOPT`.

`SOPT` requires its unperturbed and perturbed Hamiltonians to be
`QuantumLattices.OperatorGenerator`s. `SOPT` re-expands the generators term by
term through `expand(gen::OperatorGenerator, bond::Bond)`, which only looks at
`gen.terms` and completely ignores the pre-computed operators stored in
`gen.operators`. Consequently the Wannier90 hopping blocks cannot be injected
through `OperatorGenerator(operators, ...)`; they have to be encoded as
`QuantumLattices.Term`s. This module does exactly that.

The module is part of `TightBindingApproximation`; load it with

```julia
using TightBindingApproximation.Wannier90SOPT
```

# Which case applies?

| Wannier90 input | `spinors` | `num_wann` | Data | Hilbert | Constructor | Resulting `H` |
|:---|:---|:---|:---|:---|:---|:---|
| no spin-orbit coupling | `false` | `nsite * norb` | spinless | `Fock{:f}(norb, 2)` | `W90Bridge(...)` | `H_W90` duplicated onto both spins: every band twice |
| spin-orbit coupling | `true` | `2 * nsite * norb` | spinor | `Fock{:f}(norb, 2)` | `W90SpinorBridge(...)` | genuine spin-mixing `H`, bands not duplicated |

# Case 1: spinless `_hr.dat`, spinful Hilbert

This is the standard use for `SOPT`: the Wannier90 calculation omitted spin-orbit
coupling, so `_hr.dat` describes one spin channel. The Hilbert space is always
spinful (`Fock{:f}(norb, 2)`), and `W90Bridge` does two things:

1. one spin sector (`spin == 1/2`) is used to number the `num_wann` Wannier
   functions and to map them to `(site, orbital)`;
2. every hopping block is encoded with the wildcard spin component `:` in
   `𝕔⁺𝕔(sites, matrix, :)`, so the same block is generated for spin up and spin
   down.

The resulting Hamiltonian is exactly `H = H_Wannier90 ⊗ I₂`. This is **not**
spin-orbit coupling: the up and down blocks are identical, and each Wannier band
is twofold degenerate. It is the right choice when spin is a good quantum number
and the two channels are degenerate.

```julia
wan = W90(path, seedname)
hilbert = Hilbert(site => Fock{:f}(norbital, 2) for site in 1:length(wan.lattice))

bridge = W90Bridge(wan, hilbert)

U  = Hubbard(:U, Uvalue)                         # multi-orbital interactions
U′ = InterOrbitalInterSpin(:U′, Uprime)
H₀, H₁ = w90generators(bridge; interactionterms = (U, U′), neighbors = 3)

sopt = SOPT(wan.lattice, H₁, H₀,
            SecondOrderPerturbation(binaryconfigure, pickstate))
```

Note the argument order of `SOPT`: `(lattice, H₁, H₀, configure)`.

# Case 2: spinor `_hr.dat` with spin-orbit coupling

With `spinors = true` in the Wannier90 input, every spatial orbital produces two
spinor Wannier functions and `num_wann = 2 * nsite * norb`. The `H(R)` blocks
are then `2 * nsite * norb` square matrices whose entries mix spin freely:
`H(R)[i, j]` is the coefficient of `c†_{i,0} c_{j,R}` where `i` and `j` each
carry their own spin. Such a block is not a tensor product of an orbital matrix
and a spin matrix, so it cannot be represented by a single `𝕔⁺𝕔` call. Instead,
the block is split into four spin sectors

```text
O↑↑, O↑↓, O↓↑, O↓↓
```

and each sector is encoded as `𝕔⁺𝕔(sites, Oσσ′, Eσσ′)` with a 2×2 selector
`Eσσ′`; the four are summed into one `MatrixCouplingSum`. This is exactly what
`W90SpinorBridge` does internally.

```julia
hilbert = Hilbert(site => Fock{:f}(norbital, 2) for site in 1:length(wan.lattice))
bridge = W90SpinorBridge(wan, hilbert)           # spinor dataset

H₀, H₁ = w90spinorgenerators(bridge; interactionterms = (U, U′), neighbors = 3)
sopt = SOPT(wan.lattice, H₁, H₀, SecondOrderPerturbation(binaryconfigure, pickstate))
```

The interaction terms are unchanged: `Hubbard`, `InterOrbitalInterSpin`,
`InterOrbitalIntraSpin`, `SpinFlip` and `PairHopping` are spin-diagonal (or
spin-flipping in a controlled way) and work with the same spinful Hilbert.

## Spin ordering in `_hr.dat`

`_hr.dat` contains no spin label at all. After the header (`num_wann`, `nrpts`
and the Wigner-Seitz degeneracies) every line is

```text
R1 R2 R3   j   i   Re H_ji(R)   Im H_ji(R)
```

so the spin information is entirely encoded in the **ordering** of the
`num_wann` Wannier indices. For a spinor calculation Wannier90 3.x creates the
projections in an interleaved order: for each spatial orbital it emits the
spin-up component (`input_proj_s = +1`) and then the spin-down component
(`input_proj_s = -1`). The default order is therefore

```text
orbital 1 ↑, orbital 1 ↓, orbital 2 ↑, orbital 2 ↓, ...
```

`proj2wann_map(i) = i` unless `select_projections` is used, so this order is
preserved unless you ask for a subset or an explicit `(u)`/`(d)` projection
list. `W90SpinorBridge` assumes this default. Two escape hatches are provided:

* `siteindices = Dict(site => [i1, i2, ...])` gives, for every site, its Wannier
  indices in the canonical order `orbital1↑, orbital1↓, orbital2↑, ...`. Use it
  whenever the file order is not the default one.
* `spins = [...]`, a length-`num_wann` vector with entries `1//2`, `-1//2`
  (or `:up`, `:down`), states which spin each Wannier index actually carries.
  The default is the alternating `1//2, -1//2, 1//2, ...`.

As a sanity check, `W90SpinorBridge` compares the centres of consecutive pairs
`(1,2), (3,4), ...` from `_centres.xyz`; the two spin components of one spatial
orbital should be centred at the same place. A mismatch only triggers a warning
(it cannot tell up from down), but it is a strong hint that the ordering is not
the default one.

# Selecting the hoppings of H₁

`neighbors` decides which bonds exist in `H₁`; `shells` decides which of them
carry Wannier90 hopping. A bond whose kind is not in `shells` still appears in
`H₁.bonds` but gets an identically zero coupling. `bondfilter` can exclude
individual bonds on top of that. For example, to keep only first-neighbour
hopping while generating bonds up to the third shell:

```julia
H₀, H₁ = w90generators(bridge; neighbors = 3, shells = [1])
```

# Diagnostics

* `w90shells(bridge; n = 2)` lists every nonzero `H(R)` block together with the
  bond kinds of `bonds(lattice, n)` that can reach it. Use it to choose
  `neighbors` and `shells`.
* `w90coverage(bridge, bonds(lattice, n))` lists the block entries that are not
  reachable from those bonds. `n_missing == 0` means that `H₀ + H₁` reproduces
  the whole Wannier90 Hamiltonian on the selected bonds.

Both accept a `W90Bridge` and a `W90SpinorBridge`.

# Conventions

* A Wannier90 block is stored as `H(R)[i, j] = ⟨w_{i,0}|H|w_{j,R}⟩`, i.e. it is
  the coefficient of `c†_{i,0} c_{j,R}`.
* `QuantumLattices` stores the *Cartesian* displacement of a bond in
  `Point.icoordinate` (see `bonds!` and `tile`), so the integer Wannier90
  vector must be obtained with `decompose(icoordinate(bond), lattice.vectors)`,
  see [`integerR`](@ref).
* For a two-point bond with `p₁ = (site₁, translation t)` and
  `p₂ = (site₂, translation 0)`, one has `icoordinate(bond) = -t`, and the
  forward operator is `c†_{site₁,t} c_{site₂,0}`. After translating the whole
  operator by `-t` it becomes `c†_{i,0} c_{j,-t}`, so its coefficient is
  `H(-t) = H(R)` with `R = integerR(bond)`. Rows are indexed by the Wannier
  orbitals of `p₁` and columns by those of `p₂`. No extra sign is needed.
* The spin basis of a 2×2 spin matrix in `QuantumLattices` is `[+1/2, -1/2]`,
  so `E_upup = [1 0; 0 0]`, `E_updown = [0 1; 0 0]`, `E_downup = [0 0; 1 0]` and
  `E_downdown = [0 0; 0 1]`.

# Notes

* `symmetrize = true` replaces `H(R)` by `(H(R) + H(-R)')/2` and `H(0)` by
  `(H(0) + H(0)')/2`. This is a Hermitian projection: it is idempotent, it is a
  no-op on data that are already Hermitian, and it discards the anti-Hermitian
  numerical noise. Set it to `false` if you deliberately want the raw data; note
  that `Onsite` terms are Hermitian by construction, so a non-Hermitian `H(0)`
  is effectively symmetrized by the operator algebra anyway.
* Energies in `_hr.dat` are usually in eV; give `U`, `J`, `V`, ... in the same
  unit.
* Wannier90 blocks live in one particular Wannier gauge. Do not rotate the
  onsite and hopping blocks independently.
"""
module Wannier90SOPT

using ..Wannier90
import ..Wannier90: gaugetransform
using QuantumLattices

export W90Bridge, W90SpinorBridge, integerR,
       w90onsiteterm, w90hoppingterms, w90generators,
       w90spinoronsiteterm, w90spinorhoppingterms, w90spinorgenerators,
       w90shells, w90coverage

# The concrete `MatrixCoupling` element type produced by
# `𝕔⁺𝕔(sites, matrix, :)`. It does not depend on the matrix size (the basis is
# always `1:size(matrix, i)`), so a 1×1 dummy matrix is enough to extract it.
const W90COUPLING = eltype(typeof(𝕔⁺𝕔((1ˢᵗ, 2ⁿᵈ), zeros(ComplexF64, 1, 1), :)))

@inline _as_tuple(x) = x === nothing ? () : (x isa Tuple ? x : (x,))
@inline _negative(R::NTuple{3,Int}) = map(-, R)

"""
    W90Bridge

Mapping between the Wannier90 Wannier functions and the sites/orbitals of a
`QuantumLattices.Hilbert` space, together with the (optionally symmetrized)
hopping blocks.

# Fields
- `data::W90`: the Wannier90 system the data come from.
- `hilbert::Hilbert`: the Hilbert space used to build the `OperatorGenerator`s.
- `siteindices::Dict{Int, Vector{Int}}`: maps a site to its Wannier indices, in
  orbital order.
- `w90index::Vector{Tuple{Int, Int}}`: maps a Wannier index to `(site, orbital)`.
- `blocks::Dict{NTuple{3, Int}, Matrix{ComplexF64}}`: the (symmetrized) `H(R)`
  blocks.
- `tol::Float64`: tolerance kept for diagnostics.
"""
struct W90Bridge
    data::W90
    hilbert::Hilbert
    siteindices::Dict{Int, Vector{Int}}
    w90index::Vector{Tuple{Int, Int}}
    blocks::Dict{NTuple{3, Int}, Matrix{ComplexF64}}
    tol::Float64
end

"""
    W90Bridge(wan::W90, hilbert::Hilbert;
              symmetrize::Bool=true, siteindices=nothing, tol::Real=1e-8) -> W90Bridge

Construct a bridge between a Wannier90 system and a spinful Hilbert space.

The Wannier90 data are assumed to be spinless: `num_wan == nsites * norbital`.
The Hilbert space must be spinful (`Fock{:f}(norbital, 2)`), and one spin sector
(`spin == 1//2`) is used to number the Wannier functions, so that a spinless
hopping block is later duplicated onto both spin sectors by `𝕔⁺𝕔(..., :)`.
A spinless Hilbert space (`Fock{:f}(norbital, 1)`) is rejected.

The automatic mapping assumes that the Wannier indices are site-major and
orbital-minor, in the same order as
`Table(hilbert, OperatorIndexToTuple(:site, :orbital, :spin))`. If the Wannier
order in the `_hr.dat` file is different, pass `siteindices` explicitly, e.g.
`Dict(1 => [1, 4], 2 => [2, 5], 3 => [3, 6])`. The list for a site gives its
Wannier indices in orbital order and must have the same length as the number of
orbitals of that site in the Hilbert space.

With `symmetrize=true`, `H(R)` is replaced by `(H(R) + H(-R)')/2`, which makes
the generated operators exactly Hermitian and protects against numerical noise.
"""
function W90Bridge(wan::W90, hilbert::Hilbert;
                   symmetrize::Bool = true,
                   siteindices = nothing,
                   tol::Real = 1e-8)
    nwan = size(wan.centers, 2)
    size(wan.centers, 2) == size(first(wan.H).value, 1) || error("W90Bridge error: mismatched Wannier centers and hopping matrices.")
    raw = Dict{NTuple{3, Int}, Matrix{ComplexF64}}(
        R => Matrix(hop.value) for (R, hop) in wan.H.operators.contents
    )

    # 1) Wannier index <-> (site, orbital) mapping
    table = Table(hilbert, OperatorIndexToTuple(:site, :orbital, :spin))
    entries = collect(keys(table))
    any(entry -> entry[3] == 0, entries) &&
        error("W90Bridge error: the Hilbert space must be spinful, i.e. `Fock{:f}(norbital, 2)`; a spinless Hilbert space is not supported.")
    tuples = Tuple{Int, Int}[(s, o) for (s, o, σ) in entries if σ == 1//2]
    length(tuples) == nwan || error("W90Bridge error: $nwan Wannier functions but $(length(tuples)) spin-1/2 Hilbert entries.")

    automatic = build_siteindices(tuples)
    if isnothing(siteindices)
        mapping = automatic
    else
        mapping = Dict{Int, Vector{Int}}(
            site => collect(Int, indices) for (site, indices) in siteindices
        )
        keys(mapping) == keys(automatic) || error("W90Bridge error: `siteindices` must cover exactly the sites $(sort!(collect(keys(automatic)))).")
        for (site, indices) in mapping
            length(indices) == length(automatic[site]) || error("W90Bridge error: site $site has $(length(automatic[site])) orbitals in the Hilbert space but $(length(indices)) Wannier indices were given.")
        end
        perm = sort!(reduce(vcat, values(mapping); init = Int[]))
        perm == collect(1:nwan) || error("W90Bridge error: `siteindices` must contain each Wannier index in 1:$nwan exactly once.")
    end
    w90index = Vector{Tuple{Int, Int}}(undef, nwan)
    for (site, indices) in mapping, (orbital, index) in enumerate(indices)
        w90index[index] = (site, orbital)
    end

    # 2) Symmetrization: readhamiltonian already asserts that -R exists
    blocks = Dict{NTuple{3, Int}, Matrix{ComplexF64}}()
    for (R, M) in raw
        if !symmetrize
            blocks[R] = M
        elseif R == (0, 0, 0)
            blocks[R] = (M + M') / 2
        elseif haskey(raw, _negative(R))
            blocks[R] = (M + raw[_negative(R)]') / 2
        else
            @warn "W90Bridge warning: missing H($(_negative(R))); using H($R) as is."
            blocks[R] = M
        end
    end

    return W90Bridge(wan, hilbert, mapping, w90index, blocks, float(tol))
end

"""
    build_siteindices(tuples) -> Dict{Int, Vector{Int}}

Group Wannier indices by site. The input `tuples` must already be sorted by
`(site, orbital)`; the returned indices are 1-based positions in `tuples`.
"""
function build_siteindices(tuples)
    result = Dict{Int, Vector{Int}}()
    for (index, (site, _)) in enumerate(tuples)
        push!(get!(result, site, Int[]), index)
    end
    return result
end

"""
    integerR(bond::Bond, vectors; atol::Real=1e-6) -> NTuple{3, Int}

Convert the Cartesian displacement of a two-point bond into the integer lattice
translation `R` used by Wannier90.

`Point.icoordinate` stores a Cartesian vector (see `QuantumLattices.bonds!` and
`QuantumLattices.tile`), not an integer lattice coordinate, so the conversion has
to go through `decompose`.

The returned `R` is the index such that `H(R)` is the coefficient block of the
forward operator of the bond, with rows given by the Wannier orbitals of
`bond[1]` and columns by those of `bond[2]`.
"""
function integerR(bond::Bond, vectors; atol::Real = 1e-6)
    length(bond) == 2 || error("integerR error: a two-point bond is required, got a bond of length $(length(bond)).")
    v = icoordinate(bond)
    x = decompose(v, vectors)
    R = ntuple(i -> round(Int, x[i]), Val(3))
    vR = sum(vectors[i] * R[i] for i in eachindex(R))
    isapprox(v, vR; atol = atol, rtol = atol) || error("integerR error: bond icoordinate $v is not an integer lattice translation.")
    return R
end

"""
    _hoppingterm(id::Symbol, bondkind::Int, coupling) -> Term

Build a non-modulatable `Hopping` term whose coupling is the given closure.

The internal constructor is used on purpose: the public `Hopping` constructor
would wrap the `TermCoupling` a second time, which erases its concrete `Coupling`
type and makes `valtype` inference on an abstract `Bond` unreliable.
"""
_hoppingterm(id::Symbol, bondkind::Int, coupling) =
    Term{:Hopping, id}(1.0, bondkind, TermCoupling{W90COUPLING}(coupling),
                       TermAmplitude(nothing), false, false, 1.0)

"""
    _onsiteterm(id::Symbol, coupling) -> Term

Build a non-modulatable Hermitian `Onsite` term (`bondkind == 0`).
"""
_onsiteterm(id::Symbol, coupling) =
    Term{:Onsite, id}(1.0, 0, TermCoupling{W90COUPLING}(coupling),
                      TermAmplitude(nothing), true, false, 1.0)

"""
    w90onsiteterm(bridge::W90Bridge; id::Symbol=:w90_0) -> Term

Onsite term of `H₀`, built from the same-site block `H(0)`.

Both the diagonal onsite energies and the inter-orbital onsite hybridizations
are included. Because the coupling uses the spin component `:`, the spinless
block is duplicated onto both spin sectors of a spinful Hilbert space.
"""
function w90onsiteterm(bridge::W90Bridge; id::Symbol = :w90_0)
    function coupling(bond::Bond)
        site = bond[1].site
        indices = bridge.siteindices[site]
        block = get(bridge.blocks, (0, 0, 0), nothing)
        matrix = isnothing(block) ?
            zeros(ComplexF64, length(indices), length(indices)) :
            Matrix(block[indices, indices])
        return 𝕔⁺𝕔((1ˢᵗ, 1ˢᵗ), matrix, :)
    end
    return _onsiteterm(id, coupling)
end

"""
    w90hoppingterms(bridge::W90Bridge; shells=1:1, bondfilter=_->true,
                    prefix::Symbol=:w90_t) -> Tuple{Vararg{Term}}

Hopping terms of `H₁`, one per bond shell in `shells`.

A bond is included in the perturbation when all of the following hold:

1. its `bond.kind` is in `shells` (this is fixed by the term's `bondkind`);
2. `bondfilter(bond)` returns `true`;
3. the corresponding `H(integerR(bond))` block exists.

Otherwise the term contributes an identically zero coupling on that bond, so
`H₁` itself (and the bonds stored in it) stays intact and only the selected
hoppings enter the perturbation.

The shell of the term is the `bondkind` checked by `expand!`; `shells` therefore
has to be consistent with the bonds of the generator. In practice `shells`
selects the bond shells (first nearest neighbor, second nearest neighbor, ...)
that carry Wannier90 hopping, while the bonds themselves come from `neighbors`.
"""
function w90hoppingterms(bridge::W90Bridge;
                         shells = 1:1,
                         bondfilter = _ -> true,
                         prefix::Symbol = :w90_t)
    vectors = collect(bridge.data.lattice.vectors)

    # Defined first so that every branch returns the same concrete type.
    function zerocoupling(bond::Bond)
        i1 = bridge.siteindices[bond[1].site]
        i2 = bridge.siteindices[bond[2].site]
        return 𝕔⁺𝕔((1ˢᵗ, 2ⁿᵈ), zeros(ComplexF64, length(i1), length(i2)), :)
    end

    function coupling(bond::Bond)
        bondfilter(bond) || return zerocoupling(bond)
        R = integerR(bond, vectors)
        block = get(bridge.blocks, R, nothing)
        isnothing(block) && return zerocoupling(bond)
        i1 = bridge.siteindices[bond[1].site]
        i2 = bridge.siteindices[bond[2].site]
        matrix = Matrix(block[i1, i2])
        return 𝕔⁺𝕔((1ˢᵗ, 2ⁿᵈ), matrix, :)
    end

    shs = shells isa Integer ? (Int(shells),) : unique!(collect(Int, shells))
    return tuple((_hoppingterm(Symbol(prefix, shell), shell, coupling)
                  for shell in shs)...)
end

"""
    w90generators(bridge::W90Bridge; interactionterms=(), perturbationterms=(),
                  neighbors=nothing, shells=nothing, bondfilter=_->true,
                  coordination::Integer=12,
                  boundary=plain) -> Tuple{OperatorGenerator, OperatorGenerator}

Build `(H₀, H₁)` for `SOPT`.

* `H₀` lives on the one-point bonds `bonds(lattice, 0)` and contains the Wannier90
  onsite block `H(0)` plus the user-supplied `interactionterms` (typically
  multi-orbital Hubbard/Kanamori terms). Interaction terms must have
  `bondkind == 0`; terms with a larger `bondkind` never contribute on one-point
  bonds.
* `H₁` lives on `bonds(lattice, n)` and contains one Wannier90 hopping term per
  shell plus the optional `perturbationterms`. `n` is `neighbors` when it is an
  integer, or the order of a `Neighbors` object.

`neighbors` and `shells` split the bond selection in two:

* `neighbors = n` decides **which bonds exist in `H₁`** (all bonds up to the
  `n`-th nearest neighbor).
* `shells = [k₁, k₂, ...]` decides **which of those bonds carry Wannier90
  hopping**. A bond whose `bond.kind` is not in `shells` still appears in
  `H₁.bonds` but gets an identically zero coupling. When `shells` is omitted it
  defaults to `1:n`, i.e. every generated shell is used.

For example, to keep only the first nearest-neighbor hoppings while still
generating bonds up to the third shell, use
`neighbors = 3, shells = [1]`. A scalar shell is accepted as well
(`shells = 1`).

`bondfilter(bond)` can additionally exclude individual bonds; it is applied on
top of the shell selection.

`interactionterms` and `perturbationterms` may be `nothing`, a single term, or a
tuple of terms.

The `OperatorGenerator`s are created with `half=false`, so each hopping term
automatically contributes its Hermitian conjugate as well.
"""
function w90generators(bridge::W90Bridge;
                       interactionterms = (),
                       perturbationterms = (),
                       neighbors = nothing,
                       shells = nothing,
                       bondfilter = _ -> true,
                       coordination::Integer = 12,
                       boundary = plain)
    lattice = bridge.data.lattice
    n = if isnothing(neighbors)
        isnothing(shells) ? 1 : (shells isa Integer ? Int(shells) : maximum(shells))
    else
        neighbors
    end
    nmax = n isa Integer ? n : nneighbor(n)
    sh = isnothing(shells) ? (1:nmax) : shells
    shs = sh isa Integer ? (Int(sh),) : unique!(collect(Int, sh))

    h1bonds = n isa Integer ? bonds(lattice, n; coordination = coordination) :
                              bonds(lattice, n)

    available = Set(Int[b.kind for b in h1bonds if length(b) == 2])
    extra = setdiff(shs, sort!(collect(available)))
    isempty(extra) || error("w90generators error: shell(s) $(extra) are not present in the generated bonds; available shells are $(sort!(collect(available))).")

    h0terms = (w90onsiteterm(bridge), _as_tuple(interactionterms)...)
    h1terms = (w90hoppingterms(bridge; shells = shs, bondfilter)...,
               _as_tuple(perturbationterms)...)
    isempty(h1terms) && error("w90generators error: `H₁` needs at least one term; check `neighbors`/`shells`/`perturbationterms`.")

    H0 = OperatorGenerator(bonds(lattice, 0), bridge.hilbert, h0terms,
                           boundary; half = false)
    H1 = OperatorGenerator(h1bonds, bridge.hilbert, h1terms,
                           boundary; half = false)
    return H0, H1
end

# ============================================================================
# Spinor (spin-orbit coupling) support
# ============================================================================

const SPINUP = 1//2
const SPINDOWN = -1//2

# Selector matrices in the 2×2 spin space, whose basis order is [+1/2, -1/2].
const E_upup = ComplexF64[1 0; 0 0]
const E_updown = ComplexF64[0 1; 0 0]
const E_downup = ComplexF64[0 0; 1 0]
const E_downdown = ComplexF64[0 0; 0 1]

# Concrete element type of the four-sector sum. All four `MatrixCoupling`s have
# the same type (only their matrix entries differ), so the `MatrixCouplingSum`
# has a concrete `Coupling` eltype.
const W90SPINCOUPLING = eltype(MatrixCouplingSum(
    𝕔⁺𝕔((1ˢᵗ, 2ⁿᵈ), zeros(ComplexF64, 1, 1), E_upup),
    𝕔⁺𝕔((1ˢᵗ, 2ⁿᵈ), zeros(ComplexF64, 1, 1), E_updown),
    𝕔⁺𝕔((1ˢᵗ, 2ⁿᵈ), zeros(ComplexF64, 1, 1), E_downup),
    𝕔⁺𝕔((1ˢᵗ, 2ⁿᵈ), zeros(ComplexF64, 1, 1), E_downdown),
))

_spinorhoppingterm(id::Symbol, bondkind::Int, coupling) =
    Term{:Hopping, id}(1.0, bondkind, TermCoupling{W90SPINCOUPLING}(coupling),
                       TermAmplitude(nothing), false, false, 1.0)

_spinoronsiteterm(id::Symbol, coupling) =
    Term{:Onsite, id}(1.0, 0, TermCoupling{W90SPINCOUPLING}(coupling),
                      TermAmplitude(nothing), true, false, 1.0)

@inline function _spinentry(σ₁::Rational{Int}, σ₂::Rational{Int})
    σ₁ == SPINUP && return σ₂ == SPINUP ? E_upup : E_updown
    return σ₂ == SPINUP ? E_downup : E_downdown
end

@inline function _spinvalue(x)
    x === :up && return SPINUP
    x === :down && return SPINDOWN
    if x isa Rational
        (x == SPINUP || x == SPINDOWN) && return Rational{Int}(x)
    elseif x isa Integer
        x == 1 && return SPINUP
        x == -1 && return SPINDOWN
    end
    error("W90SpinorBridge error: invalid spin entry $x; use 1//2, -1//2, :up or :down.")
end

"""
    W90SpinorBridge

Mapping between spinor Wannier functions of a Wannier90 calculation with spin-orbit
coupling (`spinors = true`) and the sites/orbitals/spins of a spinful
`QuantumLattices.Hilbert` space, together with the (optionally symmetrized)
hopping blocks.

# Fields
- `data::W90`: the Wannier90 system the data come from.
- `hilbert::Hilbert`: the Hilbert space used to build the `OperatorGenerator`s.
- `siteindices::Dict{Int, Vector{Int}}`: maps a site to its Wannier indices in the
  canonical order `orbital1↑, orbital1↓, orbital2↑, orbital2↓, ...`.
- `w90index::Vector{Tuple{Int, Int, Rational{Int}}}`: maps a Wannier index to
  `(site, orbital, spin)`.
- `blocks::Dict{NTuple{3, Int}, Matrix{ComplexF64}}`: the (symmetrized) `H(R)` blocks.
- `tol::Float64`: tolerance kept for diagnostics.
"""
struct W90SpinorBridge
    data::W90
    hilbert::Hilbert
    siteindices::Dict{Int, Vector{Int}}
    w90index::Vector{Tuple{Int, Int, Rational{Int}}}
    blocks::Dict{NTuple{3, Int}, Matrix{ComplexF64}}
    tol::Float64
end

"""
    W90SpinorBridge(wan::W90, hilbert::Hilbert; spins=nothing, siteindices=nothing,
                    symmetrize::Bool=true, tol::Real=1e-8, checkcenters::Bool=true) -> W90SpinorBridge

Construct a bridge between a spinor Wannier90 system and a spinful Hilbert space.

The Wannier90 data must be spinor: `num_wann == 2 * nsite * norbital`, and the
Hilbert space must be spinful (`Fock{:f}(norbital, 2)`). Every `H(R)` block is a
`num_wann × num_wann` matrix whose entries connect Wannier functions of
arbitrary spins.

By default the Wannier index order is assumed to be the one Wannier90 creates,
i.e. `orbital1↑, orbital1↓, orbital2↑, orbital2↓, ...` (one spin-up and one
spin-down component per spatial orbital). Pass `siteindices` to give the order
explicitly, e.g. `Dict(1 => [1, 2, 3, 4], 2 => [5, 6, 7, 8])`, and `spins` to
state which spin each Wannier index carries (`1//2`, `-1//2`, `:up` or `:down`).

With `symmetrize=true`, `H(R)` is replaced by `(H(R) + H(-R)')/2`, which makes
the generated operators exactly Hermitian and protects against numerical noise.
With `checkcenters=true` a warning is issued if the two members of a spinor pair
do not share the same Wannier centre, which usually means the ordering is not
the default one.
"""
function W90SpinorBridge(wan::W90, hilbert::Hilbert;
                         spins = nothing,
                         siteindices = nothing,
                         symmetrize::Bool = true,
                         tol::Real = 1e-8,
                         checkcenters::Bool = true)
    nwan = size(wan.centers, 2)
    size(wan.centers, 2) == size(first(wan.H).value, 1) || error("W90SpinorBridge error: mismatched Wannier centers and hopping matrices.")
    iseven(nwan) || error("W90SpinorBridge error: num_wann = $nwan is odd; a spinor dataset must have num_wann = 2 * n_spatial.")
    raw = Dict{NTuple{3, Int}, Matrix{ComplexF64}}(
        R => Matrix(hop.value) for (R, hop) in wan.H.operators.contents
    )

    table = Table(hilbert, OperatorIndexToTuple(:site, :orbital, :spin))
    entries = collect(keys(table))
    all(entry -> entry[3] != 0//1, entries) || error("W90SpinorBridge error: the Hilbert space must be spinful, i.e. `Fock{:f}(norbital, 2)`.")
    length(entries) == nwan || error("W90SpinorBridge error: $nwan Wannier functions but $(length(entries)) spinful Hilbert entries.")

    # (site, orbital) skeleton of the Wannier functions, in Hilbert order
    spatial = Tuple{Int, Int}[(site, orbital) for (site, orbital, spin) in entries if spin == SPINUP]
    2 * length(spatial) == nwan || error("W90SpinorBridge error: the Hilbert space has $(length(spatial)) spatial orbitals but num_wann = $nwan.")
    spatial_siteindices = build_siteindices(spatial)

    # Default mapping: Wannier indices (2k-1, 2k) belong to the k-th spatial orbital,
    # with spin up first, as Wannier90 creates spinor projections.
    default_mapping = Dict{Int, Vector{Int}}()
    for (site, indices) in spatial_siteindices
        w90indices = Int[]
        for k in indices
            push!(w90indices, 2k - 1)
            push!(w90indices, 2k)
        end
        default_mapping[site] = w90indices
    end

    if isnothing(siteindices)
        mapping = default_mapping
    else
        mapping = Dict{Int, Vector{Int}}(
            site => collect(Int, indices) for (site, indices) in siteindices
        )
        Set(keys(mapping)) == Set(keys(default_mapping)) || error("W90SpinorBridge error: `siteindices` must cover exactly the sites $(sort!(collect(keys(default_mapping)))).")
        for (site, indices) in mapping
            length(indices) == length(default_mapping[site]) || error("W90SpinorBridge error: site $site needs $(length(default_mapping[site])) Wannier indices but $(length(indices)) were given.")
            iseven(length(indices)) || error("W90SpinorBridge error: site $site needs an even number of Wannier indices (two per spatial orbital).")
        end
        perm = sort!(reduce(vcat, values(mapping); init = Int[]))
        perm == collect(1:nwan) || error("W90SpinorBridge error: `siteindices` must contain each Wannier index in 1:$nwan exactly once.")
    end

    if isnothing(spins)
        spinvalues = Rational{Int}[isodd(w) ? SPINUP : SPINDOWN for w in 1:nwan]
    else
        length(spins) == nwan || error("W90SpinorBridge error: `spins` must have length num_wann = $nwan.")
        spinvalues = Rational{Int}[_spinvalue(σ) for σ in spins]
    end

    w90index = Vector{Tuple{Int, Int, Rational{Int}}}(undef, nwan)
    for (site, indices) in mapping, (position, w) in enumerate(indices)
        w90index[w] = (site, (position + 1) ÷ 2, spinvalues[w])
    end

    # Every site must use the same spin in a given slot for all its orbitals;
    # otherwise the (orbital, spin) block structure below is not well defined.
    for (site, indices) in mapping, slot in 1:2
        σ = w90index[indices[slot]][3]
        for k in 1:(length(indices) ÷ 2 - 1)
            w90index[indices[2k + slot]][3] == σ ||
                error("W90SpinorBridge error: site $site does not use the same spin in slot $slot for every orbital; specify `siteindices` explicitly.")
        end
    end

    if checkcenters
        mismatched = Int[]
        for (site, indices) in mapping, k in 1:(length(indices) ÷ 2)
            i, j = indices[2k - 1], indices[2k]
            scale = 1 + maximum(abs, wan.centers[:, i])
            maximum(abs, wan.centers[:, i] - wan.centers[:, j]) > 1e-3 * scale &&
                push!(mismatched, k)
        end
        isempty(mismatched) ||
            @warn "W90SpinorBridge warning: spinor pairs $(mismatched) do not share Wannier centres; the default up/down ordering may be wrong. Pass `siteindices` and/or `spins` explicitly."
    end

    # Symmetrization: readhamiltonian already asserts that -R exists
    blocks = Dict{NTuple{3, Int}, Matrix{ComplexF64}}()
    for (R, M) in raw
        if !symmetrize
            blocks[R] = M
        elseif R == (0, 0, 0)
            blocks[R] = (M + M') / 2
        elseif haskey(raw, _negative(R))
            blocks[R] = (M + raw[_negative(R)]') / 2
        else
            @warn "W90SpinorBridge warning: missing H($(_negative(R))); using H($R) as is."
            blocks[R] = M
        end
    end

    return W90SpinorBridge(wan, hilbert, mapping, w90index, blocks, float(tol))
end

"""
    _spincouplings(bridge::W90SpinorBridge, M, idx1, idx2, sites) -> MatrixCouplingSum

Split the block `M` (in canonical `(orbital, spin-slot)` order) into the four
spin sectors and encode them as `𝕔⁺𝕔(sites, Oσσ′, Eσσ′)`, one per sector.
"""
function _spincouplings(bridge::W90SpinorBridge, M::Matrix{ComplexF64},
                        idx1::Vector{Int}, idx2::Vector{Int},
                        sites::Tuple{Ordinal, Ordinal})
    size(M) == (length(idx1), length(idx2)) || error("_spincouplings error: block size $(size(M)) does not match the Wannier indices.")
    σ₁₁, σ₁₂ = bridge.w90index[idx1[1]][3], bridge.w90index[idx1[2]][3]
    σ₂₁, σ₂₂ = bridge.w90index[idx2[1]][3], bridge.w90index[idx2[2]][3]
    return MatrixCouplingSum(
        𝕔⁺𝕔(sites, Matrix{ComplexF64}(M[1:2:end, 1:2:end]), _spinentry(σ₁₁, σ₂₁)),
        𝕔⁺𝕔(sites, Matrix{ComplexF64}(M[1:2:end, 2:2:end]), _spinentry(σ₁₁, σ₂₂)),
        𝕔⁺𝕔(sites, Matrix{ComplexF64}(M[2:2:end, 1:2:end]), _spinentry(σ₁₂, σ₂₁)),
        𝕔⁺𝕔(sites, Matrix{ComplexF64}(M[2:2:end, 2:2:end]), _spinentry(σ₁₂, σ₂₂)),
    )
end

"""
    w90spinoronsiteterm(bridge::W90SpinorBridge; id::Symbol=:w90_0) -> Term

Onsite term of `H₀` for a spinor dataset, built from the same-site block `H(0)`.

All four spin sectors are included, so the diagonal onsite energies, the
inter-orbital hybridizations and the onsite spin mixing are preserved.
"""
function w90spinoronsiteterm(bridge::W90SpinorBridge; id::Symbol = :w90_0)
    function coupling(bond::Bond)
        indices = bridge.siteindices[bond[1].site]
        block = get(bridge.blocks, (0, 0, 0), nothing)
        n = length(indices)
        matrix = isnothing(block) ? zeros(ComplexF64, n, n) :
                                    Matrix(block[indices, indices])
        return _spincouplings(bridge, matrix, indices, indices, (1ˢᵗ, 1ˢᵗ))
    end
    return _spinoronsiteterm(id, coupling)
end

"""
    w90spinorhoppingterms(bridge::W90SpinorBridge; shells=1:1, bondfilter=_->true,
                          prefix::Symbol=:w90_t) -> Tuple{Vararg{Term}}

Hopping terms of `H₁` for a spinor dataset, one per bond shell in `shells`.

The selection rules are the same as for [`w90hoppingterms`](@ref): a bond is
included when its `bond.kind` is in `shells`, `bondfilter(bond)` returns `true`
and the corresponding `H(integerR(bond))` block exists. Otherwise the term
contributes an identically zero coupling.
"""
function w90spinorhoppingterms(bridge::W90SpinorBridge;
                               shells = 1:1,
                               bondfilter = _ -> true,
                               prefix::Symbol = :w90_t)
    vectors = collect(bridge.data.lattice.vectors)

    function zerocoupling(bond::Bond)
        i1 = bridge.siteindices[bond[1].site]
        i2 = bridge.siteindices[bond[2].site]
        return _spincouplings(bridge, zeros(ComplexF64, length(i1), length(i2)),
                              i1, i2, (1ˢᵗ, 2ⁿᵈ))
    end

    function coupling(bond::Bond)
        bondfilter(bond) || return zerocoupling(bond)
        R = integerR(bond, vectors)
        block = get(bridge.blocks, R, nothing)
        isnothing(block) && return zerocoupling(bond)
        i1 = bridge.siteindices[bond[1].site]
        i2 = bridge.siteindices[bond[2].site]
        matrix = Matrix(block[i1, i2])
        return _spincouplings(bridge, matrix, i1, i2, (1ˢᵗ, 2ⁿᵈ))
    end

    shs = shells isa Integer ? (Int(shells),) : unique!(collect(Int, shells))
    return tuple((_spinorhoppingterm(Symbol(prefix, shell), shell, coupling)
                  for shell in shs)...)
end

"""
    w90spinorgenerators(bridge::W90SpinorBridge; interactionterms=(), perturbationterms=(),
                        neighbors=nothing, shells=nothing, bondfilter=_->true,
                        coordination::Integer=12,
                        boundary=plain) -> Tuple{OperatorGenerator, OperatorGenerator}

Build `(H₀, H₁)` for `SOPT` from a spinor Wannier90 dataset.

The semantics of `interactionterms`, `perturbationterms`, `neighbors`, `shells`,
`bondfilter`, `coordination` and `boundary` are identical to those of
[`w90generators`](@ref); only the Wannier90 terms are replaced by their spinor
counterparts, which include the spin-mixing sectors.
"""
function w90spinorgenerators(bridge::W90SpinorBridge;
                             interactionterms = (),
                             perturbationterms = (),
                             neighbors = nothing,
                             shells = nothing,
                             bondfilter = _ -> true,
                             coordination::Integer = 12,
                             boundary = plain)
    lattice = bridge.data.lattice
    n = if isnothing(neighbors)
        isnothing(shells) ? 1 : (shells isa Integer ? Int(shells) : maximum(shells))
    else
        neighbors
    end
    nmax = n isa Integer ? n : nneighbor(n)
    sh = isnothing(shells) ? (1:nmax) : shells
    shs = sh isa Integer ? (Int(sh),) : unique!(collect(Int, sh))

    h1bonds = n isa Integer ? bonds(lattice, n; coordination = coordination) :
                              bonds(lattice, n)

    available = Set(Int[b.kind for b in h1bonds if length(b) == 2])
    extra = setdiff(shs, sort!(collect(available)))
    isempty(extra) ||
        error("w90spinorgenerators error: shell(s) $(extra) are not present in the generated bonds; available shells are $(sort!(collect(available))).")

    h0terms = (w90spinoronsiteterm(bridge), _as_tuple(interactionterms)...)
    h1terms = (w90spinorhoppingterms(bridge; shells = shs, bondfilter)...,
               _as_tuple(perturbationterms)...)
    isempty(h1terms) &&
        error("w90spinorgenerators error: `H₁` needs at least one term; check `neighbors`/`shells`/`perturbationterms`.")

    H0 = OperatorGenerator(bonds(lattice, 0), bridge.hilbert, h0terms,
                           boundary; half = false)
    H1 = OperatorGenerator(h1bonds, bridge.hilbert, h1terms,
                           boundary; half = false)
    return H0, H1
end

"""
    w90shells(bridge::W90Bridge; n::Integer=1) -> Vector{Tuple{NTuple{3,Int}, Vector{Int}}}

List the nonzero `H(R)` blocks together with the bond kinds of
`bonds(lattice, n)` that correspond to them.

This is the diagnostic to choose `neighbors` and `shells`: a nonempty kind list
means the block can be reached by a term of that shell; an empty list means the
block lies outside `bonds(lattice, n)`.
"""
function w90shells(bridge::W90Bridge; n::Integer = 1)
    lattice = bridge.data.lattice
    vectors = collect(lattice.vectors)
    bs = bonds(lattice, n)
    result = Tuple{NTuple{3, Int}, Vector{Int}}[]
    for (R, M) in bridge.blocks
        iszero(maximum(abs, M)) && continue
        # A block H(R) is reached either as the forward block of a bond with
        # integerR == R, or as the adjoint block of a bond with integerR == -R
        # (Hopping automatically adds the Hermitian conjugate).
        negative = _negative(R)
        kinds = sort!(unique!(Int[b.kind for b in bs
                                  if length(b) == 2 &&
                                     (integerR(b, vectors) == R ||
                                      integerR(b, vectors) == negative)]))
        push!(result, (R, kinds))
    end
    return sort!(result; by = first)
end

"""
    w90coverage(bridge::W90Bridge, bs; onsite::Bool=true, tol::Real=1e-10)

Check whether every nonzero Wannier90 block is reachable from the bonds `bs`.

Returns `(; covered, missing, n_missing)`, where `missing` lists the entries
`(; R, i, j, si, sj, value, reason)` that are not covered. Same-site entries of
`H(0)` are considered covered by the onsite term when `onsite=true`.

`n_missing == 0` means that `H₀ + H₁` built from `bs` reproduces the whole
Wannier90 Hamiltonian on those bonds.
"""
function w90coverage(bridge::W90Bridge, bs; onsite::Bool = true, tol::Real = 1e-10)
    vectors = collect(bridge.data.lattice.vectors)
    covered = Set{Tuple{Int, Int, NTuple{3, Int}}}()
    for bond in bs
        length(bond) == 2 || continue
        R = integerR(bond, vectors)
        push!(covered, (bond[1].site, bond[2].site, R))
        push!(covered, (bond[2].site, bond[1].site, _negative(R)))
    end

    missing = NamedTuple[]
    for (R, M) in bridge.blocks
        for j in axes(M, 2), i in axes(M, 1)
            abs(M[i, j]) <= tol && continue
            si, sj = bridge.w90index[i][1], bridge.w90index[j][1]
            if si == sj && R == (0, 0, 0)
                onsite || push!(missing, (; R, i, j, si, sj, value = M[i, j], reason = :onsite))
            elseif (si, sj, R) ∉ covered
                push!(missing, (; R, i, j, si, sj, value = M[i, j], reason = :hopping))
            end
        end
    end
    return (; covered, missing, n_missing = length(missing))
end

"""
    w90shells(bridge::W90SpinorBridge; n::Integer=1)

Spinor counterpart of [`w90shells`](@ref): the entries of `H(R)` are counted
individually, so spin-mixing entries appear next to spin-conserving ones.
"""
function w90shells(bridge::W90SpinorBridge; n::Integer = 1)
    lattice = bridge.data.lattice
    vectors = collect(lattice.vectors)
    bs = bonds(lattice, n)
    result = Tuple{NTuple{3, Int}, Vector{Int}}[]
    for (R, M) in bridge.blocks
        iszero(maximum(abs, M)) && continue
        negative = _negative(R)
        kinds = sort!(unique!(Int[b.kind for b in bs
                                  if length(b) == 2 &&
                                     (integerR(b, vectors) == R ||
                                      integerR(b, vectors) == negative)]))
        push!(result, (R, kinds))
    end
    return sort!(result; by = first)
end

"""
    w90coverage(bridge::W90SpinorBridge, bs; onsite::Bool=true, tol::Real=1e-10)

Spinor counterpart of [`w90coverage`](@ref). The entries are reported per
`(R, i, j)` Wannier matrix element, so a spin-mixing entry that is not reachable
from `bs` is listed with its own Wannier index pair.
"""
function w90coverage(bridge::W90SpinorBridge, bs; onsite::Bool = true, tol::Real = 1e-10)
    vectors = collect(bridge.data.lattice.vectors)
    covered = Set{Tuple{Int, Int, NTuple{3, Int}}}()
    for bond in bs
        length(bond) == 2 || continue
        R = integerR(bond, vectors)
        push!(covered, (bond[1].site, bond[2].site, R))
        push!(covered, (bond[2].site, bond[1].site, _negative(R)))
    end

    missing = NamedTuple[]
    for (R, M) in bridge.blocks
        for j in axes(M, 2), i in axes(M, 1)
            abs(M[i, j]) <= tol && continue
            si, sj = bridge.w90index[i][1], bridge.w90index[j][1]
            if si == sj && R == (0, 0, 0)
                onsite || push!(missing, (; R, i, j, si, sj, value = M[i, j], reason = :onsite))
            elseif (si, sj, R) ∉ covered
                push!(missing, (; R, i, j, si, sj, value = M[i, j], reason = :hopping))
            end
        end
    end
    return (; covered, missing, n_missing = length(missing))
end

# ---------------------------------------------------------------------------
# Wannier-gauge transformation (extension of gaugetransform)
# ---------------------------------------------------------------------------

"""
    gaugetransform(bridge::Union{W90Bridge, W90SpinorBridge}, rotations; kwargs...) -> W90

Rotate the Wannier gauge of `bridge.data` with a site-local unitary and return
the new `W90`.

`rotations` is either a dictionary `site => local unitary` or a function
`site -> local unitary`; the per-site Wannier index order is `bridge.siteindices`,
so the `j`-th row and column of `rotations[site]` refer to
`bridge.siteindices[site][j]`. Every `H(R)` block is transformed in one call, so
the data stay in a single Wannier gauge.

The bridge itself is **not** updated: `bridge.blocks` is a snapshot taken when the
bridge was constructed. Rebuild the bridge from the returned `W90` before calling
`w90generators` or `w90spinorgenerators`:

    wan′    = gaugetransform(bridge, rotations)
    bridge′ = W90Bridge(wan′, bridge.hilbert)          # or W90SpinorBridge
    H₀, H₁  = w90generators(bridge′; ...)

For a `W90SpinorBridge`, only rotations that preserve the spin slot of every
Wannier index are compatible with the spin labels stored in `bridge.w90index`
and with the slot decomposition used by the operator generator; in the default
`orbital1↑, orbital1↓, orbital2↑, ...` order that means `kron(R, I₂)`. A rotation
that mixes spin-up and spin-down still yields the correct matrix relation
`matrix(wan′, k) = U' * matrix(wan, k) * U`, but the Wannier functions are no
longer spin eigenstates, so the spin labels and the spin-resolved coupling
decomposition no longer carry their original meaning.

See `TightBindingApproximation.Wannier90.gaugetransform` for the conventions,
the unitarity check, the treatment of the Wannier centres and the interaction
terms.
"""
function gaugetransform(
        bridge::Union{W90Bridge, W90SpinorBridge},
        rotations::Union{AbstractDict, Function};
        kwargs...)
    return gaugetransform(
        bridge.data, bridge.siteindices, rotations; kwargs...)
end

function gaugetransform(
        bridge::Union{W90Bridge, W90SpinorBridge},
        U::AbstractMatrix{<:Number};
        kwargs...)
    return gaugetransform(bridge.data, U; kwargs...)
end

end # module
