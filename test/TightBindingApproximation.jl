using Artifacts
using LinearAlgebra: Diagonal, Eigen, Hermitian, I, eigvals, exp, ishermitian, kron
using Pkg
using QuantumLattices: Algorithm, BrillouinZone, Elastic, Fock, Formula, Generator, Hilbert, Hooke, Hopping, Kinetic, Lattice, Metric, Onsite, OperatorGenerator, OperatorSum, OperatorIndexToTuple, Pairing, Parameters, Phonon, ReciprocalPath, ReciprocalZone, StaticGenerator, Table
using QuantumLattices: atol, σᶻ, 𝕕, 𝕕𝕕, 𝕕⁺𝕕, 𝕡, 𝕦, azimuth, bonds, decompose, dimension, distance, expand, getcontent, id, icoordinate, kind, matrix, parameternames, reciprocals, rcoordinate, scalartype, update!, value, @rectangle_str
using StaticArrays: SVector
using TightBindingApproximation
using TightBindingApproximation.Fitting
using TightBindingApproximation.Wannier90
using TightBindingApproximation.Wannier90SOPT
import CairoMakie as Makie
import Plots

@time @testset "prerequisite" begin
    @test promote_type(Fermionic{:TBA}, Fermionic{:TBA}) == Fermionic{:TBA}
    @test promote_type(Fermionic{:BdG}, Fermionic{:BdG}) == Fermionic{:BdG}
    @test promote_type(Fermionic{:TBA}, Fermionic{:BdG}) == promote_type(Fermionic{:BdG}, Fermionic{:TBA}) == Fermionic{:BdG}

    @test promote_type(Bosonic{:TBA}, Bosonic{:TBA}) == Bosonic{:TBA}
    @test promote_type(Bosonic{:BdG}, Bosonic{:BdG}) == Bosonic{:BdG}
    @test promote_type(Bosonic{:TBA}, Bosonic{:BdG}) == promote_type(Bosonic{:BdG}, Bosonic{:TBA}) == Bosonic{:BdG}

    @test TBAKind(Hopping, Fock{:f}) == TBAKind(Onsite, Fock{:f}) == TBAKind(typeof(()), Fock{:f}) == Fermionic(:TBA)
    @test TBAKind(Pairing, Fock{:f}) == Fermionic(:BdG)
    @test TBAKind(Hopping, Fock{:b}) == TBAKind(Onsite, Fock{:b}) == TBAKind(typeof(()), Fock{:b}) == Bosonic(:TBA)
    @test TBAKind(Pairing, Fock{:b}) == Bosonic(:BdG)
    @test TBAKind(Kinetic, Phonon) == TBAKind(Hooke, Phonon) == TBAKind(Elastic, Phonon) == TBAKind(typeof(()), Phonon) == Phononic()
    @test TBAKind(Tuple{Hopping, Onsite}, Fock{:f}) == Fermionic(:TBA)
    @test TBAKind(Tuple{Hopping, Onsite, Pairing}, Fock{:f}) == Fermionic(:BdG)
    @test TBAKind(Tuple{Hopping, Onsite}, Fock{:b}) == Bosonic(:TBA)
    @test TBAKind(Tuple{Hopping, Onsite, Pairing}, Fock{:b}) == Bosonic(:BdG)

    @test infinitesimal(Fermionic(:TBA)) == infinitesimal(Fermionic(:BdG)) == infinitesimal(Bosonic(:TBA)) == 0
    @test infinitesimal(Bosonic(:BdG)) == infinitesimal(Phononic()) == atol/5

    @test Metric(Fermionic(:TBA), Hilbert(1=>Fock{:f}(1, 1))) == OperatorIndexToTuple(:site, :orbital, :spin)
    @test Metric(Fermionic(:BdG), Hilbert(1=>Fock{:f}(1, 1))) == OperatorIndexToTuple(:nambu, :site, :orbital, :spin)
    @test Metric(Bosonic(:TBA), Hilbert(1=>Fock{:b}(1, 1))) == OperatorIndexToTuple(:site, :orbital, :spin)
    @test Metric(Bosonic(:BdG), Hilbert(1=>Fock{:b}(1, 1))) == OperatorIndexToTuple(:nambu, :site, :orbital, :spin)
    @test Metric(Phononic(), Hilbert(1=>Phonon(2))) == OperatorIndexToTuple(kind, :site, :direction)

    by = OperatorIndexToTuple(kind, :site, :direction)
    @test Table(Hilbert(Phonon(2), 2), by) == Table([𝕦(1, 'x'), 𝕦(2, 'x'), 𝕦(1, 'y'), 𝕦(2, 'y'), 𝕡(1, 'x'), 𝕡(2, 'x'), 𝕡(1, 'y'), 𝕡(2, 'y')], by)

    @test isnothing(commutator(Fermionic(:TBA), Hilbert(1=>Fock{:f}(1, 2))))
    @test isnothing(commutator(Fermionic(:BdG), Hilbert(1=>Fock{:f}(1, 2))))
    @test isnothing(commutator(Bosonic(:TBA), Hilbert(1=>Fock{:b}(1, 2))))
    @test commutator(Bosonic(:BdG), Hilbert(1=>Fock{:b}(1, 2))) == Diagonal([1, 1, -1, -1])
    @test commutator(Phononic(), Hilbert(1=>Phonon(2))) == Hermitian([0 0 -im 0; 0 0 0 -im; im 0 0 0; 0 im 0 0])
end

@time @testset "Quadratic" begin
    m = Quadratic(1.0, (1, 2), [0.0], [0.0])
    @test parameternames(typeof(m)) == (:value, :coordinate)
    @test getcontent(m, :id) == ((1, 2), [0.0], [0.0])
    @test m == Quadratic(1.0, ((1, 2), [0.0], [0.0]))
    @test string(m) == "Quadratic(1.0, (1, 2), [0.0], [0.0])"
end

@time @testset "Quadraticization{<:TBAKind{:TBA}}" begin
    t = Hopping(:t, -1.0, 1)
    μ = Onsite(:μ, 2.0)
    lattice = Lattice([0.0], [0.5]; vectors=[[1.0]])
    hilbert = Hilbert(Fock{:f}(1, 1), length(lattice))
    gen = Generator(bonds(lattice, 1), hilbert, (t, μ); half=false)
    tbakind = TBAKind(typeof(gen.terms), valtype(gen.hilbert))
    table = Table(gen.hilbert, Metric(tbakind, gen.hilbert))
    quadraticization = Quadraticization{typeof(tbakind)}(table)
    @test valtype(quadraticization, expand(gen)) == OperatorSum{Quadratic{Float64, SVector{1, Float64}}, Tuple{Tuple{Int, Int}, SVector{1, Float64}, SVector{1, Float64}}}
    @test quadraticization(first(expand(gen))) == OperatorSum(Quadratic(-1.0, (2, 1), [-0.5], [0.0]))
    ops = quadraticization(expand(gen))
    @test ops == OperatorSum(
        Quadratic(-1.0, (2, 1), [-0.5], [0.0]), Quadratic(-1.0, (1, 2), [0.5], [0.0]), Quadratic(-1.0, (2, 1), [0.5], [1.0]),
        Quadratic(-1.0, (1, 2), [-0.5], [-1.0]), Quadratic(2.0, (1, 1), [0.0], [0.0]), Quadratic(2.0, (2, 2), [0.0], [0.0])
    )
end

@time @testset "Quadraticization{Fermionic{:BdG}}" begin
    t = Hopping(:t, -1.0, 1)
    μ = Onsite(:μ, 2.0)
    Δ = Pairing(:Δ, 0.1, 1, 𝕕𝕕(:, :, :); amplitude=bond->rcoordinate(bond)[1]>0 ? 1 : -1)
    lattice = Lattice([0.0], [0.5]; vectors=[[1.0]])
    hilbert = Hilbert(Fock{:f}(1, 1), length(lattice))
    gen = Generator(bonds(lattice, 1), hilbert, (t, μ, Δ); half=false)
    tbakind = TBAKind(typeof(gen.terms), valtype(gen.hilbert))
    table = Table(gen.hilbert, Metric(tbakind, gen.hilbert))
    quadraticization = Quadraticization{typeof(tbakind)}(table)
    @test quadraticization(expand(gen)) == OperatorSum(
        Quadratic(-1.0, (2, 1), [-0.5], [0.0]), Quadratic(1.0, (3, 4), [0.5], [-0.0]), Quadratic(-1.0, (1, 2), [0.5], [0.0]), Quadratic(1.0, (4, 3), [-0.5], [-0.0]),
        Quadratic(-1.0, (2, 1), [0.5], [1.0]), Quadratic(1.0, (3, 4), [-0.5], [-1.0]), Quadratic(-1.0, (1, 2), [-0.5], [-1.0]), Quadratic(1.0, (4, 3), [0.5], [1.0]),
        Quadratic(2.0, (1, 1), [0.0], [0.0]), Quadratic(-2.0, (3, 3), [-0.0], [-0.0]), Quadratic(2.0, (2, 2), [0.0], [0.0]), Quadratic(-2.0, (4, 4), [-0.0], [-0.0]),
        Quadratic(-0.1, (4, 1), [-0.5], [0.0]), Quadratic(-0.1, (1, 4), [0.5], [0.0]), Quadratic(0.1, (3, 2), [0.5], [0.0]), Quadratic(0.1, (2, 3), [-0.5], [0.0]),
        Quadratic(0.1, (4, 1), [0.5], [1.0]), Quadratic(0.1, (1, 4), [-0.5], [-1.0]), Quadratic(-0.1, (3, 2), [-0.5], [-1.0]), Quadratic(-0.1, (2, 3), [0.5], [1.0])
    )
end

@time @testset "Quadraticization{Bosonic{:BdG}}" begin
    t = Hopping(:t, -1.0, 1)
    μ = Onsite(:μ, 2.0)
    Δ = Pairing(:Δ, 0.1, 1, 𝕕𝕕(:, :, :); amplitude=bond->rcoordinate(bond)[1]>0 ? 1 : -1)
    lattice = Lattice([0.0], [0.5]; vectors=[[1.0]])
    hilbert = Hilbert(Fock{:b}(1, 1), length(lattice))
    gen = Generator(bonds(lattice, 1), hilbert, (t, μ, Δ); half=false)
    tbakind = TBAKind(typeof(gen.terms), valtype(gen.hilbert))
    table = Table(gen.hilbert, Metric(tbakind, gen.hilbert))
    quadraticization = Quadraticization{typeof(tbakind)}(table)
    @test quadraticization(expand(gen)) == OperatorSum(
        Quadratic(-1.0, (2, 1), [-0.5], [0.0]), Quadratic(-1.0, (3, 4), [0.5], [-0.0]), Quadratic(-1.0, (1, 2), [0.5], [0.0]), Quadratic(-1.0, (4, 3), [-0.5], [-0.0]),
        Quadratic(-1.0, (2, 1), [0.5], [1.0]), Quadratic(-1.0, (3, 4), [-0.5], [-1.0]), Quadratic(-1.0, (1, 2), [-0.5], [-1.0]), Quadratic(-1.0, (4, 3), [0.5], [1.0]),
        Quadratic(2.0, (1, 1), [0.0], [0.0]), Quadratic(2.0, (3, 3), [-0.0], [-0.0]), Quadratic(2.0, (2, 2), [0.0], [0.0]), Quadratic(2.0, (4, 4), [-0.0], [-0.0]),
        Quadratic(-0.1, (4, 1), [-0.5], [0.0]), Quadratic(-0.1, (1, 4), [0.5], [0.0]), Quadratic(0.1, (3, 2), [0.5], [0.0]), Quadratic(0.1, (2, 3), [-0.5], [0.0]),
        Quadratic(0.1, (4, 1), [0.5], [1.0]), Quadratic(0.1, (1, 4), [-0.5], [-1.0]), Quadratic(-0.1, (3, 2), [-0.5], [-1.0]), Quadratic(-0.1, (2, 3), [0.5], [1.0])
    )
end

@time @testset "Quadraticization{Phononic}" begin
    T = Kinetic(:T, 0.5)
    V = Hooke(:V, 2.0, 1)
    lattice = Lattice([0.0], [0.5]; vectors=[[1.0]])
    hilbert = Hilbert(Phonon(1), length(lattice))
    gen = Generator(bonds(lattice, 1), hilbert, (T, V); half=false)
    tbakind = TBAKind(typeof(gen.terms), valtype(gen.hilbert))
    table = Table(gen.hilbert, Metric(tbakind, gen.hilbert))
    quadraticization = Quadraticization{typeof(tbakind)}(table)
    @test quadraticization(expand(gen)) == OperatorSum(
        Quadratic(0.5, (1, 1), [0.0], [0.0]), Quadratic(0.5, (1, 1), [-0.0], [-0.0]), Quadratic(0.5, (2, 2), [0.0], [0.0]), Quadratic(0.5, (2, 2), [-0.0], [-0.0]),
        Quadratic(4.0, (4, 4), [0.0], [0.0]), Quadratic(4.0, (4, 4), [-0.0], [-0.0]), Quadratic(-2.0, (4, 3), [-0.5], [0.0]), Quadratic(-2.0, (3, 4), [0.5], [-0.0]),
        Quadratic(-2.0, (3, 4), [0.5], [0.0]), Quadratic(-2.0, (4, 3), [-0.5], [-0.0]), Quadratic(4.0, (3, 3), [0.0], [0.0]), Quadratic(4.0, (3, 3), [-0.0], [-0.0]),
        Quadratic(-4.0, (4, 3), [0.5], [1.0]), Quadratic(-4.0, (3, 4), [-0.5], [-1.0])
    )
end

@time @testset "TBAMatrixization" begin
    ops = OperatorSum(
        Quadratic(-1.0, (2, 1), [-0.5], [0.0]), Quadratic(-1.0, (1, 2), [0.5], [0.0]), Quadratic(-1.0, (2, 1), [0.5], [1.0]),
        Quadratic(-1.0, (1, 2), [-0.5], [-1.0]), Quadratic(2.0, (1, 1), [0.0], [0.0]), Quadratic(2.0, (2, 2), [0.0], [0.0])
    )
    matrixization = TBAMatrixization{ComplexF64}([0.0], 2, :icoordinate)
    @test valtype(matrixization, ops) == Matrix{ComplexF64}
    @test zero(matrixization, ops) == zeros(ComplexF64, 2, 2)
    @test matrixization(first(ops)) == [0.0+0.0im 0.0+0.0im; -1.0+0.0im 0.0+0.0im]

    @test matrixization(ops) == [2.0+0.0im -2.0+0.0im; -2.0+0.0im 2.0+0.0im]
    @test TBAMatrixization{ComplexF64}([pi/2], 2, :icoordinate)(ops) == [2.0+0.0im -1.0+1.0im; -1.0-1.0im 2.0+0.0im]
    @test TBAMatrixization{ComplexF64}([pi/2], 2, :rcoordinate)(ops) ≈ [2.0+0.0im -√2+0.0im; -√2+0.0im 2.0+0.0im]
    @test TBAMatrixization{Float64}(nothing, 2, :rcoordinate)(ops) == TBAMatrixization{Float64}(nothing, 2, :icoordinate)(ops) == [2.0 -2.0; -2.0 2.0]
end

@time @testset "TBAMatrix" begin
    data = [2.0+0.0im -2.0+0.0im; -2.0+0.0im 2.0+0.0im]
    m = TBAMatrix(Hermitian(data), nothing)
    @test size(m) == (2, 2)
    @test [m[i, j] for i=1:2, j=1:2] == data
    @test ishermitian(m) == ishermitian(typeof(m)) == true
    @test Hermitian(m) == Hermitian(data)
    @test Matrix(m) == data

    es, vs = [0.0, 4.0], [-√2/2 -√2/2; -√2/2 √2/2]
    @test [eigen(m)...] ≈ [es, vs]
    @test eigvals(m) ≈ es
    @test eigvecs(m) ≈ vs

    data = [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 8.0 -4.0+4.0im; 0.0 0.0 -4.0-4.0im 8.0]
    commutator = [0 0 -im 0; 0 0 0 -im; im 0 0 0; 0 im 0 0]
    m = TBAMatrix(Hermitian(data), commutator)
    es = [3.695518130045136, 1.5307337294603554, 1.5307337294603642, 3.695518130045147]
    vs = [
        -0.679661508587653-0.6796615085876528im     0.43742624084814996+0.4374262408481499im    -0.43742624084815235-0.43742624084815235im  0.6796615085876522+0.6796615085876521im;
        0.9611865232676121im                        0.6186141223453353im                        -0.6186141223453371im                       -0.9611865232676166im;
        -0.18391507893355885+0.1839150789335589im   0.28576246307864384-0.28576246307864384im   0.28576246307864483-0.28576246307864483im   -0.18391507893355896+0.18391507893355902im
        0.26009519895275734                         0.40412915090295953                         0.40412915090295987                         0.26009519895275796
    ]
    @test [eigen(m)...] ≈ [es, vs]
    @test eigvals(m) ≈ es
    @test eigvecs(m) ≈ vs
end

@time @testset "TBA" begin
    t = Hopping(:t, 1.0, 1)
    μ = Onsite(:μ, 0.0)
    lattice = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0], [0.0, 1.0]])
    hilbert = Hilbert(Fock{:f}(1, 1), length(lattice))
    tba = TBA(lattice, hilbert, (t, μ))
    @test kind(tba) == kind(typeof(tba)) == Fermionic(:TBA)
    @test isnothing(getcontent(tba, :commutator))
    @test scalartype(tba) == scalartype(typeof(tba)) == Float64
    @test Parameters(tba) == (t=1.0, μ=0.0)
    @test dimension(tba) == 1

    A(t, μ, k=[0.0, 0.0]; kwargs...) = [2t*cos(k[1])+2t*cos(k[2])+μ;;]
    another = TBA{Fermionic{:TBA}}(Formula(A, (t=1.0, μ=0.0)))
    @test scalartype(another) == scalartype(typeof(another)) == Float64
    @test Parameters(another) == (t=1.0, μ=0.0)
    @test dimension(another) == 1

    third = TBA{Fermionic{:TBA}}(tba.H)
    @test scalartype(third) == scalartype(typeof(third)) == Float64
    @test Parameters(third) == (t=1.0, μ=0.0)
    @test dimension(third) == 1

    fourth = TBA{Fermionic{:TBA}}(tba.system, tba.quadraticization)
    @test scalartype(fourth) == scalartype(typeof(fourth)) == Float64
    @test Parameters(fourth) == (t=1.0, μ=0.0)
    @test dimension(fourth) == 1

    fifth = TBA(tba.system)
    @test scalartype(fifth) == scalartype(typeof(fifth)) == Float64
    @test Parameters(fifth) == (t=1.0, μ=0.0)
    @test dimension(fifth) == 1

    sixth = TBA{Fermionic{:TBA}}(tba.system)
    @test scalartype(sixth) == scalartype(typeof(sixth)) == Float64
    @test Parameters(sixth) == (t=1.0, μ=0.0)
    @test dimension(sixth) == 1

    path = ReciprocalPath(lattice, rectangle"Γ-X-M-Γ", length=8)
    for k in path
        @test matrix(tba, k) ≈ matrix(another, k) ≈ matrix(third, k) ≈ matrix(fourth, k) ≈ matrix(fifth, k) ≈ matrix(sixth, k)
        @test [eigen(tba, k)...] ≈ [eigen(another, k)...] ≈ [eigen(third, k)...] ≈ [eigen(fourth, k)...] ≈ [eigen(fifth, k)...] ≈ [eigen(sixth, k)...]
        @test eigvals(tba, k) ≈ eigvals(another, k) ≈ eigvals(third, k) ≈ eigvals(fourth, k) ≈ eigvals(fifth, k) ≈ eigvals(sixth, k)
        @test eigvecs(tba, k) ≈ eigvecs(another, k) ≈ eigvecs(third, k) ≈ eigvecs(fourth, k) ≈ eigvecs(fifth, k) ≈ eigvecs(sixth, k)
    end
    @test [eigen(tba, path)...] ≈ [eigen(another, path)...] ≈ [eigen(third, path)...] ≈ [eigen(fourth, path)...] ≈ [eigen(fifth, path)...] ≈ [eigen(sixth, path)...]
    @test eigvals(tba, path) ≈ eigvals(another, path) ≈ eigvals(third, path) ≈ eigvals(fourth, path) ≈ eigvals(fifth, path) ≈ eigvals(sixth, path)
    @test eigvecs(tba, path) ≈ eigvecs(another, path) ≈ eigvecs(third, path) ≈ eigvecs(fourth, path) ≈ eigvecs(fifth, path) ≈ eigvecs(sixth, path)

    update!(tba; μ=2.2)
    update!(another; μ=2.2)
    update!(third; μ=2.2)
    update!(fourth; μ=2.2)
    update!(fifth; μ=2.2)
    update!(sixth; μ=2.2)
    @test [eigen(tba, path)...] ≈ [eigen(another, path)...] ≈ [eigen(third, path)...] ≈ [eigen(fourth, path)...] ≈ [eigen(fifth, path)...] ≈ [eigen(sixth, path)...]
end

@time @testset "EnergyBands and BerryCurvature" begin
    unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0], [0.0, 1.0]])
    hilbert = Hilbert(Fock{:f}(1, 1), length(unitcell))
    t = Hopping(:t, 1.0, 1)
    μ = Onsite(:μ, 3.5)
    Δ = Pairing(:Δ, Complex(0.5), 1, 𝕕𝕕(:, :, :); amplitude=bond->exp(im*azimuth(rcoordinate(bond))))
    sc = Algorithm(Symbol("p+ip"), TBA(unitcell, hilbert, (t, μ, Δ)))
    @test kind(sc) == kind(typeof(sc)) == kind(sc.frontend)
    @test scalartype(sc) == scalartype(typeof(sc)) == scalartype(sc.frontend)
    @test matrix(sc) == matrix(sc.frontend)
    @test eigen(sc) == Eigen(eigvals(sc), eigvecs(sc))

    path = ReciprocalPath(unitcell, rectangle"Γ-X-M-Γ", length=100)
    @test eigen(sc, path) == (eigvals(sc, path), eigvecs(sc, path))
    Plots.savefig(Plots.plot(sc(:EB, EnergyBands(path, 1:2))), "PlotsEB.png")
    Makie.save("MakieEB.png", Makie.plot(sc(:EB, EnergyBands(path, 1:2))))
    Plots.savefig(Plots.plot(sc(:EB, EnergyBands(path, :, 1:1, 2:2)); weightlabels=["particle", "hole"], legend=(0.4, 0.1), legendfont=10, legendcolumn=-1), "PlotsEBWeighted.png")
    Makie.save("MakieEBWeighted.png", Makie.plot(sc(:EB, EnergyBands(path, :, 1:1, 2:2)); weightlabels=["particle", "hole"]))

    brillouin = BrillouinZone(unitcell, 100)
    Plots.savefig(Plots.plot(sc(:BerryCurvatureAbelian, BerryCurvature(brillouin, [1, 2]))), "PlotsBerryCurvatureAbelian.png")
    Makie.save("MakieBerryCurvatureAbelian.png", Makie.plot(sc(:BerryCurvatureAbelian, BerryCurvature(brillouin, [1, 2]))))
    Plots.savefig(Plots.plot(sc(:BerryCurvatureNonabelian, BerryCurvature(brillouin, [1, 2], false))), "PlotsBerryCurvatureNonabelian.png")
    Makie.save("MakieBerryCurvatureNonabelian.png", Makie.plot(sc(:BerryCurvatureNonabelian, BerryCurvature(brillouin, [1, 2], false))))
    Plots.savefig(Plots.plot(sc(:BerryCurvatureKubo, BerryCurvature(brillouin, Kubo(0; d=0.1, kx=[1.0, 0.0], ky=[0.0, 1.0])))), "PlotsBerryCurvatureKubo.png")
    Makie.save("MakieBerryCurvatureKubo.png", Makie.plot(sc(:BerryCurvatureKubo, BerryCurvature(brillouin, Kubo(0; d=0.1, kx=[1.0, 0.0], ky=[0.0, 1.0])))))

    reciprocalzone = ReciprocalZone(unitcell, [-1.0=>1.0, -1.0=>1.0]; length=201, ends=(true, true))
    Plots.savefig(Plots.plot(sc(:BerryCurvatureExtendedAbelian, BerryCurvature(reciprocalzone, [1, 2]))), "PlotsBerryCurvatureExtendedAbelian.png")
    Makie.save("MakieBerryCurvatureExtendedAbelian.png", Makie.plot(sc(:BerryCurvatureExtendedAbelian, BerryCurvature(reciprocalzone, [1, 2]))))
    Plots.savefig(Plots.plot(sc(:BerryCurvatureExtendedNonabelian, BerryCurvature(reciprocalzone, [1, 2], false))), "PlotsBerryCurvatureExtendedNonabelian.png")
    Makie.save("MakieBerryCurvatureExtendedNonabelian.png", Makie.plot(sc(:BerryCurvatureExtendedNonabelian, BerryCurvature(reciprocalzone, [1, 2], false))))
    Plots.savefig(Plots.plot(sc(:BerryCurvatureExtendedKubo, BerryCurvature(reciprocalzone, Kubo(0; d=0.1, kx=[1., 0], ky=[0, 1.])))), "PlotsBerryCurvatureExtendedKubo.png")
    Makie.save("MakieBerryCurvatureExtendedKubo.png", Makie.plot(sc(:BerryCurvatureExtendedKubo, BerryCurvature(reciprocalzone, Kubo(0; d=0.1, kx=[1., 0], ky=[0, 1.])))))
    Plots.savefig(Plots.plot(sc(:BerryCurvaturePath, BerryCurvature(path, 0.0))), "PlotsBerryCurvaturePath.png")
    Makie.save("MakieBerryCurvaturePath.png", Makie.plot(sc(:BerryCurvaturePath, BerryCurvature(path, 0.0))))
end

@time @testset "FermiSurface and DensityOfStates" begin
    unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0], [0.0, 1.0]])
    hilbert = Hilbert(Fock{:f}(1, 2), length(unitcell))
    t = Hopping(:t, 1.0, 1)
    h = Onsite(:h, 0.1, 𝕕⁺𝕕(:, :, σᶻ))
    tba = Algorithm(:tba, TBA(unitcell, hilbert, (t, h)))

    brillouin = BrillouinZone(unitcell, 400)
    Plots.savefig(Plots.plot(tba(:FermiSurface, FermiSurface(brillouin, 0.0))), "PlotsFermiSurface.png")
    Makie.save("MakieFermiSurface.png", Makie.plot(tba(:FermiSurface, FermiSurface(brillouin, 0.0))))
    Plots.savefig(Plots.plot(tba(:FermiSurfaceSpinDependent, FermiSurface(brillouin, 0.0, :, [1], [2])); weightlabels=("↑", "↓")), "PlotsFermiSurfaceSpinDependent.png")
    Makie.save("MakieFermiSurfaceSpinDependent.png", Makie.plot(tba(:FermiSurfaceSpinDependent, FermiSurface(brillouin, 0.0, :, [1], [2])); weightlabels=("↑", "↓")))
    dos = tba(:DensityOfStates, DensityOfStates(brillouin, :, :, [1], [2]))
    Plots.savefig(Plots.plot(dos), "PlotsDensityOfStates.png")
    Makie.save("MakieDensityOfStates.png", Makie.plot(dos))
    @test isapprox(sum(dos.data.values[:, 1]), 2.0; atol=10^-3)

    # Test fermi_dirac, occupation, chemical_potential
    @test fermi_dirac(-1.0, 0.0, 0.0) == 1.0
    @test fermi_dirac( 1.0, 0.0, 0.0) == 0.0
    @test fermi_dirac( 0.0, 0.0, 0.0) == 1.0
    @test isapprox(fermi_dirac(0.0, 0.0, 1.0), 0.5; atol=1e-10)
    @test isapprox(fermi_dirac(100.0, 0.0, 1e-10), 0.0; atol=1e-10)

    # occupation: dos.values has columns (total, spin_up, spin_down)
    n_full = occupation(dos.data, 100.0, 0.0)
    @test isapprox(n_full[1], 2.0; atol=0.05)   # total, fully occupied
    @test isapprox(n_full[2], 1.0; atol=0.05)   # spin up

    n_empty = occupation(dos.data, -100.0, 0.0)
    @test isapprox(n_empty[1], 0.0; atol=0.01)  # total, empty

    # chemical_potential with total DOS (orbitals=:)
    dos_tot = tba(:DensityOfStates_total, DensityOfStates(brillouin))
    μ = chemical_potential(dos_tot.data, 1.0; T=0.0)
    n_half = occupation(dos_tot.data, μ, 0.0)
    @test isapprox(n_half[1], 1.0; atol=0.05)

    reciprocalzone = ReciprocalZone(unitcell, [-1.0=>1.0, -1.0=>1.0]; length=401, ends=(true, true))
    Plots.savefig(Plots.plot(tba(:FermiSurfaceExtended, FermiSurface(reciprocalzone, 0.0))), "PlotsFermiSurfaceExtended.png")
    Makie.save("MakieFermiSurfaceExtended.png", Makie.plot(tba(:FermiSurfaceExtended, FermiSurface(reciprocalzone, 0.0))))
    Plots.savefig(Plots.plot(tba(:FermiSurfaceExtendedSpinDependent, FermiSurface(reciprocalzone, 0.0, :, [1], [2])); weightlabels=("↑", "↓")), "PlotsFermiSurfaceExtendedSpinDependent.png")
    Makie.save("MakieFermiSurfaceExtendedSpinDependent.png", Makie.plot(tba(:FermiSurfaceExtendedSpinDependent, FermiSurface(reciprocalzone, 0.0, :, [1], [2])); weightlabels=("↑", "↓")))
end

@time @testset "InelasticNeutronScatteringSpectra" begin
    unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0], [0.0, 1.0]])
    hilbert = Hilbert(Phonon(2), length(unitcell))
    T = Kinetic(:T, 0.5)
    V₁ = Hooke(:V₁, 0.5, 1)
    V₂ = Hooke(:V₂, 0.25, 2)
    phonon = Algorithm(:Phonon, TBA(unitcell, hilbert, (T, V₁, V₂)))
    path = ReciprocalPath(unitcell, rectangle"Γ-X-M-Γ", length=100)

    energybands = phonon(:EB, EnergyBands(path))
    Plots.savefig(Plots.plot(energybands), "PlotsPhononEB.png")
    Makie.save("MakiePhononEB.png", Makie.plot(energybands))

    inelastic = phonon(:INSS, InelasticNeutronScatteringSpectra(path, range(0.0, 2.5, length=501)); fwhm=0.05, rescale=x->log(1+x))
    plt = Plots.plot()
    Plots.plot!(plt, inelastic)
    Plots.plot!(plt, energybands, color=:white, linestyle=:dash)
    Plots.savefig(plt, "PlotsInelasticNeutronScatteringSpectra.png")
    # Makie equivalent
    fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1])
    Makie.plot!(ax, inelastic)
    Makie.plot!(ax, energybands; color=:white, linestyle=:dash)
    Makie.save("MakieInelasticNeutronScatteringSpectra.png", fig)
end

@time @testset "Fitting" begin
    lattice = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0], [0.0, 1.0]])
    A(t, μ, Δ, k=[0.0, 0.0]; kwargs...) = [2t*cos(k[1])+2t*cos(k[2])+μ -2im*Δ*sin(k[1])-2Δ*sin(k[2]); 2im*Δ*sin(k[1])-2Δ*sin(k[2]) -2t*cos(k[1])-2t*cos(k[2])-μ]
    tba = TBA{Fermionic{:BdG}}(Formula(A, (t=2.0, μ=0.3, Δ=1.0)))
    samplesets = [
        SampleNode(reciprocals(lattice), [0.0, 0.0], [1, 2], [-16.0, 16.0], 1.0),
        SampleNode(reciprocals(lattice), [0.25, 0.0], [1, 2], [-10.0, 10.0], 1.0),
        SampleNode(reciprocals(lattice), [0.25, 0.25], [1, 2], [-8.485281374238571, 8.485281374238571], 1.0)
    ]
    op = optimize!(tba, samplesets)[2]
    @test isapprox(op.minimizer, [4.0, 0.0, 3.0], atol=5*10^-10)
end

toml = Artifacts.find_artifacts_toml(@__DIR__)
const dir = Pkg.Artifacts.ensure_artifact_installed("WannierDataSets", toml)
@time @testset "Wannier90" begin
    silicon₁ = readlattice(dir, "silicon"; name=:silicon)
    silicon₂ = readlattice(dir, "silicon-second"; name=:silicon)
    silicon₃ = readlattice(dir; name=:silicon)
    @test silicon₁.coordinates ≈ silicon₂.coordinates ≈ silicon₃.coordinates
    @test silicon₁.vectors ≈ silicon₂.vectors ≈ silicon₃.vectors

    seedname = "silicon"
    wan = Algorithm(:silicon, W90(dir, seedname))
    @test W90Matrixization([0.0, 0.0, 0.0], wan.frontend.lattice.vectors, wan.frontend.centers, :icoordinate)(first(wan.frontend.H)) ≈ ComplexF64[
        0.016239+4.75e-6im -0.00301675+2.5e-6im -0.0030155-6.0e-6im -0.0030155-3.25e-6im -0.01056425-1.3e-5im 0.00237225+7.5e-7im 0.002374+5.75e-6im 0.0023735+2.5e-6im;
        -0.0030155+3.25e-6im 0.01623875-2.5e-7im -0.003016-1.0e-6im -0.00301625-3.25e-6im 0.0023735+1.0e-6im 0.00477025+2.5e-7im -0.0055565-2.5e-6im -0.00555775+3.25e-6im;
        -0.0030175-6.0e-6im -0.0030155+2.5e-7im 0.01623925+5.0e-7im -0.0030155+5.75e-6im 0.0023755+4.25e-6im -0.0055575-1.25e-6im 0.004769+1.5e-6im -0.0055575-2.75e-6im;
        -0.00301575-1.5e-6im -0.00301625-2.5e-7im -0.00301625+5.75e-6im 0.0162385-1.75e-6im 0.00237375+3.5e-6im -0.00555675+2.75e-6im -0.0055585-4.0e-6im 0.00477075-7.5e-7im;
        0.008246-3.0e-6im 0.0022645+0.0im 0.00226525+4.25e-6im 0.0022645-7.5e-7im 0.016239+4.75e-6im -0.00301475+4.0e-6im -0.00301675-8.25e-6im -0.003016+2.5e-7im;
        0.00226475+7.5e-7im -0.00638575-2.25e-6im -0.00396775+1.5e-6im -0.0039695-1.25e-6im -0.00301675+3.0e-6im 0.016238-5.0e-7im -0.0030145+7.5e-7im -0.0030165-1.25e-6im;
        0.00226725+2.5e-6im -0.00396925+1.75e-6im -0.0063855-3.5e-6im -0.00396775-3.25e-6im -0.0030155-7.5e-6im -0.0030175-7.5e-7im 0.016239+1.0e-6im -0.00301575+5.5e-6im;
        0.0022645+5.0e-7im -0.003968-1.75e-6im -0.0039695-3.25e-6im -0.00638525+4.25e-6im -0.0030165-2.5e-6im -0.0030155-4.25e-6im -0.00301625+7.0e-6im 0.016239-2.0e-6im
    ]
    path = readpath(dir, seedname)
    bands = readbands(dir, seedname)
    plt = Plots.plot()
    Plots.plot!(plt, bands...; xlim=(0.0, distance(path)), label=false, color=:green, alpha=0.6, lw=2.5)
    Plots.plot!(plt, wan(:EB, EnergyBands(path); gauge=:rcoordinate), color=:black)
    Plots.savefig(plt, "PlotsSiliconWannier90Center.png")
    # Makie equivalent
    fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1])
    Makie.plot!(ax, bands[1], bands[2]; color=:green, alpha=0.6, linewidth=2.5)
    Makie.plot!(ax, wan(:EB, EnergyBands(path); gauge=:rcoordinate); color=:black)
    Makie.save("MakieSiliconWannier90Center.png", fig)

    hilbert = Hilbert(Fock{:f}(4, 1), length(wan.frontend.lattice))
    another = Algorithm(:silicon, W90(wan.frontend.lattice, hilbert, wan.frontend.H))
    update!(another)
    plt = Plots.plot()
    Plots.plot!(plt, bands...; xlim=(0.0, distance(path)), label=false, color=:green, alpha=0.6, lw=2.5)
    Plots.plot!(plt, another(:EB, EnergyBands(path); gauge=:rcoordinate), color=:black)
    Plots.savefig(plt, "PlotsSiliconWannier90Atom.png")
    # Makie equivalent
    fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1])
    Makie.plot!(ax, bands[1], bands[2]; color=:green, alpha=0.6, linewidth=2.5)
    Makie.plot!(ax, another(:EB, EnergyBands(path); gauge=:rcoordinate); color=:black)
    Makie.save("MakieSiliconWannier90Atom.png", fig)

    tba = Algorithm(wan, hilbert; complement_spin=false)
    k = rand(3)
    @test matrix(tba, k; gauge=:icoordinate) ≈ matrix(wan, k; gauge=:icoordinate)
    @test matrix(tba, k; gauge=:rcoordinate) ≈ matrix(wan, k; gauge=:rcoordinate)
    plt = Plots.plot()
    Plots.plot!(plt, bands...; xlim=(0.0, distance(path)), label=false, color=:green, alpha=0.6, lw=2.5)
    Plots.plot!(plt, tba(:EB, EnergyBands(path)), color=:black)
    Plots.savefig(plt, "PlotsSiliconOperatorizationSpinless.png")
    # Makie equivalent
    fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1])
    Makie.plot!(ax, bands[1], bands[2]; color=:green, alpha=0.6, linewidth=2.5)
    Makie.plot!(ax, tba(:EB, EnergyBands(path)); color=:black)
    Makie.save("MakieSiliconOperatorizationSpinless.png", fig)

    tba = Algorithm(wan, hilbert; complement_spin=true)
    plt = Plots.plot()
    Plots.plot!(plt, bands...; xlim=(0.0, distance(path)), label=false, color=:green, alpha=0.6, lw=2.5)
    Plots.plot!(plt, tba(:EB, EnergyBands(path)), color=:black)
    Plots.savefig(plt, "PlotsSiliconOperatorizationSpinful.png")
    # Makie equivalent
    fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1])
    Makie.plot!(ax, bands[1], bands[2]; color=:green, alpha=0.6, linewidth=2.5)
    Makie.plot!(ax, tba(:EB, EnergyBands(path)); color=:black)
    Makie.save("MakieSiliconOperatorizationSpinful.png", fig)

    @testset "choosehamiltonian" begin
        ham = wan.frontend.H
        lat = wan.frontend.lattice
        cnt = wan.frontend.centers

        # No filters: should not lose any R-vector
        result = choosehamiltonian(ham, lat, cnt)
        @test length(result.operators) == length(ham.operators)

        # min_hopping_norm filter
        result = choosehamiltonian(ham, lat, cnt; min_hopping_norm=0.01)
        @test length(result.operators) <= length(ham.operators)

        # max_distance filter
        result = choosehamiltonian(ham, lat, cnt; max_distance=5.0)
        @test length(result.operators) <= length(ham.operators)

        # Combined filters
        result = choosehamiltonian(ham, lat, cnt; min_hopping_norm=0.001, max_distance=8.0)

        # Result is usable to construct W90
        filtered_wan = W90(lat, cnt, result)
        @test dimension(filtered_wan) == dimension(wan.frontend)

        # Filtered result structural correctness
        for m in result.operators
            @test size(m.value) == size(first(ham).value)
            @test m.id isa NTuple{3, Int}
        end
    end

    @testset "distancehamiltonian" begin
        ham = wan.frontend.H
        lat = wan.frontend.lattice
        cnt = wan.frontend.centers

        dists, idxs, vals = distancehamiltonian(ham, lat, cnt)
        @test length(dists) == length(idxs) == length(vals)
        @test all(d -> d >= 0.0, dists)
        @test all(v -> v >= 0.0, vals)
        @test length(dists) > 0
        for (R, i, j) in idxs
            @test R isa NTuple{3, Int}
            @test 1 <= i <= size(cnt, 2)
            @test 1 <= j <= size(cnt, 2)
        end
    end

    @testset "foldcenters" begin
        lat = Lattice((0.0, 0.0, 0.0); name = :chain,
                      vectors = [[2.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        centers = [0.0 1.0 -2.0 2.0 3.0; zeros(1, 5); zeros(1, 5)]
        folded = foldcenters(centers, lat)
        @test folded isa Matrix{Float64}
        @test size(folded) == size(centers)
        @test folded[1, :] ≈ [0.0, 1.0, 0.0, 0.0, 1.0]
        @test foldcenters(folded, lat) ≈ folded
        # a centre exactly on the cell boundary is folded to zero, not kept at the boundary
        @test foldcenters(reshape([2.0 - 1e-12, 0.0, 0.0], 3, 1), lat)[1, 1] ≈ 0.0 atol = 1e-9
        # centres already inside the first cell are unchanged
        @test foldcenters(reshape([1.0, 0.0, 0.0], 3, 1), lat) ≈ reshape([1.0, 0.0, 0.0], 3, 1)
        @test all(0.0 .<= [decompose(folded[:, i], lat.vectors)[1] for i in 1:size(folded, 2)] .< 1.0)
        @test_throws ErrorException foldcenters(zeros(2, 3), lat)

        # an unfolded centre shifts the hopping distance by a lattice vector
        ham = OperatorSum{W90Hoppings}()
        value0 = ComplexF64[0.0 1.0; 1.0 0.0]
        ham.contents[(0, 0, 0)] = W90Hoppings(copy(value0), (0, 0, 0))
        ham.contents[(1, 0, 0)] = W90Hoppings(zeros(ComplexF64, 2, 2), (1, 0, 0))
        ham.contents[(-1, 0, 0)] = W90Hoppings(zeros(ComplexF64, 2, 2), (-1, 0, 0))
        hamc = StaticGenerator(ham)
        centers = [0.0 2.0; zeros(1, 2); zeros(1, 2)]   # orbital 2 is one vector away, i.e. at the same site
        # folding makes the (i, j) = (1, 2) hopping onsite and keeps it below the cutoff
        @test choosehamiltonian(hamc, lat, centers; max_distance = 0.5).operators.contents[(0, 0, 0)].value ≈ value0
        @test iszero(maximum(abs, choosehamiltonian(hamc, lat, centers; max_distance = 0.5, fold = false).operators.contents[(0, 0, 0)].value))
        dists1, idxs1, _ = distancehamiltonian(hamc, lat, centers; fold = true)
        dists2, idxs2, _ = distancehamiltonian(hamc, lat, centers; fold = false)
        pos1 = findfirst(idx -> idx == ((0, 0, 0), 1, 2), idxs1)
        pos2 = findfirst(idx -> idx == ((0, 0, 0), 1, 2), idxs2)
        @test dists1[pos1] ≈ 0.0 atol = 1e-10
        @test dists2[pos2] ≈ 2.0 atol = 1e-10
        # the read-back centres of a Wannier90 run may be outside the first cell
        foldedwan = foldcenters(wan.frontend.centers, wan.frontend.lattice)
        @test all(-1e-6 .<= decompose(foldedwan[:, i], wan.frontend.lattice.vectors)[k] .< 1.0
                  for i in 1:size(foldedwan, 2), k in 1:3)
        # folding an atom position only shifts it by a lattice vector, so it stays an
        # equivalent position even when it starts inside the Wigner-Seitz cell
        latwan = wan.frontend.lattice
        foldedlat = foldcenters(latwan.coordinates, latwan)
        fracshift(c, i) = decompose(foldedlat[:, i] - latwan.coordinates[:, i], latwan.vectors)[c]
        @test all(abs(fracshift(k, i) - round(fracshift(k, i))) < 1e-6
                  for i in 1:size(latwan.coordinates, 2), k in 1:3)
        @test foldcenters(foldedlat, latwan) ≈ foldedlat

        # `W90(path, seedname)` folds by default, `fold = false` keeps the file values
        wanraw = W90(dir, "silicon"; fold = false)
        @test wanraw.centers ≠ wan.frontend.centers
        @test foldcenters(wanraw.centers, wanraw.lattice) ≈ wan.frontend.centers
        # folding only changes the gauge of `:rcoordinate`, never the eigenvalues
        kk = [0.13, 0.21, -0.07]
        @test Matrix(matrix(wanraw, kk; gauge = :icoordinate).H) ==
              Matrix(matrix(wan.frontend, kk; gauge = :icoordinate).H)
        @test eigvals(Hermitian(Matrix(matrix(wanraw, kk; gauge = :rcoordinate).H))) ≈
              eigvals(Hermitian(Matrix(matrix(wan.frontend, kk; gauge = :rcoordinate).H)))
    end

    @testset "showhamiltonian" begin
        ham = wan.frontend.H
        lat = wan.frontend.lattice
        cnt = wan.frontend.centers
        nsublattice = size(lat.coordinates, 2)  # number of sites in the unit cell

        io = IOBuffer()
        showhamiltonian(io, ham, lat, cnt, nsublattice)
        output = String(take!(io))
        @test length(output) > 0
        @test occursin("sublattice(", output)
        @test occursin("distance = ", output)

        # Write to file
        tmpfile = tempname() * ".txt"
        showhamiltonian(tmpfile, ham, lat, cnt, nsublattice)
        @test isfile(tmpfile)
        file_output = read(tmpfile, String)
        @test occursin("sublattice(", file_output)
        rm(tmpfile)

        # `digits` rounds the matrix elements and keeps the standard display
        io = IOBuffer()
        showhamiltonian(io, ham, lat, cnt, nsublattice; digits = 4)
        @test occursin("Matrix{ComplexF64}", String(take!(io)))
        @test_throws ErrorException showhamiltonian(IOBuffer(), ham, lat, cnt, nsublattice; digits = -1)

        # `fold` decides whether the printed centres and distances use the folded coordinates
        cntraw = W90(dir, "silicon"; fold = false).centers
        @test cntraw ≠ cnt
        showhamiltonian(io, ham, lat, cntraw, nsublattice)
        @test String(take!(io)) == output                       # folding restores the same text
        showhamiltonian(io, ham, lat, cntraw, nsublattice; fold = false)
        @test String(take!(io)) ≠ output
    end

    @testset "showW90" begin
        nsublattice = size(wan.frontend.lattice.coordinates, 2)  # number of sites in the unit cell
        io = IOBuffer()
        showW90(io, wan.frontend, nsublattice)
        output = String(take!(io))
        @test length(output) > 0
        @test occursin("sublattice(", output)

        tmpfile = tempname() * ".txt"
        showW90(tmpfile, wan.frontend, nsublattice)
        @test isfile(tmpfile)
        file_output = read(tmpfile, String)
        @test occursin("sublattice(", file_output)
        rm(tmpfile)

        # the store of a `fold = false` system keeps the file values; folding on print
        # restores exactly the text above, and `fold = false` prints the file values
        wanraw = W90(dir, "silicon"; fold = false)
        io = IOBuffer()
        showW90(io, wanraw, nsublattice)
        @test String(take!(io)) == output
        showW90(io, wanraw, nsublattice; fold = false)
        @test String(take!(io)) ≠ output
    end

    @testset "showhamiltonian (SOC)" begin
        # synthetic spinor data: 2 sites x 1 orbital x 2 spins, stored in the
        # Wannier90 order ↑o₁, ↓o₁, ↑o₂, ↓o₂ (site 1 = 1:2, site 2 = 3:4)
        lat = Lattice((0.0, 0.0, 0.0), (1.0, 0.0, 0.0); name = :chain,
                      vectors = [[2.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        cnt = [0.0 0.0 1.0 1.0;
               0.0 0.0 0.0 0.0;
               0.0 0.0 0.0 0.0]
        M0 = ComplexF64[1.0 0.2im 0.3 0.0;
                        -0.2im 0.9 0.1im 0.2;
                        0.3 -0.1im 0.8 0.15im;
                        0.0 0.2 -0.15im 0.7]
        M0 = (M0 + M0') / 2
        M1 = zeros(ComplexF64, 4, 4)
        M1[3:4, 1:2] = [0.25 0.1im; -0.1im 0.15]
        hop = OperatorSum{W90Hoppings}()
        hop.contents[(0, 0, 0)] = W90Hoppings(M0, (0, 0, 0))
        hop.contents[(1, 0, 0)] = W90Hoppings(M1, (1, 0, 0))
        hop.contents[(-1, 0, 0)] = W90Hoppings(M1', (-1, 0, 0))
        wanSpinor = W90(lat, cnt, StaticGenerator(hop))

        io = IOBuffer()
        showhamiltonian(io, wanSpinor.H, wanSpinor.lattice, wanSpinor.centers, 2; spinors = true)
        output = String(take!(io))
        @test occursin("spin-orbit coupling", output)
        @test occursin("↑o₁, ↓o₁", output)
        @test occursin("norbit/sublattice = 1, spins = 2", output)
        @test count(line -> occursin("sublattice(", line), split(output, '\n')) ==
              2^2 * length(wanSpinor.H.operators)

        # the Wannier90 spinor order is kept: the printed 2×2 blocks are the raw blocks
        @test occursin("1.0+0.0im", output)        # block (0,0,0)[1:2, 1:2]
        @test occursin("0.0+0.2im", output)
        @test occursin("0.0-0.2im", output)
        @test occursin("0.25+0.0im", output)       # block (1,0,0)[3:4, 1:2]
        @test occursin("0.0+0.1im", output)

        # showW90 forwards to the same printer
        io2 = IOBuffer()
        showW90(io2, wanSpinor, 2; spinors = true)
        @test String(take!(io2)) == output

        # the filename variant writes the same text
        tmpfile = tempname() * ".txt"
        showhamiltonian(tmpfile, wanSpinor.H, wanSpinor.lattice, wanSpinor.centers, 2; spinors = true)
        @test isfile(tmpfile)
        @test read(tmpfile, String) == output
        rm(tmpfile)

        # without `spinors` the same data are printed by the spinless path
        io3 = IOBuffer()
        showhamiltonian(io3, wanSpinor.H, wanSpinor.lattice, wanSpinor.centers, 2)
        output3 = String(take!(io3))
        @test !occursin("spin-orbit coupling", output3)
        @test !occursin("spin order", output3)
        @test occursin("norbit/sublattice = 2, R-vectors", output3)

        # 4 Wannier functions cannot be split into spinor pairs on 3 sublattices
        @test_throws ErrorException showhamiltonian(IOBuffer(), wanSpinor.H, wanSpinor.lattice,
                                                    wanSpinor.centers, 3; spinors = true)

        # `digits` keeps the significant digits, adds no trailing zeros
        io4 = IOBuffer()
        showhamiltonian(io4, wanSpinor.H, wanSpinor.lattice, wanSpinor.centers, 2; spinors = true, digits = 8)
        output4 = String(take!(io4))
        @test occursin("1.0+0.0im", output4)
        @test occursin("0.0+0.2im", output4)
        @test occursin("0.0-0.2im", output4)
        @test !occursin("1.00000000", output4)
        @test occursin("Matrix{ComplexF64}", output4)
        io5 = IOBuffer()
        showW90(io5, wanSpinor, 2; spinors = true, digits = 8)
        @test String(take!(io5)) == output4

        # all the digits of the stored data survive, binary noise is removed
        Mnoisy = copy(M0)
        Mnoisy[1, 1] = 0.1 + 0.2                     # 0.30000000000000004
        Mnoisy[2, 2] = 4.91394713456
        hopnoisy = OperatorSum{W90Hoppings}()
        hopnoisy.contents[(0, 0, 0)] = W90Hoppings(Mnoisy, (0, 0, 0))
        wannoise = W90(lat, cnt, StaticGenerator(hopnoisy))
        io6 = IOBuffer()
        showhamiltonian(io6, wannoise.H, wannoise.lattice, wannoise.centers, 2; spinors = true, digits = 16)
        output6 = String(take!(io6))
        @test occursin("0.3+0.0im", output6)
        @test !occursin("0.30000000000000004", output6)
        @test occursin("4.91394713456+0.0im", output6)

        # without `digits` every stored digit is printed, at full precision
        io7 = IOBuffer()
        showhamiltonian(io7, wannoise.H, wannoise.lattice, wannoise.centers, 2; spinors = true)
        output7 = String(take!(io7))
        @test occursin("4.91394713456+0.0im", output7)
        @test !occursin("4.91395+", output7)          # Julia's own display would shorten it
        @test occursin("0.30000000000000004+0.0im", output7)
        # `digits` is the number of significant digits: 1..17 only
        @test_throws ErrorException showhamiltonian(IOBuffer(), wannoise.H, wannoise.lattice,
                                                    wannoise.centers, 2; spinors = true, digits = 0)
        @test_throws ErrorException showhamiltonian(IOBuffer(), wannoise.H, wannoise.lattice,
                                                    wannoise.centers, 2; spinors = true, digits = 18)

        @test_throws ErrorException showhamiltonian(IOBuffer(), wanSpinor.H, wanSpinor.lattice,
                                                    wanSpinor.centers, 2; spinors = true, digits = -1)
    end
end

@testset "gaugetransform and Wannier90SOPT" begin
    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    blochmatrix(wan, k; gauge = :icoordinate) = Matrix(matrix(wan, k; gauge = gauge).H)
    spectrum(M) = sort(real.(eigvals(Hermitian(M))))
    unitarydeviation(U) = maximum(abs.(U' * U - I))

    """
    orderedmatrix(hilbert, gens, k, keytowannier)

    Bloch matrix of the operators of `gens` at `k`. With `keytowannier === nothing`
    the result is in `Table` order `(site, orbital, spin)`; otherwise `keytowannier`
    maps a table key `(site, orbital, spin)` to a Wannier index and the matrix is
    reordered accordingly.
    """
    function orderedmatrix(hilbert, gens, k, keytowannier)
        table = Table(hilbert, OperatorIndexToTuple(:site, :orbital, :spin))
        tblkeys = collect(keys(table))
        position = Dict{Any, Int}(key => index for (key, index) in pairs(table))
        M = zeros(ComplexF64, length(tblkeys), length(tblkeys))
        for gen in gens, op in expand(gen)
            i1, i2 = id(op)[1], id(op)[2]
            key1 = (i1.index.site, i1.index.internal.orbital, i1.index.internal.spin)
            key2 = (i2.index.site, i2.index.internal.orbital, i2.index.internal.spin)
            M[position[key1], position[key2]] += value(op) * exp(im * sum(k .* icoordinate(op)))
        end
        keytowannier === nothing && return M
        permutation = Vector{Int}(undef, length(tblkeys))
        for key in tblkeys
            permutation[keytowannier(key)] = position[key]
        end
        return M[permutation, permutation]
    end

    # ------------------------------------------------------------------
    # synthetic Wannier90 data: 2 sites x 2 orbitals, nearest-neighbour chain
    # ------------------------------------------------------------------
    lattice = Lattice((0.0, 0.0, 0.0), (1.0, 0.0, 0.0); name = :chain,
                      vectors = [[2.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    norb, nsites = 2, length(lattice)
    nwan = norb * nsites
    centers = zeros(Float64, 3, nwan)
    for site in 1:nsites, orbital in 1:norb
        centers[:, (site - 1) * norb + orbital] = [site - 1.0, 0.0, 0.0]
    end
    M0 = zeros(ComplexF64, nwan, nwan)
    M0[1:2, 1:2] = [1.0 0.2im; -0.2im -0.5]
    M0[3:4, 3:4] = [0.8 0.15im; -0.15im -0.9]
    M0[1:2, 3:4] = [0.3 0.0; 0.1im 0.2]
    M0[3:4, 1:2] = M0[1:2, 3:4]'
    M1 = zeros(ComplexF64, nwan, nwan)
    M1[3:4, 1:2] = [-0.3 0.1; 0.4 -0.25]
    ham = OperatorSum{W90Hoppings}()
    ham.contents[(0, 0, 0)] = W90Hoppings((M0 + M0') / 2, (0, 0, 0))
    ham.contents[(1, 0, 0)] = W90Hoppings(M1, (1, 0, 0))
    ham.contents[(-1, 0, 0)] = W90Hoppings(M1', (-1, 0, 0))
    wan = W90(lattice, centers, StaticGenerator(ham))

    snapshot = Dict(R => copy(hop.value) for (R, hop) in wan.H.operators.contents)
    centers0 = copy(wan.centers)
    ks = ([0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.7, 0.2, 0.1], [1.1, 0.0, 0.0])

    # deterministic unitary rotations
    θ = 0.4
    R1 = ComplexF64[cos(θ) -sin(θ); sin(θ) cos(θ)]
    φ = 0.3
    R2 = ComplexF64[0 exp(im * φ); exp(-im * φ) 0]
    A = ComplexF64[0.2 0.1+0.3im 0.0 0.2;
                   0.1-0.3im -0.4 0.5im 0.0;
                   0.0 -0.5im 0.7 0.1;
                   0.2 0.0 0.1 -0.1]
    A = (A + A') / 2
    U = exp(im * A)
    @test unitarydeviation(U) < 1e-12
    @test unitarydeviation(R1) < 1e-12
    @test unitarydeviation(R2) < 1e-12

    # ==================================================================
    @testset "gaugetransform: full unitary" begin
        wanU = gaugetransform(wan, U)
        @test wanU !== wan
        @test wanU.centers == wan.centers
        @test wanU.centers !== wan.centers
        for (R, hop) in wan.H.operators.contents
            @test wanU.H.operators.contents[R].value !== hop.value
            @test hop.value == snapshot[R]
        end
        @test wan.centers == centers0
        for k in ks
            @test blochmatrix(wanU, k) ≈ U' * blochmatrix(wan, k) * U
            @test spectrum(blochmatrix(wanU, k)) ≈ spectrum(blochmatrix(wan, k)) atol = 1e-10
        end

        @test_throws ErrorException gaugetransform(wan, zeros(ComplexF64, nwan, nwan))
        @test_throws ErrorException gaugetransform(wan, 2 .* U)
        @test_throws ErrorException gaugetransform(wan, zeros(ComplexF64, nwan, nwan + 1))
        # without the unitarity check a general congruence is accepted
        @test gaugetransform(wan, 2 .* U; checkunitary = false) isa W90
    end

    # ==================================================================
    @testset "gaugetransform: per site" begin
        siteindices = Dict(1 => [1, 2], 2 => [3, 4])
        rotations = Dict(1 => R1, 2 => R2)
        Ublock = zeros(ComplexF64, nwan, nwan)
        Ublock[1:2, 1:2] = R1
        Ublock[3:4, 3:4] = R2

        wanP = gaugetransform(wan, siteindices, rotations)
        wanF = gaugetransform(wan, siteindices, site -> rotations[site])
        wanM = gaugetransform(wan, Ublock)
        for R in keys(snapshot)
            @test wanP.H.operators.contents[R].value ≈ wanM.H.operators.contents[R].value
            @test wanF.H.operators.contents[R].value ≈ wanM.H.operators.contents[R].value
            @test wanP.H.operators.contents[R].value ≈ Ublock' * snapshot[R] * Ublock
        end
        for k in ks
            @test blochmatrix(wanP, k) ≈ Ublock' * blochmatrix(wan, k) * Ublock
            # all Wannier functions of a site share one centre -> exact for :rcoordinate
            @test blochmatrix(wanP, k; gauge = :rcoordinate) ≈
                  Ublock' * blochmatrix(wan, k; gauge = :rcoordinate) * Ublock
        end

        # sites without a rotation keep the identity
        wanPart = gaugetransform(wan, siteindices, Dict(1 => R1))
        Upart = Matrix{ComplexF64}(I, nwan, nwan)
        Upart[1:2, 1:2] = R1
        for k in ks
            @test blochmatrix(wanPart, k) ≈ Upart' * blochmatrix(wan, k) * Upart
        end

        @test_throws ErrorException gaugetransform(wan, siteindices, Dict(1 => R1, 5 => R2))
        @test_throws ErrorException gaugetransform(wan, Dict(1 => [1, 2], 2 => [3, 3]), rotations)
        @test_throws ErrorException gaugetransform(wan, Dict(1 => [1, 2], 2 => [3, 5]), rotations)
        @test_throws ErrorException gaugetransform(wan, Dict(1 => [1, 2], 2 => [3]), rotations)
        @test_throws ErrorException gaugetransform(wan, Dict(1 => Set([1, 2]), 2 => [3, 4]), rotations)
        @test_throws ErrorException gaugetransform(wan, siteindices, Dict(1 => rand(3, 3), 2 => R2))
        @test_throws ErrorException gaugetransform(wan, siteindices, Dict(1 => 1.0im, 2 => R2))
    end

    # ==================================================================
    @testset "W90Bridge" begin
        # the Hilbert space must be spinful: the spinless Wannier data are
        # duplicated onto both spin sectors, i.e. H = H_W90 ⊗ I₂
        hilbert = Hilbert(pid => Fock{:f}(norb, 2) for pid in 1:nsites)
        bridge = W90Bridge(wan, hilbert)
        @test bridge.siteindices == Dict(1 => [1, 2], 2 => [3, 4])
        @test bridge.w90index == [(1, 1), (1, 2), (2, 1), (2, 2)]

        shells = w90shells(bridge; n = 1)
        @test ((1, 0, 0), [1]) in shells
        @test ((-1, 0, 0), [1]) in shells
        coverage = w90coverage(bridge, bonds(lattice, 1))
        @test coverage.n_missing == 0

        # the Table order is (site, orbital, spin), spin fastest, so the Wannier
        # index w of a spin-up entry maps to 2w-1 and of a spin-down entry to 2w
        H0, H1 = w90generators(bridge; neighbors = 1)
        lookup = Dict(bridge.w90index[w] => w for w in eachindex(bridge.w90index))
        keytowannier = key -> 2 * lookup[(key[1], key[2])] - (key[3] == 1//2 ? 1 : 0)
        for k in ks
            @test orderedmatrix(hilbert, (H0, H1), k, keytowannier) ≈
                  kron(blochmatrix(wan, k), Matrix{ComplexF64}(I, 2, 2))
        end

        # a spinless Hilbert space is not supported
        hilbert0 = Hilbert(pid => Fock{:f}(norb, 1) for pid in 1:nsites)
        @test_throws "the Hilbert space must be spinful" W90Bridge(wan, hilbert0)
        # `siteindices` must use each Wannier index exactly once ...
        @test_throws "must contain each Wannier index" W90Bridge(
            wan, hilbert; siteindices = Dict(1 => [1, 1], 2 => [3, 4])
        )
        # ... and must give one Wannier index per orbital of the site
        @test_throws "but 1 Wannier indices were given" W90Bridge(
            wan, hilbert; siteindices = Dict(1 => [1], 2 => [3, 4])
        )

        # rotate through the W90 API, then rebuild the bridge from the rotated
        # data: a bridge is a derived mapping and is not transformed itself
        rotations = Dict(1 => R1, 2 => R2)
        Ublock = zeros(ComplexF64, nwan, nwan)
        Ublock[1:2, 1:2] = R1
        Ublock[3:4, 3:4] = R2
        wanB = gaugetransform(wan, bridge.siteindices, rotations)
        for k in ks
            @test blochmatrix(wanB, k) ≈ Ublock' * blochmatrix(wan, k) * Ublock
        end
        @test blochmatrix(gaugetransform(wan, Ublock), ks[2]) ≈
              Ublock' * blochmatrix(wan, ks[2]) * Ublock
        @test blochmatrix(gaugetransform(wan, bridge.siteindices, site -> rotations[site]), ks[2]) ≈
              blochmatrix(wanB, ks[2])

        # the rebuilt bridge reproduces the rotated Hamiltonian
        bridgeB = W90Bridge(wanB, hilbert)
        @test bridgeB.siteindices == bridge.siteindices
        H0b, H1b = w90generators(bridgeB; neighbors = 1)
        for k in ks
            @test orderedmatrix(hilbert, (H0b, H1b), k, nothing) ≈
                  kron(blochmatrix(wanB, k), Matrix{ComplexF64}(I, 2, 2))
        end
    end

    # ==================================================================
    @testset "W90SpinorBridge" begin
        nwanS = 2 * norb * nsites
        centersS = zeros(Float64, 3, nwanS)
        for site in 1:nsites, orbital in 1:norb, slot in 1:2
            w = 4 * (site - 1) + 2 * (orbital - 1) + slot
            centersS[:, w] = [site - 1.0, 0.0, 0.0]
        end

        state = Ref{UInt64}(12345)
        nextvalue() = begin
            state[] = state[] * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            return Float64(state[] % UInt64(10000)) / 10000.0
        end
        randblock(scale) = ComplexF64[
            scale * (0.1 * (nextvalue() - 0.5) + 0.1im * (nextvalue() - 0.5)) for _ in 1:nwanS, _ in 1:nwanS
        ]
        S0 = randblock(1.0)
        S0 = (S0 + S0') / 2
        for i in 1:nwanS
            S0[i, i] += 0.5 + 0.1im
        end
        S1 = randblock(0.3)             # only site 2 <- site 1
        S1[1:4, :] .= 0
        S1[:, 5:8] .= 0

        hamS = OperatorSum{W90Hoppings}()
        hamS.contents[(0, 0, 0)] = W90Hoppings(S0, (0, 0, 0))
        hamS.contents[(1, 0, 0)] = W90Hoppings(S1, (1, 0, 0))
        hamS.contents[(-1, 0, 0)] = W90Hoppings(S1', (-1, 0, 0))
        wanS = W90(lattice, centersS, StaticGenerator(hamS))

        hilbertS = Hilbert(pid => Fock{:f}(norb, 2) for pid in 1:nsites)
        bridgeS = W90SpinorBridge(wanS, hilbertS)
        @test bridgeS.siteindices == Dict(1 => [1, 2, 3, 4], 2 => [5, 6, 7, 8])
        @test bridgeS.w90index[[1, 2, 3, 4]] ==
              [(1, 1, 1//2), (1, 1, -1//2), (1, 2, 1//2), (1, 2, -1//2)]
        @test w90coverage(bridgeS, bonds(lattice, 1)).n_missing == 0
        @test ((1, 0, 0), [1]) in w90shells(bridgeS; n = 1)

        keytowannier = key -> Dict(bridgeS.w90index[w] => w for w in 1:nwanS)[key]
        H0S, H1S = w90spinorgenerators(bridgeS; neighbors = 1)
        for k in ks
            @test orderedmatrix(hilbertS, (H0S, H1S), k, keytowannier) ≈ blochmatrix(wanS, k)
        end
        # spinor bands are not duplicated
        @test length(unique(round.(real.(spectrum(blochmatrix(wanS, ks[2]))); digits = 8))) == nwanS

        # slot-preserving rotation: orbital mixing without spin mixing
        Rsp = ComplexF64[cos(θ) -sin(θ); sin(θ) cos(θ)]
        Rsite = kron(Rsp, Matrix{ComplexF64}(I, 2, 2))
        rotationsS = Dict(site => Rsite for site in keys(bridgeS.siteindices))
        wanS2 = gaugetransform(wanS, bridgeS.siteindices, rotationsS)
        US = zeros(ComplexF64, nwanS, nwanS)
        for (_, indices) in bridgeS.siteindices
            US[indices, indices] = Rsite
        end
        for k in ks
            @test blochmatrix(wanS2, k) ≈ US' * blochmatrix(wanS, k) * US
        end
        bridgeS2 = W90SpinorBridge(wanS2, hilbertS)
        @test bridgeS2.siteindices == bridgeS.siteindices
        @test [entry[3] for entry in bridgeS2.w90index] == [entry[3] for entry in bridgeS.w90index]
        H0S2, H1S2 = w90spinorgenerators(bridgeS2; neighbors = 1)
        for k in ks
            @test orderedmatrix(hilbertS, (H0S2, H1S2), k, keytowannier) ≈ blochmatrix(wanS2, k)
        end

        # a spin-mixing rotation keeps the matrix relation but invalidates the spin labels
        Rmix = ComplexF64[0 1 0 0; 1 0 0 0; 0 0 1 0; 0 0 0 1]
        wanS3 = gaugetransform(wanS, bridgeS.siteindices, Dict(1 => Rmix))
        U3 = Matrix{ComplexF64}(I, nwanS, nwanS)
        U3[1:4, 1:4] = Rmix
        for k in ks
            @test blochmatrix(wanS3, k) ≈ U3' * blochmatrix(wanS, k) * U3
        end

        @test_throws ErrorException W90SpinorBridge(wanS, Hilbert(pid => Fock{:f}(norb, 1) for pid in 1:nsites))
        @test_throws ErrorException W90SpinorBridge(wanS, hilbertS;
                                                    siteindices = Dict(1 => [1, 2, 3, 4], 2 => [5, 6, 7, 7]))
        @test_throws ErrorException W90SpinorBridge(wanS, Hilbert(pid => Fock{:f}(2 * norb, 2) for pid in 1:nsites))
    end
end
