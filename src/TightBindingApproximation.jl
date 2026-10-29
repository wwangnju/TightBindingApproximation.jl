module TightBindingApproximation

# Core of tight binding approximation
include("Core.jl")
export Bosonic, Fermionic, Phononic, TBAKind
export CompositeTBA, Quadratic, Quadraticization, SimpleTBA, TBA, TBAMatrix, TBAMatrixization, berrycurvature, commutator, eigen, eigvals, eigvecs, infinitesimal
export BerryCurvature, BerryCurvatureData, DensityOfStates, DensityOfStatesData, EnergyBands, EnergyBandsData, FermiSurface, FermiSurfaceData, InelasticNeutronScatteringSpectra, InelasticNeutronScatteringSpectraData
export BerryCurvatureMethod, Fukui, Kubo

# Fitting parameters
include("Fitting.jl")

# Integration with Wannier90
include("Wannier90.jl")

# Utilities (fermi_dirac, occupation, chemical_potential, etc.)
include("Utilities.jl")
export chemical_potential, fermi_dirac, occupation

# Bridge between the Wannier90 data and the operator generators used by
# SecondOrderPerturbationTheory (`Wannier90SOPT.W90Bridge`, ...)
include("Wannier90SOPT.jl")

end # module
