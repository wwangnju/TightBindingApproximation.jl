"""
    fermi_dirac(ε::Real, μ::Real, T::Real) -> Float64

Fermi-Dirac distribution function.

Returns `f(ε) = 1 / (1 + exp((ε - μ) / T))` for `T > 0`, and the step function
`f(ε) = (ε <= μ ? 1.0 : 0.0)` for `T = 0`.

Numerically stable: clips `(ε-μ)/T` to ±50 to avoid overflow.
"""
function fermi_dirac(ε::Real, μ::Real, T::Real)
    if iszero(T)
        return ε <= μ ? 1.0 : 0.0
    end
    x = (ε - μ) / T
    if x > 50
        return 0.0
    elseif x < -50
        return 1.0
    else
        return 1.0 / (1.0 + exp(x))
    end
end

"""
    occupation(dos::DensityOfStatesData, μ::Real, T::Real=0.0) -> Vector{Float64}

Compute orbital-projected occupation numbers from density of states data via trapezoidal integration:

    n = ∫ DOS(ε) × f_FD(ε; μ, T) dε

Since `DensityOfStatesData.values` already represent bin-integrated spectral weights
(they contain the `dE` factor), the trapezoidal rule on the uniform energy grid is

    n ≈ DOS_values[1]×f₁/2 + Σ_{i=2}^{n-1} DOS_values[i]×f_i + DOS_values[n]×fₙ/2.

Returns a vector of length `size(dos.values, 2)`, one element per orbital projection column.
"""
function occupation(dos::DensityOfStatesData, μ::Real, T::Real=0.0)
    μ, T = Float64(μ), Float64(T)
    ε = dos.energies
    n = length(ε)
    n_orbitals = size(dos.values, 2)
    result = zeros(Float64, n_orbitals)
    for io in 1:n_orbitals
        s = fermi_dirac(ε[begin], μ, T) * dos.values[begin, io] / 2
        for i in 2:n-1
            s += fermi_dirac(ε[i], μ, T) * dos.values[i, io]
        end
        s += fermi_dirac(ε[end], μ, T) * dos.values[end, io] / 2
        result[io] = s
    end
    return result
end

"""
    chemical_potential(dos::DensityOfStatesData, target_n::Real; T::Real=0.0, tol::Real=1e-10) -> Float64

Find the chemical potential μ such that the total electron number matches `target_n`.

Uses bisection on `n(μ) = ∫ total_DOS(ε) × f_FD(ε; μ, T) dε` with trapezoidal integration
(same convention as [`occupation`](@ref)). Since `n(μ)` is monotonically increasing, bisection
is robust and converges in ~50 iterations.
"""
function chemical_potential(dos::DensityOfStatesData, target_n::Real; T::Real=0.0, tol::Real=1e-10)
    T = Float64(T)
    ε = dos.energies
    n = length(ε)
    total_dos = size(dos.values, 2) == 1 ? vec(dos.values) : vec(sum(dos.values, dims=2))

    # Bracket: n(μ_low) ≈ 0, n(μ_high) ≈ total number of states
    μ_low = ε[1] - 10 * (ε[end] - ε[1] + 1)
    μ_high = ε[end] + 10 * (ε[end] - ε[1] + 1)

    # Bisection with trapezoidal integration
    while μ_high - μ_low > tol
        μ_mid = (μ_low + μ_high) / 2
        n_mid = fermi_dirac(ε[begin], μ_mid, T) * total_dos[begin] / 2
        for i in 2:n-1
            n_mid += fermi_dirac(ε[i], μ_mid, T) * total_dos[i]
        end
        n_mid += fermi_dirac(ε[end], μ_mid, T) * total_dos[end] / 2
        if n_mid < target_n
            μ_low = μ_mid
        else
            μ_high = μ_mid
        end
    end
    return (μ_low + μ_high) / 2
end
