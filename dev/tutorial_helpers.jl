module BornsamplingTutorialHelpers

import Bornsampling
using Bornsampling.FiniteMPS
using Bornsampling.Random
using CairoMakie
using Statistics

const TK = Bornsampling.TK

export direct_szsz_by_distance
export hubbard_hamiltonian
export hubbard_sz_lookup
export plot_szsz_comparison
export prepare_ground_state
export prepare_thermal_factor
export sampled_szsz_by_distance

function hubbard_hamiltonian(nsites; U, mu)
    tree = InteractionTree(nsites)
    for i in 1:(nsites - 1)
        for (left, right) in ((i, i + 1), (i + 1, i))
            addIntr!(
                tree,
                U1SU2Fermion.FdagF,
                (left, right),
                (true, true),
                -1.0;
                name=(:Fdag, :F),
                Z=U1SU2Fermion.Z,
            )
        end
    end
    for i in 1:nsites
        addIntr!(tree, U1SU2Fermion.nd, i, U; name=:nd)
        addIntr!(tree, U1SU2Fermion.n, i, -mu; name=:n)
    end
    return AutomataMPO(tree)
end

function direct_szsz_by_distance(state; ntasks::Int)
    nsites = length(state)
    observable_tree = ObservableTree(nsites)
    for i in 1:nsites, j in i:nsites
        addObs!(
            observable_tree,
            U1SU2Fermion.SS,
            (i, j),
            (false, false);
            name=(:S, :S),
        )
    end
    calObs!(observable_tree, state; normalize=true, ntasks=ntasks)
    observables = convert(Dict, observable_tree)

    # Spin-rotation invariance gives SzSz = (S⋅S)/3.
    return [
        mean(
            real(observables["SS"][(i, i + r)]) / 3
            for i in 1:(nsites - r)
        )
        for r in 0:(nsites - 1)
    ]
end

function hubbard_sz_lookup()
    space = U1SU2Fermion.pspace
    values = zeros(Float64, Int(TK.dim(space)))
    for sector in TK.sectors(space)
        irrep_dimension = Int(TK.dim(sector))
        multiplicity = Int(TK.dim(space, sector))
        sector_range = TK.axes(space, sector)
        for degeneracy in 1:multiplicity, irrep in 1:irrep_dimension
            flat = first(sector_range) - 1 + irrep +
                   (degeneracy - 1) * irrep_dimension
            values[flat] = (irrep_dimension + 1 - 2 * irrep) / 2
        end
    end
    return values
end

function sampled_szsz_by_distance(configuration, sz_lookup)
    sz = sz_lookup[configuration]
    nsites, nshots = size(sz)
    estimate = Vector{Float64}(undef, nsites)
    standard_error = Vector{Float64}(undef, nsites)
    radial_per_shot = Vector{Float64}(undef, nshots)

    for r in 0:(nsites - 1)
        for shot in 1:nshots
            radial_per_shot[shot] = mean(
                sz[i, shot] * sz[i + r, shot]
                for i in 1:(nsites - r)
            )
        end
        estimate[r + 1] = mean(radial_per_shot)
        standard_error[r + 1] =
            std(radial_per_shot; corrected=true) / sqrt(nshots)
    end
    return estimate, standard_error
end

function _line_with_errorbar(color)
    return [
        LineElement(
            color=color,
            linewidth=2,
            points=[Point2f(0, 0.5), Point2f(1, 0.5)],
        ),
        LineElement(
            color=color,
            points=[Point2f(0.5, 0.15), Point2f(0.5, 0.85)],
        ),
        LineElement(
            color=color,
            points=[Point2f(0.35, 0.15), Point2f(0.65, 0.15)],
        ),
        LineElement(
            color=color,
            points=[Point2f(0.35, 0.85), Point2f(0.65, 0.85)],
        ),
    ]
end

function plot_szsz_comparison(
    direct,
    sampled_series;
    direct_label,
    filename,
)
    distances = 0:(length(direct) - 1)
    figure = Figure(size=(720, 440))
    axis = Axis(
        figure[1, 1];
        xlabel="distance r",
        ylabel="distance-averaged ⟨Sᵢᶻ Sⱼᶻ⟩",
        xticks=distances,
    )

    direct_plot = scatterlines!(
        axis,
        distances,
        direct;
        color=:black,
        linewidth=2,
    )
    legend_entries = Any[direct_plot]
    legend_labels = [direct_label]
    for series in sampled_series
        lines!(
            axis,
            distances,
            series.mean;
            color=series.color,
            linewidth=2,
        )
        errorbars!(
            axis,
            distances,
            series.mean,
            series.standard_error;
            color=series.color,
            whiskerwidth=8,
        )
        push!(legend_entries, _line_with_errorbar(series.color))
        push!(legend_labels, series.label)
    end

    lower, upper = extrema(direct)
    for series in sampled_series, index in eachindex(series.mean)
        lower = min(lower, series.mean[index] - series.standard_error[index])
        upper = max(upper, series.mean[index] + series.standard_error[index])
    end
    margin = max(0.05 * (upper - lower), eps(Float64))
    ylims!(axis, lower - margin, upper + margin)
    axislegend(axis, legend_entries, legend_labels)

    mkpath("figures")
    save(joinpath("figures", filename), figure)
    return nothing
end

function _optimize_ground_state!(environment; D, nsites)
    energy_tolerance = 1e-5
    maximum_sweeps = 12
    previous_energy = Inf
    final_energy = Inf

    for sweep in 1:maximum_sweeps
        info, _ = DMRGSweep1!(
            environment;
            K=16,
            trunc=truncdim(D) & truncbelow(1e-12),
            CBEAlg=NaiveCBE(D + div(D, 4), 1e-8; rsvd=true),
            GCsweep=true,
        )
        final_energy = real(info[2].dmrg[1].Eg)

        if isfinite(previous_energy)
            energy_change = abs(final_energy - previous_energy) / nsites
            energy_change < energy_tolerance && return (;
                sweeps=sweep,
                final_energy,
            )
        end
        previous_energy = final_energy
    end
    return (; sweeps=maximum_sweeps, final_energy)
end

function prepare_ground_state(hamiltonian; nsites, D, seed)
    Random.seed!(seed)
    left_space = Rep[U₁×SU₂]((0, 0) => 1)
    bulk_space = Rep[U₁×SU₂]((0, spin) => 1 for spin in 0:1//2:1//2)
    bond_spaces = vcat(left_space, fill(bulk_space, nsites - 1))
    state = randMPS(fill(U1SU2Fermion.pspace, nsites), bond_spaces)
    environment = Environment(state', hamiltonian, state)
    result = _optimize_ground_state!(environment; D, nsites)
    return state, result
end

function prepare_thermal_factor(hamiltonian; D)
    beta_grid = 2.0 .^ (-15:0)
    beta = first(beta_grid)
    factor, _ = SETTN(
        hamiltonian,
        beta;
        CBEAlg=NaiveCBE(D + div(D, 4), 1e-8; rsvd=true),
        trunc=truncdim(D) & truncbelow(1e-16),
        maxorder=4,
        maxiter=6,
        tol=1e-12,
        lsnoise=[(1 / 4, noise) for noise in (0.01, 0.001)],
        GCsweep=true,
    )
    normalize!(factor)

    environment = Environment(factor', hamiltonian, factor)
    for next_beta in beta_grid[2:end]
        delta_beta = next_beta - beta
        TDVPSweep1!(
            environment,
            -delta_beta / 2;
            CBEAlg=NaiveCBE(D + div(D, 4), 1e-8; rsvd=true),
            trunc=truncdim(D) & truncbelow(1e-12),
            GCsweep=true,
        )
        normalize!(factor)
        beta = next_beta
    end
    return factor, beta
end

end
