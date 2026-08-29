import Bornsampling
using Bornsampling.FiniteMPS
using Bornsampling.Random
using CairoMakie
using FiniteLattices
using Statistics

const TK = Bornsampling.TK

# Model and tanTRG parameters.
const L = 8
const W = 4
const U = 8.0
const MU = 4.0
const D = 1024
const BETA_GRID = 2.0 .^ (-15:0)

# Sampling parameters.
const Ns = 100
const SAMPLING_TASKS = 4
const RANDOM_SEED = 1234

function hubbard_hamiltonian(lattice)
    tree = InteractionTree(size(lattice))

    # `ordered=true` supplies both hopping directions.
    for (i, j) in neighbor(lattice; ordered=true)
        addIntr!(
            tree,
            U1SU2Fermion.FdagF,
            (i, j),
            (true, true),
            -1.0;
            name=(:Fdag, :F),
            Z=U1SU2Fermion.Z,
        )
    end
    for i in 1:size(lattice)
        addIntr!(tree, U1SU2Fermion.nd, i, U; name=:nd)
        addIntr!(tree, U1SU2Fermion.n, i, -MU; name=:n)
    end
    return AutomataMPO(tree)
end

function thermal_factor(hamiltonian)
    beta = first(BETA_GRID)
    factor, _ = SETTN(
        hamiltonian,
        beta;
        CBEAlg=NaiveCBE(
            D + div(D, 4),
            1e-8;
            rsvd=true,
        ),
        trunc=truncdim(D) & truncbelow(1e-16),
        maxorder=4,
        maxiter=6,
        tol=1e-12,
        lsnoise=[(1 / 4, noise) for noise in (0.01, 0.001)],
        GCsweep=true,
        verbose=1,
    )
    normalize!(factor)

    environment = Environment(factor', hamiltonian, factor)
    for index in 2:length(BETA_GRID)
        next_beta = BETA_GRID[index]
        delta_beta = next_beta - beta
        TDVPSweep1!(
            environment,
            -delta_beta / 2;
            CBEAlg=NaiveCBE(
                D + div(D, 4),
                1e-8;
                rsvd=true,
            ),
            trunc=truncdim(D) & truncbelow(1e-12),
            GCsweep=true,
            verbose=1,
        )
        normalize!(factor)
        beta = next_beta
        @info "tanTRG step" beta delta_beta D
    end

    return factor
end

function szsz_observable_tree(nsites::Int)
    tree = ObservableTree(nsites)

    # Sᵢ⋅Sⱼ is symmetric under i↔j, so only the upper triangle is added.
    for i in 1:nsites, j in i:nsites
        addObs!(
            tree,
            U1SU2Fermion.SS,
            (i, j),
            (false, false);
            name=(:S, :S),
        )
    end
    return tree
end

function szsz_matrix(tree, nsites::Int)
    observables = convert(Dict, tree)

    correlation = Matrix{Float64}(undef, nsites, nsites)
    for i in 1:nsites, j in i:nsites
        value = real(observables["SS"][(i, j)]) / 3
        correlation[i, j] = value
        correlation[j, i] = value
    end
    return correlation
end

function u1su2_sz_lookup(space)
    values = zeros(Float64, Int(TK.dim(space)))
    for sector in TK.sectors(space)
        irrep_dimension = Int(TK.dim(sector))
        multiplicity = Int(TK.dim(space, sector))
        sector_range = TK.axes(space, sector)
        for degeneracy in 1:multiplicity, irrep in 1:irrep_dimension
            flat = first(sector_range) - 1 + irrep +
                   (degeneracy - 1) * irrep_dimension

            # TensorKit orders an SU(2) carrier from m=j down to m=-j.
            values[flat] = (irrep_dimension + 1 - 2 * irrep) / 2
        end
    end
    return values
end

function sample_szsz_statistics(sz)
    nshots = size(sz, 2)

    sum_products = sz * transpose(sz)
    estimate = sum_products / nshots

    sz_squared = abs2.(sz)
    sum_squared_products = sz_squared * transpose(sz_squared)
    sample_variance = (
        sum_squared_products .- nshots .* abs2.(estimate)
    ) / (nshots - 1)
    standard_error = sqrt.(sample_variance ./ nshots)

    return estimate, standard_error
end

function distance_bins(lattice)
    pairs_by_squared_distance = Dict{Int,Vector{NTuple{2,Int}}}()
    for i in 1:size(lattice), j in i:size(lattice)
        squared_distance = round(Int, distance(lattice, i, j)^2)
        pairs = get!(pairs_by_squared_distance, squared_distance) do
            NTuple{2,Int}[]
        end
        push!(pairs, (i, j))
    end

    squared_distances = sort!(collect(keys(pairs_by_squared_distance)))
    radii = sqrt.(Float64.(squared_distances))
    pair_bins = [pairs_by_squared_distance[r2] for r2 in squared_distances]
    return radii, pair_bins
end

function radial_exact(correlation, pair_bins)
    return [mean(correlation[i, j] for (i, j) in pairs) for pairs in pair_bins]
end

function radial_sample_standard_error(sz, pair_bins)
    nshots = size(sz, 2)
    per_shot = Matrix{Float64}(undef, length(pair_bins), nshots)

    # Same-distance pairs are averaged inside each independent shot first.
    for (bin, pairs) in enumerate(pair_bins), shot in 1:nshots
        value = zero(Float64)
        for (i, j) in pairs
            value += sz[i, shot] * sz[j, shot]
        end
        per_shot[bin, shot] = value / length(pairs)
    end

    return vec(std(per_shot; dims=2, corrected=true)) / sqrt(nshots)
end

function plot_correlations(radii, exact, sampled, standard_error)
    figure = Figure(size=(680, 420))
    axis = Axis(
        figure[1, 1];
        xlabel="distance r",
        ylabel="distance-averaged ⟨Sᵢᶻ Sⱼᶻ⟩",
    )
    exact_plot = scatterlines!(
        axis,
        radii,
        exact;
        linewidth=2,
    )
    lines!(
        axis,
        radii,
        sampled;
        color=:darkorange,
        linewidth=2,
    )
    errorbars!(
        axis,
        radii,
        sampled,
        standard_error;
        color=:darkorange,
        whiskerwidth=8,
    )

    sampled_legend = [
        LineElement(
            color=:darkorange,
            linewidth=2,
            points=[Point2f(0, 0.5), Point2f(1, 0.5)],
        ),
        LineElement(
            color=:darkorange,
            points=[Point2f(0.5, 0.15), Point2f(0.5, 0.85)],
        ),
        LineElement(
            color=:darkorange,
            points=[Point2f(0.35, 0.15), Point2f(0.65, 0.15)],
        ),
        LineElement(
            color=:darkorange,
            points=[Point2f(0.35, 0.85), Point2f(0.65, 0.85)],
        ),
    ]
    axislegend(
        axis,
        [exact_plot, sampled_legend],
        ["Expectation", "Sample mean"],
    )

    output_directory = joinpath(@__DIR__, "output")
    mkpath(output_directory)
    output_path = joinpath(
        output_directory,
        "hubbard_tantrg_beta1_szsz_vs_distance.png",
    )
    save(output_path, figure)
    @info "saved correlation comparison" output_path
    return nothing
end

function main()
    Ns >= 2 || throw(ArgumentError(
        "Ns must be at least two to estimate a standard error",
    ))
    last(BETA_GRID) == 1.0 || error("this example must evolve to beta=1")
    MU == U / 2 || error("half filling requires the hard-coded mu=U/2")

    Random.seed!(RANDOM_SEED)
    lattice = YCSqua(L, W) |> Snake!
    hamiltonian = hubbard_hamiltonian(lattice)

    local factor
    tantrg_seconds = @elapsed factor = thermal_factor(hamiltonian)
    @info "tanTRG cost" tanTRG_total_seconds=tantrg_seconds beta=last(BETA_GRID) D

    observable_tree = szsz_observable_tree(length(factor))
    calobs_seconds = @elapsed calObs!(
        observable_tree,
        factor;
        normalize=true,
        ntasks=SAMPLING_TASKS,
    )
    exact_matrix = szsz_matrix(observable_tree, length(factor))

    # All observables have been measured. BornSampler may now canonicalize the
    # thermal factor in place for the final sampling phase.
    local sampler
    sampler_compilation_seconds =
        @elapsed sampler = Bornsampling.BornSampler(factor)

    local configurations
    sampling_seconds = @elapsed begin
        configurations = Bornsampling.bornsample!(
            MersenneTwister(RANDOM_SEED),
            sampler,
            Ns;
            ntasks=SAMPLING_TASKS,
        ).configuration
    end
    @info "correlation costs" calObs_total_seconds=calobs_seconds sampling_total_seconds=sampling_seconds sampling_seconds_per_sample=sampling_seconds / Ns nshots=Ns
    timings = (
        tantrg_seconds=tantrg_seconds,
        calobs_seconds=calobs_seconds,
        sampler_compilation_seconds=sampler_compilation_seconds,
        sampling_seconds=sampling_seconds,
    )

    sz_lookup = u1su2_sz_lookup(U1SU2Fermion.pspace)
    sz = sz_lookup[configurations]
    sampled_matrix, sampled_matrix_se = sample_szsz_statistics(sz)

    radii, pair_bins = distance_bins(lattice)
    exact_radial = radial_exact(exact_matrix, pair_bins)
    sampled_radial = radial_exact(sampled_matrix, pair_bins)
    sampled_radial_se = radial_sample_standard_error(sz, pair_bins)
    plot_correlations(
        radii,
        exact_radial,
        sampled_radial,
        sampled_radial_se,
    )

    return (
        timings=timings,
        nshots=Ns,
        exact_szsz=exact_matrix,
        sampled_szsz=sampled_matrix,
        sampled_szsz_standard_error=sampled_matrix_se,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
