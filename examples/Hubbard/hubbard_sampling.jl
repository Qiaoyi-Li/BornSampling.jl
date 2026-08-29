import Bornsampling
using Bornsampling.FiniteMPS
using Bornsampling.Random
using CairoMakie
using FiniteLattices
using Statistics

const TK = Bornsampling.TK

# Model and DMRG parameters.
const L = 8
const W = 4
const U = 12.0
const HOLE_DOPING = 1 // 8
const D = 1024
const ENERGY_TOLERANCE = 1e-5
const MAX_SWEEPS = 100

# Sampling parameters.
const Ns = 1000
const SAMPLING_TASKS = 4
const RANDOM_SEED = 1234

const NSITES = L * W
const NPARTICLES = Int((1 - HOLE_DOPING) * NSITES)

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
    end
    return AutomataMPO(tree)
end

function initial_fixed_charge_state(nsites::Int, nparticles::Int)
    occupations = zeros(Int, nsites)
    for _ in 1:nparticles
        least_occupied = findall(==(minimum(occupations)), occupations)
        occupations[rand(least_occupied)] += 1
    end

    # Charge labels of the horizontal bonds, following the FiniteMPS tutorial.
    bond_charges = [0]
    for occupation in reverse(occupations)
        increment = iszero(occupation) ? 1 : occupation == 1 ? 0 : -1
        push!(bond_charges, bond_charges[end] + increment)
    end
    bond_charges = reverse(bond_charges[2:end])

    bond_spaces = map(1:nsites) do i
        if i == 1
            total_spin = iseven(nparticles) ? 0 : 1 // 2
            return Rep[U₁×SU₂]((bond_charges[1], total_spin) => 1)
        end
        return Rep[U₁×SU₂](
            (bond_charges[i], spin) => 1 for spin in 0:1//2:1//2
        )
    end

    state = randMPS(fill(U1SU2Fermion.pspace, nsites), bond_spaces)
    return state
end

function optimize_ground_state!(state, hamiltonian)
    environment = Environment(state', hamiltonian, state)
    energies = Float64[]

    for sweep in 1:MAX_SWEEPS
        info, _ = DMRGSweep1!(
            environment;
            K=16,
            trunc=truncdim(D) & truncbelow(1e-12),
            CBEAlg=NaiveCBE(
                D + div(D, 4),
                1e-8;
                rsvd=true,
            ),
            GCsweep=true,
        )
        energy = real(info[2].dmrg[1].Eg)
        push!(energies, energy)

        energy_change = length(energies) == 1 ? Inf :
                        abs(energies[end] - energies[end - 1]) / NSITES
        @info "DMRG sweep" sweep D energy energy_change

        if length(energies) >= 2 && energy_change < ENERGY_TOLERANCE
            return energies
        end
    end

    @warn "DMRG reached MAX_SWEEPS before energy convergence" D
    return energies
end

function exact_szsz(state, ntasks::Int)
    nsites = length(state)
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
    calobs_seconds = @elapsed calObs!(
        tree,
        state;
        normalize=true,
        ntasks=ntasks,
    )
    observables = convert(Dict, tree)

    correlation = Matrix{Float64}(undef, nsites, nsites)
    for i in 1:nsites, j in i:nsites
        value = real(observables["SS"][(i, j)]) / 3
        correlation[i, j] = value
        correlation[j, i] = value
    end
    return correlation, calobs_seconds
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

    # Accumulate all per-shot outer products with two reusable
    # nsites×nsites matrices.
    sum_products = sz * transpose(sz)
    estimate = sum_products / nshots

    # (SᵢᶻSⱼᶻ)² = (Sᵢᶻ)²(Sⱼᶻ)² for each shot s.
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
        # Squared distances are integers on this YC square cylinder.
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

    # Average same-distance pairs inside each shot first. Each shot is one
    # independent statistical observation.
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
    output_path = joinpath(output_directory, "hubbard_szsz_vs_distance.png")
    save(output_path, figure)
    @info "saved correlation comparison" output_path
    return nothing
end

function main()
    Ns >= 2 || throw(ArgumentError("Ns must be at least two to estimate a standard error"))
    iseven(NPARTICLES) || error("this example expects an SU(2)-singlet sector")

    Random.seed!(RANDOM_SEED)
    lattice = YCSqua(L, W) |> Snake!
    hamiltonian = hubbard_hamiltonian(lattice)
    state = initial_fixed_charge_state(NSITES, NPARTICLES)

    local energies
    dmrg_seconds = @elapsed energies = optimize_ground_state!(state, hamiltonian)

    exact_matrix, calobs_seconds = exact_szsz(state, SAMPLING_TASKS)

    # All observables have been measured, so the sampler may canonicalize this
    # state in place while memory holds one D=1024 state.
    local sampler
    sampler_compilation_seconds =
        @elapsed sampler = Bornsampling.BornSampler(state)
    local configurations
    sampling_seconds = @elapsed begin
        configurations = Bornsampling.bornsample!(
            MersenneTwister(RANDOM_SEED),
            sampler,
            Ns;
            ntasks=SAMPLING_TASKS,
        ).configuration
    end
    timings = (
        dmrg_seconds=dmrg_seconds,
        calobs_seconds=calobs_seconds,
        sampler_compilation_seconds=sampler_compilation_seconds,
        sampling_seconds=sampling_seconds,
    )
    @info "phase timings" timings
    @info "correlation costs" calObs_total_seconds=calobs_seconds sampling_total_seconds=sampling_seconds sampling_seconds_per_sample=sampling_seconds / Ns nshots=Ns

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
        energies=energies,
        timings=timings,
        exact_szsz=exact_matrix,
        sampled_szsz=sampled_matrix,
        sampled_szsz_standard_error=sampled_matrix_se,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
