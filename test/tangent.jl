function compatible_base_state(base, ::FiniteMPS.MPS)
    tensors = vcat([deepcopy(base.A[1])], deepcopy.(base.Ar[2:end]))
    return FiniteMPS.MPS(tensors)
end

function compatible_base_state(base, ::FiniteMPS.MPO)
    tensors = vcat([deepcopy(base.A[1])], deepcopy.(base.Ar[2:end]))
    return FiniteMPS.MPO(tensors)
end

function tangent_with_persistent_symmetry(base, symmetry; seed=0x7173_796d)
    rng = MersenneTwister(seed)
    tensors = FiniteMPS.MPSTensor[]
    for tensor in base.A
        codomain = ⊗(BS.leftspace(tensor), BS.physspace(tensor))
        domain = if BS._tensor_rank(tensor) == 3
            ⊗(symmetry, BS.rightspace(tensor))
        else
            ⊗(BS.purspace(tensor), symmetry, BS.rightspace(tensor))
        end
        push!(
            tensors,
            FiniteMPS.MPSTensor(TK.randn(rng, ComplexF64, codomain, domain)),
        )
    end
    return FiniteMPSTangents.TangentMPS{length(tensors)}(base, tensors)
end

function tangent_with_random_insertions(base; seed=0x7461_6e62)
    rng = MersenneTwister(seed)
    tensors = FiniteMPS.MPSTensor[
        FiniteMPS.MPSTensor(TK.randn(
            rng, ComplexF64, TK.codomain(tensor.A), TK.domain(tensor.A),
        )) for tensor in base.A
    ]
    return FiniteMPSTangents.TangentMPS{length(tensors)}(base, tensors)
end

"Three spins with a two-dimensional spin-half boundary and two internal fusion channels."
function tangent_nonabelian_base_state()
    physical = FiniteMPS.SU2Spin.pspace
    left = TK.Rep[TK.SU₂](1 // 2 => 1)
    first_bond = TK.Rep[TK.SU₂](0 => 1, 1 => 1)
    second_bond = TK.Rep[TK.SU₂](1 // 2 => 1)
    right = TK.Rep[TK.SU₂](0 => 1)
    rng = MersenneTwister(0x746e_6162)
    state = FiniteMPS.MPS(FiniteMPS.MPSTensor.([
        TK.randn(rng, ComplexF64, ⊗(left, physical), first_bond),
        TK.randn(rng, ComplexF64, ⊗(first_bond, physical), second_bond),
        TK.randn(rng, ComplexF64, ⊗(second_bond, physical), right),
    ]))
    FiniteMPS.canonicalize!(state, 1)
    return state
end

"Two SU(2) purification sites with singlet and triplet internal fusion channels."
function tangent_nonabelian_operator_base_state()
    physical = FiniteMPS.SU2Spin.pspace
    boundary = TK.Rep[TK.SU₂](0 => 1)
    virtual = TK.Rep[TK.SU₂](0 => 1, 1 => 1)
    rng = MersenneTwister(0x7479_7375)
    state = FiniteMPS.MPO(FiniteMPS.MPSTensor.([
        TK.randn(rng, ComplexF64, ⊗(boundary, physical), ⊗(physical, virtual)),
        TK.randn(rng, ComplexF64, ⊗(virtual, physical), ⊗(physical, boundary)),
    ]))
    FiniteMPS.canonicalize!(state, 1)
    return state
end

"Rank-three first site, real dimension-one y at site 2, and dimension-two y at site 3."
function tangent_mixed_rank_base_state()
    physical = FiniteMPS.NoSymSpinOneHalf.pspace
    boundary = TK.ComplexSpace(1)
    virtual = TK.ComplexSpace(2)
    rng = MersenneTwister(0x746d_6978)
    state = FiniteMPS.MPO(FiniteMPS.MPSTensor.([
        TK.randn(rng, ComplexF64, ⊗(boundary, physical), virtual),
        TK.randn(rng, ComplexF64, ⊗(virtual, physical), ⊗(boundary, virtual)),
        TK.randn(rng, ComplexF64, ⊗(virtual, physical), ⊗(virtual, boundary)),
    ]))
    FiniteMPS.canonicalize!(state, 1)
    return state
end

"Independent dense sum over insertion sites, with b, every actual y, and q kept open."
function dense_tangent_boundary_weights(state; purified=true, coherent=true)
    left = [convert(Array, tensor.A) for tensor in state.base.Al]
    right = [convert(Array, tensor.A) for tensor in state.base.Ar]
    insertion = [convert(Array, tensor.A) for tensor in state.B]
    L = length(left)
    operator_like = any(A -> ndims(A) == 4, left)
    has_symmetry = ndims(first(insertion)) == ndims(first(left)) + 1
    physical_dimensions = [size(A, 2) for A in left]
    local_dimensions = [ndims(A) == 4 ? size(A, 3) : 1 for A in left]
    symmetry_dimension = has_symmetry ? size(first(insertion), ndims(first(insertion)) - 1) : 1
    boundary_dimension = size(first(left), 1)
    weights = Dict{Tuple{Vararg{Int}},Float64}()
    for physical in all_configurations(physical_dimensions)
        physical_weight = 0.0
        for local_indices in all_configurations(local_dimensions), q in 1:symmetry_dimension
            terms = Matrix{ComplexF64}[]
            for inserted_site in 1:L
                term = Matrix{ComplexF64}(I, boundary_dimension, boundary_dimension)
                for site in 1:L
                    A = site < inserted_site ? left[site] : right[site]
                    indices = (Colon(), physical[site])
                    ndims(left[site]) == 4 && (indices = (indices..., local_indices[site]))
                    if site == inserted_site
                        A = insertion[site]
                        has_symmetry && (indices = (indices..., q))
                    end
                    channel = view(A, indices..., Colon())
                    term = transpose(channel) * term
                end
                push!(terms, term)
            end
            # Interference is retained between insertion positions, while
            # orthogonal auxiliary-basis labels contribute incoherent weights.
            boundary_weights = if coherent
                amplitudes = reduce(+, terms)
                [real(sum(abs2, @view amplitudes[:, b])) for b in 1:boundary_dimension]
            else
                [sum(term -> real(sum(abs2, @view term[:, b])), terms) for b in 1:boundary_dimension]
            end
            if purified
                physical_weight += sum(boundary_weights)
            else
                y = [ndims(left[site]) == 4 ? local_indices[site] : 0 for site in 1:L]
                for b in 1:boundary_dimension
                    configuration = operator_like ? vcat(physical, y, b) : vcat(physical, b)
                    has_symmetry && push!(configuration, q)
                    weights[Tuple(configuration)] = boundary_weights[b]
                end
            end
        end
        purified && (weights[Tuple(physical)] = physical_weight)
    end
    return weights
end

"Dense suffix Gram operators with both independent q indices retained."
function dense_tangent_suffix_grams(state, cut)
    left = [convert(Array, tensor.A) for tensor in state.base.Al]
    right = [convert(Array, tensor.A) for tensor in state.base.Ar]
    insertion = [convert(Array, tensor.A) for tensor in state.B]
    has_q = ndims(first(insertion)) == ndims(first(left)) + 1
    Q = has_q ? size(first(insertion), ndims(first(insertion)) - 1) : 1
    D = cut == 0 ? size(first(left), 1) : size(left[cut], ndims(left[cut]))
    sites = (cut + 1):length(left)
    physical_dimensions = [size(left[site], 2) for site in sites]
    local_dimensions = [ndims(left[site]) == 4 ? size(left[site], 3) : 1 for site in sites]
    inserted = zeros(ComplexF64, D, D)
    cross = zeros(ComplexF64, D, D, Q)
    uninserted = zeros(ComplexF64, D, D, Q, Q)
    for physical in all_configurations(physical_dimensions), local_indices in all_configurations(local_dimensions)
        function suffix(inserted_site, q)
            amplitude = Matrix{ComplexF64}(I, D, D)
            for (offset, site) in enumerate(sites)
                tensor = site == inserted_site ? insertion[site] :
                         site < inserted_site ? left[site] : right[site]
                indices = (Colon(), physical[offset])
                ndims(left[site]) == 4 && (indices = (indices..., local_indices[offset]))
                site == inserted_site && has_q && (indices = (indices..., q))
                amplitude = transpose(view(tensor, indices..., Colon())) * amplitude
            end
            return amplitude
        end
        R = suffix(0, 1)
        T = [sum((suffix(site, q) for site in sites); init=zeros(ComplexF64, size(R))) for q in 1:Q]
        inserted .+= adjoint(R) * R
        for q in 1:Q
            cross[:, :, q] .+= adjoint(R) * T[q]
            for qprime in 1:Q
                uninserted[:, :, q, qprime] .+= adjoint(T[q]) * T[qprime]
            end
        end
    end
    return (; inserted, cross, uninserted)
end

function dense_tangent_block_matrix(matrix, ranges)
    dimension = sum(length, ranges)
    dense = zeros(ComplexF64, dimension, dimension)
    for (key, block) in zip(matrix.keys, matrix.blocks)
        dense[ranges[key[1]], ranges[key[2]]] .= block
    end
    return dense
end

function test_tangent_residual_completion(metric, dense, plan; side, purified)
    full = side === :left ? plan.left : plan.right
    residual = side === :left ? plan.residual_left : plan.residual_right
    order = zeros(Int, full.fulldim)
    for (slot, sector) in enumerate(full.sectors)
        embedding = residual.embeddings[slot]
        order[metric.ranges[embedding.residual_slot][embedding.rows]] .= TK.axes(full.space, sector)
    end
    @test sort(order) == collect(1:full.fulldim)
    @test dense_tangent_block_matrix(metric.inserted, metric.ranges) ≈
          dense.inserted[order, order] rtol=3e-11 atol=3e-13
    for q in eachindex(metric.cross)
        @test dense_tangent_block_matrix(metric.cross[q], metric.ranges) ≈
              dense.cross[order, order, q] rtol=3e-11 atol=3e-13
    end
    @test length(metric.uninserted) == (purified ? 1 : size(dense.cross, 3))
    if purified
        traced = sum(dense.uninserted[:, :, q, q] for q in axes(dense.cross, 3))
        @test dense_tangent_block_matrix(only(metric.uninserted), metric.ranges) ≈
              traced[order, order] rtol=3e-11 atol=3e-13
    else
        for q in eachindex(metric.uninserted)
            @test dense_tangent_block_matrix(metric.uninserted[q], metric.ranges) ≈
                  dense.uninserted[order, order, q, q] rtol=3e-11 atol=3e-13
        end
    end
end

"Force external `[x..., optional y..., b, optional q]` using the sampler's draw order."
function uniforms_for_tangent_configuration(probabilities, target, state; purified)
    purified && return uniforms_for_configuration(probabilities, target)
    L = length(state.B)
    ranks = BS._tensor_rank.(state.base.Al)
    operator_like = any(==(4), ranks)
    has_symmetry = BS._tensor_rank(first(state.B)) == first(ranks) + 1
    boundary_row = (operator_like ? 2 * L : L) + 1
    function draw_order(configuration)
        outcomes = Int[configuration[boundary_row]]
        has_symmetry && push!(outcomes, last(configuration))
        for site in 1:L
            local_dimension = ranks[site] == 4 ? Int(TK.dim(BS.purspace(state.base.Al[site]))) : 1
            y = ranks[site] == 4 ? configuration[L + site] : 1
            push!(outcomes, (configuration[site] - 1) * local_dimension + y)
        end
        return outcomes
    end
    ordered = Dict(Tuple(draw_order(configuration)) => weight for (configuration, weight) in probabilities)
    return uniforms_for_configuration(ordered, draw_order(target))
end

function tangent_temporary_directories(prefix)
    return Set(filter(readdir(tempdir(); join=true)) do path
        startswith(basename(path), prefix)
    end)
end

"Compare tangent and ordinary joint output with the same retained boundary."
function test_joint_tangent_isomorphism(result, ordinary)
    state = ordinary.state
    is_mps = state isa FiniteMPS.MPS
    reference = normalize_weights!(
        is_mps ? dense_mps_boundary_weights(state; purified=false) :
        dense_mpo_boundary_weights(state; purified=false),
    )
    for shot in eachindex(result.log_probability)
        target = collect(@view result.configuration[:, shot])
        uniforms = if is_mps
            uniforms_for_mps_boundary_configuration(reference, target; purified=false)
        else
            uniforms_for_mpo_boundary_configuration(reference, target, state)
        end
        ordinary_shot = BS.bornsample!(SequenceRNG(uniforms), ordinary)
        @test ordinary_shot.configuration == target
        @test ordinary_shot.log_probability ≈
              result.log_probability[shot] atol=2e-12 rtol=2e-12
        @test exp(result.log_probability[shot]) ≈
              reference[Tuple(target)] atol=2e-12 rtol=2e-12
    end
end

@testset "TangentMPS sampling contract" begin
    @testset "natural Hilbert-space isomorphism" begin
        for state in (
            rank3_state(length=3, bonddim=2),
            FiniteMPS.identityMPO(
                ComplexF64,
                3,
                FiniteMPS.NoSymSpinOneHalf.pspace,
            ),
        )
            base = FiniteMPSTangents.BaseMPS(state)
            tangent = FiniteMPSTangents.TangentMPS(base)
            tangent_sampler = BS.BornSampler(tangent)
            ordinary_sampler = BS.BornSampler(
                compatible_base_state(base, state),
            )

            tangent_result = BS.bornsample!(
                MersenneTwister(0x7461_6e67),
                tangent_sampler,
                8;
                ntasks=2,
            )
            ordinary_result = BS.bornsample!(
                MersenneTwister(0x7461_6e67),
                ordinary_sampler,
                8;
                ntasks=2,
            )
            @test tangent_result.configuration == ordinary_result.configuration
            @test tangent_result.log_probability ≈
                  ordinary_result.log_probability atol=2e-12 rtol=2e-12

            tangent_joint = BS.BornSampler(tangent; purified=false)
            # Force matching outcomes without assuming an identical RNG stream.
            ordinary_joint = BS.BornSampler(
                compatible_base_state(base, state);
                purified=false,
            )
            tangent_joint_result = BS.bornsample!(
                MersenneTwister(0x6a6f_696e),
                tangent_joint,
                8;
                ntasks=2,
            )
            test_joint_tangent_isomorphism(tangent_joint_result, ordinary_joint)

            direct_result = BS.bornsample!(
                MersenneTwister(0x6469_7265),
                tangent,
            )
            sampler_result = BS.bornsample!(
                MersenneTwister(0x6469_7265),
                tangent_sampler,
            )
            @test direct_result == sampler_result
        end
    end

    @testset "shared batch and disk codec" begin
        state = rank3_state(length=3, bonddim=2)
        tangent = FiniteMPSTangents.TangentMPS(
            FiniteMPSTangents.BaseMPS(state),
        )
        sampler = BS.BornSampler(tangent)
        memory = BS.bornsample!(
            MersenneTwister(0x6361_6368),
            sampler,
            12;
            ntasks=3,
        )
        disk = BS.bornsample!(
            MersenneTwister(0x6361_6368),
            sampler,
            12;
            ntasks=3,
            disk=true,
            maxsize=1,
        )
        @test memory == disk
        @test size(memory.configuration) == (3, 12)
        @test all(isfinite, memory.log_probability)

        joint = BS.bornsample!(
            MersenneTwister(0x6a6f_696e),
            tangent,
            12;
            purified=false,
            ntasks=3,
        )
        @test size(joint.configuration) == (4, 12)
        @test all(==(1), @view joint.configuration[end, :])
        @test all(isfinite, joint.log_probability)
    end

    @testset "multi-site global-q root and shared scheduler" begin
        state = rank3_state(length=3, bonddim=2)
        symmetry = TK.ComplexSpace(2)
        tangent = tangent_with_persistent_symmetry(
            FiniteMPSTangents.BaseMPS(state),
            symmetry;
            seed=0x716d_756c,
        )

        for purified in (true, false)
            sampler = BS.BornSampler(tangent; purified)
            @test (first(sampler.plans) isa BS.TangentBoundaryPlan) == !purified
            @test length(sampler.plans) == length(state) + 2 * Int(!purified)
            !purified && @test sampler.plans[2] isa BS.TangentGlobalQPlan
            run = BS._begin_sampling_run(sampler; disk=true)
            completion_directory = run.store.directory
            expected_files = Set(
                "completion_$site.bin" for
                site in (purified ? (2:length(state)) : (1:length(state)))
            )
            try
                @test Set(readdir(completion_directory)) == expected_files
                BS._take_sampling_completion!(run, 1)
                @test Set(readdir(completion_directory)) == expected_files
                for layer in 2:length(sampler.plans)
                    BS._take_sampling_completion!(run, layer)
                    site = purified ? layer : layer - 2
                    delete!(expected_files, "completion_$site.bin")
                    @test Set(readdir(completion_directory)) == expected_files
                end
            finally
                BS._cleanup_sampling_run!(run)
            end
            @test !ispath(completion_directory)

            serial = BS.bornsample!(
                MersenneTwister(0x7162_6174),
                sampler,
                16;
                ntasks=1,
            )
            parallel = BS.bornsample!(
                MersenneTwister(0x7162_6174),
                sampler,
                16;
                ntasks=Threads.nthreads() + 3,
            )
            completion_directories = tangent_temporary_directories(
                "BornSampling-tangent-completion-",
            )
            prefix_directories = tangent_temporary_directories(
                "BornSampling-tangent-prefix-",
            )
            disk_result = BS.bornsample!(
                MersenneTwister(0x7162_6174),
                sampler,
                16;
                ntasks=Threads.nthreads() + 3,
                disk=true,
                maxsize=1,
            )
            @test tangent_temporary_directories(
                "BornSampling-tangent-completion-",
            ) == completion_directories
            @test tangent_temporary_directories(
                "BornSampling-tangent-prefix-",
            ) == prefix_directories
            @test parallel == serial
            @test disk_result == serial
            @test size(serial.configuration) ==
                  (purified ? 3 : 5, 16)
            @test all(isfinite, serial.log_probability)
            if !purified
                @test all(==(1), @view serial.configuration[4, :])
                @test all(q -> 1 <= q <= 2, serial.configuration[5, :])
            end
        end
    end

    @testset "dense complete-boundary and auxiliary probabilities: $name" for
        (name, source, symmetry) in (
            ("SU(2)", tangent_nonabelian_base_state(), TK.Rep[TK.SU₂](1 => 1)),
            ("mixed rank with singleton y", tangent_mixed_rank_base_state(), TK.ComplexSpace(2)),
        )
        base = FiniteMPSTangents.BaseMPS(source)
        L = length(base.Al)
        operator_like = any(tensor -> BS._tensor_rank(tensor) == 4, base.Al)
        boundary_dimension = Int(TK.dim(BS.leftspace(first(base.Al))))
        for has_symmetry in (false, true)
            tangent = has_symmetry ? tangent_with_persistent_symmetry(base, symmetry) :
                      tangent_with_random_insertions(base)
            traced = normalize_weights!(dense_tangent_boundary_weights(tangent))
            joint = normalize_weights!(dense_tangent_boundary_weights(tangent; purified=false))
            incoherent = normalize_weights!(dense_tangent_boundary_weights(tangent; coherent=false))
            @test maximum(abs(traced[configuration] - incoherent[configuration]) for configuration in keys(traced)) > 1e-5
            @test sum(values(traced)) ≈ 1 atol=3e-13
            @test sum(values(joint)) ≈ 1 atol=3e-13
            for (physical, probability) in traced
                marginal = sum(joint) do (configuration, weight)
                    configuration[1:L] == physical ? weight : 0.0
                end
                @test marginal ≈ probability rtol=3e-11 atol=3e-13
            end

            joint_length = (operator_like ? 2 * L : L) + 1 + Int(has_symmetry)
            for purified in (true, false)
                sampler = BS.BornSampler(tangent; purified)
                reference = purified ? traced : joint
                expected_length = purified ? L : joint_length
                initial = sampler.initial_factor
                @test size(initial.uninserted) == (boundary_dimension, boundary_dimension)
                @test initial.uninserted * adjoint(initial.uninserted) ≈
                      Matrix{ComplexF64}(I, boundary_dimension, boundary_dimension) / boundary_dimension atol=3e-13
                @test all(iszero, initial.inserted)
                for (target_key, probability) in reference
                    probability <= 1e-15 && continue
                    target = collect(target_key)
                    uniforms = uniforms_for_tangent_configuration(reference, target, tangent; purified)
                    configuration = fill(-1, expected_length)
                    log_probability = BS.bornsample!(SequenceRNG(uniforms), sampler, configuration)
                    @test configuration == target
                    @test exp(log_probability) ≈ probability rtol=3e-11 atol=3e-13
                end

                seed = 0x7461_7578
                direct = BS.bornsample!(MersenneTwister(seed), tangent; purified)
                scalar = BS.bornsample!(MersenneTwister(seed), sampler)
                @test direct == scalar
                @test length(scalar.configuration) == expected_length
                @test exp(scalar.log_probability) ≈ reference[Tuple(scalar.configuration)] rtol=3e-11 atol=3e-13
                @test_throws DimensionMismatch BS.bornsample!(
                    MersenneTwister(seed), sampler, Vector{Int}(undef, expected_length + 1),
                )

                nshots = 16
                seed_rng = MersenneTwister(seed)
                shot_seeds = [rand(seed_rng, UInt64) for _ in 1:nshots]
                expected_configuration = Matrix{Int}(undef, expected_length, nshots)
                expected_log_probability = [
                    BS.bornsample!(Random.Xoshiro(shot_seeds[shot]), sampler, @view expected_configuration[:, shot])
                    for shot in 1:nshots
                ]
                expected_next_caller_value = rand(seed_rng, UInt64)
                baseline = nothing
                for disk in (false, true), ntasks in (1, Threads.nthreads() + 2)
                    caller_rng = MersenneTwister(seed)
                    result = BS.bornsample!(caller_rng, sampler, nshots; disk, ntasks, maxsize=1)
                    @test result.configuration == expected_configuration
                    @test result.log_probability ≈ expected_log_probability rtol=3e-11 atol=3e-13
                    @test rand(caller_rng, UInt64) == expected_next_caller_value
                    if baseline === nothing
                        baseline = result
                    else
                        @test result == baseline
                    end
                    for shot in eachindex(result.log_probability)
                        @test exp(result.log_probability[shot]) ≈
                              reference[Tuple(@view result.configuration[:, shot])] rtol=3e-11 atol=3e-13
                    end
                    if !purified && operator_like
                        @test all(==(0), @view result.configuration[L + 1, :])
                        @test all(==(1), @view result.configuration[L + 2, :])
                        @test all(y -> 1 <= y <= 2, @view result.configuration[L + 3, :])
                        @test all(==(1), @view result.configuration[2 * L + 1, :])
                    end
                end
                direct_batch = BS.bornsample!(
                    MersenneTwister(seed), tangent, nshots;
                    purified, disk=true, ntasks=2, maxsize=1,
                )
                @test direct_batch == baseline
                empty_batch = BS.bornsample!(MersenneTwister(seed), sampler, 0; disk=true, maxsize=1)
                @test size(empty_batch.configuration) == (expected_length, 0)
                @test isempty(empty_batch.log_probability)
            end
        end
    end

    @testset "multi-site non-Abelian local and global auxiliary legs" begin
        product_sector = TK.Irrep[TK.:×(TK.U₁, TK.SU₂)]
        product_source = residual_route_rank4_state()
        FiniteMPS.canonicalize!(product_source, 1)
        for (source, symmetry) in (
            (tangent_nonabelian_operator_base_state(), TK.Rep[TK.SU₂](1 => 1)),
            (product_source, TK.GradedSpace(
                product_sector(0, 1) => 1,
                product_sector(1, 1 // 2) => 1,
            )),
        )
            tangent = tangent_with_persistent_symmetry(
                FiniteMPSTangents.BaseMPS(source), symmetry;
                seed=0x7479_7172,
            )
            @test BS._tensor_rank.(tangent.base.Al) == [4, 4]
            @test BS._tensor_rank.(tangent.B) == [5, 5]
            traced = normalize_weights!(dense_tangent_boundary_weights(tangent))
            joint = normalize_weights!(dense_tangent_boundary_weights(tangent; purified=false))
            incoherent = normalize_weights!(dense_tangent_boundary_weights(tangent; coherent=false))
            @test maximum(abs(traced[key] - incoherent[key]) for key in keys(traced)) > 1e-5
            for (physical, probability) in traced
                @test sum(weight for (key, weight) in joint if key[1:2] == physical) ≈
                      probability rtol=3e-11 atol=3e-13
            end
            for purified in (true, false)
                sampler = BS.BornSampler(tangent; purified)
                reference = purified ? traced : joint
                # Exercise every supported output, including correlations
                # between the sampled q and both local purification legs.
                for (target_key, probability) in reference
                    probability <= 1e-15 && continue
                    target = collect(target_key)
                    sample = BS.bornsample!(
                        SequenceRNG(uniforms_for_tangent_configuration(
                            reference, target, tangent; purified,
                        )),
                        sampler,
                    )
                    @test sample.configuration == target
                    @test exp(sample.log_probability) ≈ probability rtol=3e-11 atol=3e-13
                end
                memory = BS.bornsample!(MersenneTwister(0x7479_6261), sampler, 24; ntasks=1)
                disk = BS.bornsample!(
                    MersenneTwister(0x7479_6261), sampler, 24;
                    disk=true, maxsize=1, ntasks=Threads.nthreads() + 2,
                )
                @test disk == memory
                for shot in eachindex(memory.log_probability)
                    @test exp(memory.log_probability[shot]) ≈
                          reference[Tuple(@view memory.configuration[:, shot])] rtol=3e-11 atol=3e-13
                end
            end
        end
    end

    @testset "original-symmetry suffixes and complete conversion before sampling" begin
        nonabelian_base = FiniteMPSTangents.BaseMPS(tangent_nonabelian_base_state())
        product_sector = TK.Irrep[TK.:×(TK.U₁, TK.SU₂)]
        product_source = residual_route_rank4_state()
        FiniteMPS.canonicalize!(product_source, 1)
        dual_q = TK.GradedSpace(
            product_sector(0, 1) => 1,
            product_sector(1, 1 // 2) => 1,
        )'
        for tangent in (
            tangent_with_random_insertions(nonabelian_base),
            tangent_with_persistent_symmetry(nonabelian_base, TK.Rep[TK.SU₂](1 => 1)),
            tangent_with_persistent_symmetry(FiniteMPSTangents.BaseMPS(product_source), dual_q),
        ), purified in (true, false)
            sampler = BS.BornSampler(tangent; purified)
            steps = BS._tangent_local_steps(sampler)
            mode = purified ? BS.TracedTangentMode : BS.JointTangentMode
            has_q = BS._tensor_rank(first(tangent.B)) == BS._tensor_rank(first(tangent.base.Al)) + 1
            original_sector = TK.sectortype(first(tangent.B).A)
            metric = BS._right_symmetric_tangent_completion(ComplexF64, tangent, mode)
            for cut in length(steps):-1:0
                dense = dense_tangent_suffix_grams(tangent, cut)
                @test metric isa BS.SymmetricTangentCompletion
                @test TK.sectortype(metric.inserted) === original_sector
                @test TK.sectortype(metric.cross) === original_sector
                @test TK.sectortype(metric.uninserted) === original_sector
                @test TK.numind(metric.inserted) == 2
                @test TK.numind(metric.cross) == (has_q ? 3 : 2)
                @test TK.numind(metric.uninserted) == (has_q && !purified ? 4 : 2)
                @test convert(Array, metric.inserted) ≈ transpose(dense.inserted) rtol=3e-11 atol=3e-13
                expected_cross = has_q ? permutedims(dense.cross, (2, 3, 1)) :
                                 transpose(dense.cross[:, :, 1])
                @test convert(Array, metric.cross) ≈ expected_cross rtol=3e-11 atol=3e-13
                expected_uninserted = if purified
                    transpose(sum(dense.uninserted[:, :, q, q] for q in axes(dense.cross, 3)))
                elseif has_q
                    permutedims(dense.uninserted, (2, 3, 4, 1))
                else
                    transpose(dense.uninserted[:, :, 1, 1])
                end
                @test convert(Array, metric.uninserted) ≈ expected_uninserted rtol=3e-11 atol=3e-13
                step = steps[max(cut, 1)]
                side = cut == 0 ? :left : :right
                converted = BS._residual_tangent_completion(metric, step, mode; side)
                test_tangent_residual_completion(converted, dense, step.left; side, purified)
                cut == 0 && continue
                metric = BS._retreat_symmetric_tangent_completion(
                    metric, tangent.base.Al[cut], tangent.base.Ar[cut], tangent.B[cut], mode,
                )
            end

            for disk in (false, true)
                run = BS._begin_sampling_run(sampler; disk)
                try
                    @test run.root_completion isa BS.TangentCompletionMetric
                    pending = if disk
                        [open(BS.deserialize, path) for path in readdir(run.store.directory; join=true)]
                    else
                        filter(!isnothing, run.store.values)
                    end
                    @test all(value -> value isa BS.TangentCompletionMetric, pending)
                    for layer in eachindex(sampler.plans)
                        @test BS._take_sampling_completion!(run, layer) isa BS.TangentCompletionMetric
                    end
                    @test all(isnothing, run.store.values)
                    disk && @test isempty(readdir(run.store.directory))
                finally
                    BS._cleanup_sampling_run!(run)
                end
            end
        end
    end

    @testset "consume-once completion store" begin
        for disk in (false, true)
            store = BS.TangentCompletionStore{Vector{Int}}(3; disk)
            directory = store.directory
            try
                BS._put_completion!(store, 2, [2, 3])
                BS._put_completion!(store, 1, [1])
                BS._put_completion!(store, 3, [4, 5, 6])
                if disk
                    @test Set(readdir(directory)) == Set(
                        "completion_$site.bin" for site in 1:3
                    )
                end

                @test BS._take_completion!(store, 1) == [1]
                if disk
                    @test Set(readdir(directory)) ==
                          Set(["completion_2.bin", "completion_3.bin"])
                else
                    @test isnothing(store.values[1])
                end

                @test BS._take_completion!(store, 2) == [2, 3]
                if disk
                    @test Set(readdir(directory)) == Set(["completion_3.bin"])
                else
                    @test isnothing(store.values[2])
                    @test store.values[3] == [4, 5, 6]
                end
            finally
                # Site 3 is deliberately left pending. Cleanup must remove an
                # unconsumed in-memory value or disk file after an early exit.
                BS._cleanup_completion_store!(store)
            end
            BS._cleanup_completion_store!(store)
            @test all(isnothing, store.values)
            if directory !== nothing
                @test !ispath(directory)
            end
        end
    end


    @testset "mixed-rank MPO base" begin
        state = mixed_rank_three_site_mpo_state()
        FiniteMPS.canonicalize!(state, 1)
        tangent = FiniteMPSTangents.TangentMPS(
            FiniteMPSTangents.BaseMPS(state),
        )
        @test sort(unique(BS._tensor_rank.(tangent.base.Al))) == [3, 4]
        result = BS.bornsample!(
            MersenneTwister(0x6d69_7865),
            tangent,
            4;
            ntasks=2,
        )
        @test size(result.configuration) == (3, 4)
        @test all(isfinite, result.log_probability)

        joint = BS.bornsample!(
            MersenneTwister(0x6d69_786a),
            tangent,
            4;
            purified=false,
            ntasks=2,
        )
        @test size(joint.configuration) == (7, 4)
        @test all(view(joint.configuration, 5:6, :) .== 0)
        @test all(view(joint.configuration, 7, :) .== 1)
        @test all(isfinite, joint.log_probability)
        ordinary = BS.BornSampler(
            compatible_base_state(tangent.base, state); purified=false,
        )
        test_joint_tangent_isomorphism(joint, ordinary)
    end

    @testset "rank-derived layout and unsupported boundaries" begin
        original = rank3_state(length=2, bonddim=2)
        all_rank_three_mpo = FiniteMPS.MPO(deepcopy(original.A))
        FiniteMPS.canonicalize!(all_rank_three_mpo, 1)
        tangent = FiniteMPSTangents.TangentMPS(FiniteMPSTangents.BaseMPS(all_rank_three_mpo))
        @test all(tensor -> BS._tensor_rank(tensor) == 3, tangent.base.Al)
        result = BS.bornsample!(MersenneTwister(0x7261_6e6b), tangent; purified=false)
        @test length(result.configuration) == 3
        @test last(result.configuration) == 1
        reference = normalize_weights!(dense_tangent_boundary_weights(tangent; purified=false))
        @test exp(result.log_probability) ≈ reference[Tuple(result.configuration)] rtol=3e-11 atol=3e-13

        singleton_symmetry = tangent_with_persistent_symmetry(tangent.base, TK.ComplexSpace(1))
        singleton_result = BS.bornsample!(MersenneTwister(0x7369_6e71), singleton_symmetry; purified=false)
        @test length(singleton_result.configuration) == 4
        @test singleton_result.configuration[3:4] == [1, 1]
        singleton_reference = normalize_weights!(dense_tangent_boundary_weights(singleton_symmetry; purified=false))
        @test exp(singleton_result.log_probability) ≈ singleton_reference[Tuple(singleton_result.configuration)] rtol=3e-11 atol=3e-13

        left_multiplicity = FiniteMPSTangents.TangentMPS(
            FiniteMPSTangents.BaseMPS(nontrivial_left_state()),
        )
        for purified in (true, false)
            @test_throws ArgumentError BS.BornSampler(left_multiplicity; purified)
            @test_throws ArgumentError BS.BornSampler(tangent; purified, left_boundary=ComplexF64[1])
            @test_throws ArgumentError BS.bornsample!(
                MersenneTwister(1), tangent; purified, left_boundary=ComplexF64[1],
            )
            @test_throws ArgumentError BS.bornsample!(
                MersenneTwister(1), tangent, 2; purified, left_boundary=ComplexF64[1],
            )
        end
    end

    @testset "product symmetry keeps its Abelian residual" begin
        state = residual_route_rank4_state()
        FiniteMPS.canonicalize!(state, 1)
        base = FiniteMPSTangents.BaseMPS(state)
        tangent = FiniteMPSTangents.TangentMPS(
            base,
        )
        sampler = BS.BornSampler(tangent; purified=false)
        has_multiple_residual_sectors = false
        for plan in sampler.plans
            plan isa BS.TangentSitePlan || continue
            step = plan.step
            for local_plan in (step.left, step.right, step.insertion)
                @test local_plan isa BS.SitePlan{4,BS.FusionTreeStyle}
                for info in (
                    local_plan.residual_left,
                    local_plan.residual_right,
                )
                    @test eltype(info.sectors) === TK.Irrep[TK.U₁]
                    has_multiple_residual_sectors |= length(info.sectors) > 1
                end
            end
        end
        @test has_multiple_residual_sectors

        ordinary = BS.BornSampler(
            compatible_base_state(base, state);
            purified=false,
        )
        tangent_result = BS.bornsample!(
            MersenneTwister(0x7265_7369),
            sampler,
            8;
            ntasks=2,
        )
        test_joint_tangent_isomorphism(tangent_result, ordinary)
    end

    @testset "persistent non-Abelian symmetry leg" begin
        physical = FiniteMPS.SU2Spin.pspace
        vacuum = TK.Rep[TK.SU₂](0 => 1)
        left = TK.Rep[TK.SU₂](1 // 2 => 1)
        symmetry = TK.Rep[TK.SU₂](1 => 1)
        state = FiniteMPS.MPS([
            FiniteMPS.MPSTensor(
                TK.randn(
                    MersenneTwister(0x7375_3273),
                    ComplexF64,
                    ⊗(left, physical),
                    vacuum,
                ),
            ),
        ])
        rank_four = tangent_with_persistent_symmetry(
            FiniteMPSTangents.BaseMPS(state),
            symmetry,
        )
        result_four = BS.bornsample!(
            MersenneTwister(0x7134),
            rank_four,
        )
        @test length(result_four.configuration) == 1
        @test isfinite(result_four.log_probability)
        joint_four = BS.bornsample!(
            MersenneTwister(0x7134),
            rank_four;
            purified=false,
        )
        @test length(joint_four.configuration) == 3
        @test 1 <= joint_four.configuration[2] <= Int(TK.dim(left))
        @test 1 <= joint_four.configuration[3] <= Int(TK.dim(symmetry))
        @test isfinite(joint_four.log_probability)

        operator_left = TK.Rep[TK.SU₂](1 => 1)
        operator_state = FiniteMPS.MPO([
            FiniteMPS.MPSTensor(TK.randn(
                MersenneTwister(0x7179_6261),
                ComplexF64,
                ⊗(operator_left, physical),
                ⊗(physical, vacuum),
            )),
        ])
        rank_five = tangent_with_persistent_symmetry(
            FiniteMPSTangents.BaseMPS(operator_state),
            symmetry;
            seed=0x7135,
        )
        result_five = BS.bornsample!(MersenneTwister(0x7135), rank_five)
        @test length(result_five.configuration) == 1
        @test isfinite(result_five.log_probability)
        joint_five = BS.bornsample!(
            MersenneTwister(0x7135),
            rank_five;
            purified=false,
        )
        @test length(joint_five.configuration) == 4
        @test 1 <= joint_five.configuration[2] <= Int(TK.dim(physical))
        @test 1 <= joint_five.configuration[3] <= Int(TK.dim(operator_left))
        @test 1 <= joint_five.configuration[4] <= Int(TK.dim(symmetry))
        @test isfinite(joint_five.log_probability)
        @test BS._tensor_rank.(rank_four.B) == [4]
        @test BS._tensor_rank.(rank_five.B) == [5]

        for tangent in (rank_four, rank_five), purified in (true, false)
            reference = normalize_weights!(dense_tangent_boundary_weights(tangent; purified))
            sampler = BS.BornSampler(tangent; purified)
            for (target_key, probability) in reference
                probability <= 1e-15 && continue
                target = collect(target_key)
                sample = BS.bornsample!(
                    SequenceRNG(uniforms_for_tangent_configuration(
                        reference, target, tangent; purified,
                    )),
                    sampler,
                )
                @test sample.configuration == target
                @test exp(sample.log_probability) ≈ probability rtol=3e-11 atol=3e-13
            end
            serial = BS.bornsample!(MersenneTwister(0x7134_7135), sampler, 12; ntasks=1)
            disk = BS.bornsample!(
                MersenneTwister(0x7134_7135), sampler, 12;
                ntasks=Threads.nthreads() + 2, disk=true, maxsize=1,
            )
            @test disk == serial
            for shot in eachindex(serial.log_probability)
                @test exp(serial.log_probability[shot]) ≈
                      reference[Tuple(@view serial.configuration[:, shot])] rtol=3e-11 atol=3e-13
            end
        end
    end
end
