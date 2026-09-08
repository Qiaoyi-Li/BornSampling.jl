# Count logical reads only while a selected source frontier is read-only.
# Destination writes use another cache, and the normal path method remains the
# implementation under test. No production counters or codec replacement.
const prefix_read_probe = Ref{Any}(nothing)

function BS._factor_path(cache::BS.PrefixCache{T,R,F,F}, id::Int) where {T,R,F}
    probe = prefix_read_probe[]
    if probe !== nothing && cache === probe.cache
        Threads.atomic_add!(probe.counts[id], 1)
    end
    return invoke(BS._factor_path, Tuple{BS.PrefixCache,Int}, cache, id)
end

@testset "batched prefix-tree sampling" begin
    @testset "output layout, validation, and deterministic scheduling" begin
        sampler3 = BS.BornSampler(rank3_state(length=4, bonddim=3))
        nshots = 24
        seed = 0xb47c_31a9

        serial3 = BS.bornsample!(
            MersenneTwister(seed), sampler3, nshots;
            ntasks=1,
            disk=false,
        )
        oversubscribed_tasks = Threads.nthreads() + 3
        parallel3 = BS.bornsample!(
            MersenneTwister(seed), sampler3, nshots;
            ntasks=oversubscribed_tasks,
            disk=false,
        )

        @test keys(serial3) == (:configuration, :log_probability)
        @test size(serial3.configuration) == (length(sampler3.state), nshots)
        @test length(serial3.log_probability) == nshots
        @test parallel3.configuration == serial3.configuration
        @test parallel3.log_probability == serial3.log_probability
        @test length(sampler3.workspaces) >= min(nshots, oversubscribed_tasks)

        direct_state = rank3_state(length=2, bonddim=2)
        direct_batch = BS.bornsample!(
            MersenneTwister(seed), direct_state, 3;
            ntasks=oversubscribed_tasks,
        )
        @test size(direct_batch.configuration) == (length(direct_state), 3)

        reference3 = normalize_weights!(dense_physical_weights(sampler3.state))
        for shot in 1:nshots
            configuration = Tuple(@view serial3.configuration[:, shot])
            @test isapprox(
                exp(serial3.log_probability[shot]),
                reference3[configuration];
                rtol=2e-11,
                atol=2e-13,
            )
        end

        empty_batch = BS.bornsample!(
            MersenneTwister(seed), sampler3, 0;
            ntasks=oversubscribed_tasks,
            disk=true,
            maxsize=1,
        )
        @test size(empty_batch.configuration) == (length(sampler3.state), 0)
        @test isempty(empty_batch.log_probability)

        # `maxsize` is irrelevant without disk storage, just as documented.
        @test length(BS.bornsample!(
            MersenneTwister(seed), sampler3, 1;
            ntasks=1,
            disk=false,
            maxsize=0,
        ).log_probability) == 1

        @test_throws ArgumentError BS.bornsample!(
            MersenneTwister(seed), sampler3, -1,
        )
        @test_throws ArgumentError BS.bornsample!(
            MersenneTwister(seed), sampler3, 1; ntasks=0,
        )
        @test_throws ArgumentError BS.bornsample!(
            MersenneTwister(seed), sampler3, 1;
            ntasks=1,
            disk=true,
            maxsize=0,
        )
    end

    @testset "independent per-shot RNG oracle" begin
        sampler = BS.BornSampler(mixed_rank_three_site_mpo_state())
        nshots = 24
        seed = 0x6c61_7965
        seed_rng = MersenneTwister(seed)
        shot_seeds = [rand(seed_rng, UInt64) for _ in 1:nshots]
        expected_configuration = Matrix{Int}(
            undef,
            length(sampler.state),
            nshots,
        )
        expected_log_probability = Vector{Float64}(undef, nshots)
        for shot in 1:nshots
            expected_log_probability[shot] = BS.bornsample!(
                Random.Xoshiro(shot_seeds[shot]),
                sampler,
                @view(expected_configuration[:, shot]),
            )
        end
        expected_next_caller_value = rand(seed_rng, UInt64)

        task_counts = unique((1, 2, Threads.nthreads() + 3, nshots + 3))
        for disk in (false, true), ntasks in task_counts
            caller_rng = MersenneTwister(seed)
            result = BS.bornsample!(
                caller_rng,
                sampler,
                nshots;
                ntasks=ntasks,
                disk=disk,
                maxsize=1,
            )
            @test result.configuration == expected_configuration
            @test result.log_probability == expected_log_probability
            @test rand(caller_rng, UInt64) == expected_next_caller_value
        end
    end

    @testset "rank-4 disk cache and parallel determinism" begin
        sampler4 = BS.BornSampler(rank4_state())
        nshots = 20
        seed = 0x91e2_00d5
        memory = BS.bornsample!(
            MersenneTwister(seed), sampler4, nshots;
            ntasks=1,
            disk=false,
        )
        memory_parallel = BS.bornsample!(
            MersenneTwister(seed), sampler4, nshots;
            ntasks=Threads.nthreads() + 5,
            disk=false,
        )

        directories_before = Set(filter(
            path -> startswith(basename(path), "BornSampling-prefix-"),
            readdir(tempdir(); join=true),
        ))
        disk = BS.bornsample!(
            MersenneTwister(seed), sampler4, nshots;
            ntasks=Threads.nthreads() + 5,
            disk=true,
            maxsize=1,
        )
        disk_serial = BS.bornsample!(
            MersenneTwister(seed), sampler4, nshots;
            ntasks=1,
            disk=true,
            maxsize=1,
        )
        directories_after = Set(filter(
            path -> startswith(basename(path), "BornSampling-prefix-"),
            readdir(tempdir(); join=true),
        ))

        @test memory_parallel.configuration == memory.configuration
        @test memory_parallel.log_probability == memory.log_probability
        @test disk.configuration == memory.configuration
        @test disk.log_probability == memory.log_probability
        @test disk_serial.configuration == memory.configuration
        @test disk_serial.log_probability == memory.log_probability
        @test directories_after == directories_before

        reference4 = normalize_weights!(dense_physical_weights(sampler4.state))
        for shot in 1:nshots
            configuration = Tuple(@view disk.configuration[:, shot])
            @test isapprox(
                exp(disk.log_probability[shot]),
                reference4[configuration];
                rtol=3e-11,
                atol=5e-13,
            )
        end
    end

    @testset "cold parents load once for different children" begin
        fixtures = (
            (rank3_state(length=3, bonddim=2), true),
            (rank3_state(length=3, bonddim=2), false),
            (mixed_rank_three_site_mpo_state(), false),
        )
        for (state, purified) in fixtures
            # Deterministic positive amplitudes keep both outcomes appreciable
            # while allowing different parents to have different factors.
            tensors = map(1:length(state)) do site
                A = copy(state[site].A)
                for entry in eachindex(A.data)
                    A.data[entry] = 1 + ((entry + site) % 7) / 10
                end
                FiniteMPS.MPSTensor(A)
            end
            state = state isa FiniteMPS.MPS ? FiniteMPS.MPS(tensors) : FiniteMPS.MPO(tensors)
            for worker_count in (1, Threads.nthreads() + 2)
                sampler = BS.BornSampler(deepcopy(state); purified)
                BS._ensure_workspaces!(sampler, worker_count)
                workspace = first(sampler.workspaces)
                F, R = typeof(sampler.initial_factor), eltype(workspace.q)
                caches = BS.PrefixCache[]
                function new_cache(capacity; disk=false)
                    cache = BS._new_prefix_cache(F, R, capacity; disk, maxsize=1)
                    push!(caches, cache)
                    return cache
                end
                try
                    current = new_cache(1)
                    parent = BS._initialize_prefix_cache!(sampler, workspace, current)
                    site = 1
                    # Joint modes draw their synthetic boundary before the
                    # first physical site, even when that boundary has size one.
                    if sampler.plans[site] isa BS.BoundaryPlan
                        boundary_children = new_cache(length(parent.q))
                        parent = BS._get_or_build_prefix_child!(
                            sampler, workspace, current, boundary_children,
                            parent, argmax(parent.q), site,
                        )
                        current = boundary_children
                        site += 1
                    end
                    source = new_cache(length(parent.q); disk=true)
                    parents = [BS._get_or_build_prefix_child!(
                        sampler, workspace, current, source, parent, selected, site,
                    ) for selected in findall(>(zero(R)), parent.q)]
                    site += 1
                    @test length(parents) >= 2
                    @test length(source.resident) == 1
                    resident_ids = Set(keys(source.resident))
                    cold_ids = setdiff(Set(node.id for node in parents), resident_ids)
                    @test !isempty(cold_ids)
                    originals = Dict(node.id => copy(BS._prefix_factor(source, node.id))
                                     for node in parents)
                    current_ids = repeat([node.id for node in parents], 32)
                    nshots = length(current_ids)
                    target = new_cache(nshots)
                    next_ids = zeros(Int, nshots)
                    configuration = zeros(Int, BS._configuration_length(sampler), nshots)
                    expected_configuration = copy(configuration)
                    shot_rngs = [Random.Xoshiro(shot) for shot in 1:nshots]
                    reference_rngs = copy.(shot_rngs)
                    reference_workspace = BS._clone_workspace(workspace)
                    expected_selected = Vector{Int}(undef, nshots)
                    expected_q = Vector{Vector{R}}(undef, nshots)
                    expected_log = Vector{R}(undef, nshots)
                    for shot in 1:nshots
                        node, selected, z = BS._draw_cached_outcome!(
                            reference_rngs[shot], sampler, reference_workspace,
                            source, current_ids, expected_configuration, shot, site,
                        )
                        expected_selected[shot] = selected
                        factor = BS._advance_factor!(
                            reference_workspace, originals[node.id],
                            sampler.plans[site], selected, node.q[selected],
                        )
                        BS._compute_weights!(
                            reference_workspace, factor, sampler.plans[site + 1],
                        )
                        count = BS._outcome_count(reference_workspace, sampler.plans[site + 1])
                        expected_q[shot] = BS._copy_active_weights(reference_workspace, count)
                        expected_log[shot] = node.log_probability +
                                             (log(node.q[selected]) - log(z))
                    end
                    for id in cold_ids
                        @test length(unique(expected_selected[current_ids .== id])) >= 2
                    end
                    counts = [Threads.Atomic{Int}(0) for _ in source.nodes]
                    prefix_read_probe[] = (; cache=source, counts)
                    try
                        BS._advance_cached_layer!(
                            shot_rngs, sampler, source, target, current_ids, next_ids,
                            configuration, site, nshots, worker_count,
                        )
                    finally
                        prefix_read_probe[] = nothing
                    end
                    @test all(id -> counts[id][] == 1, cold_ids)
                    @test all(id -> counts[id][] == 0, resident_ids)
                    @test Set(keys(source.resident)) == resident_ids
                    @test all(node -> BS._prefix_factor(source, node.id) == originals[node.id], parents)
                    @test configuration == expected_configuration
                    @test rand.(shot_rngs, UInt64) == rand.(reference_rngs, UInt64)
                    edges = Dict{Tuple{Int,Int},Int}()
                    for shot in 1:nshots
                        edge = (current_ids[shot], expected_selected[shot])
                        id = next_ids[shot]
                        @test id == get!(edges, edge, id)
                        child = BS._published_prefix_node(target, id)
                        @test child.q == expected_q[shot]
                        @test child.log_probability == expected_log[shot]
                        node = BS._published_prefix_node(source, current_ids[shot])
                        @test BS._child_id(node.children[expected_selected[shot]]) == id
                    end
                    @test target.next_node_id[] == length(edges)
                    @test length(edges) < nshots
                finally
                    prefix_read_probe[] = nothing
                    for cache in reverse(caches)
                        BS._cleanup_prefix_cache!(cache)
                    end
                end
            end
        end
    end

    @testset "layer scheduling deduplicates prefixes and releases frontiers" begin
        sampler = BS.BornSampler(deterministic_rank3_state(length=5))
        nshots = 64
        worker_count = min(nshots, Threads.nthreads() + 11)
        BS._ensure_workspaces!(sampler, worker_count)
        Factor = typeof(sampler.initial_factor)
        Rprob = eltype(first(sampler.workspaces).q)
        shot_rngs = [Random.Xoshiro(i) for i in 1:nshots]
        configuration = Matrix{Int}(undef, length(sampler.state), nshots)
        log_probability = Vector{Rprob}(undef, nshots)
        current_cache = BS._new_prefix_cache(
            Factor,
            Rprob,
            1;
            disk=true,
            maxsize=1,
        )
        next_cache = nothing
        retired_directories = String[]
        try
            root = BS._initialize_prefix_cache!(
                sampler,
                first(sampler.workspaces),
                current_cache,
            )
            current_node_ids = fill(root.id, nshots)
            next_node_ids = Vector{Int}(undef, nshots)

            for site in 1:(length(sampler.state) - 1)
                next_cache = BS._new_prefix_cache(
                    Factor,
                    Rprob,
                    nshots;
                    disk=site + 1 < length(sampler.state),
                    maxsize=1,
                )
                BS._advance_cached_layer!(
                    shot_rngs,
                    sampler,
                    current_cache,
                    next_cache,
                    current_node_ids,
                    next_node_ids,
                    configuration,
                    site,
                    nshots,
                    worker_count,
                )

                # All shots have the same deterministic prefix, so dynamic
                # workers must publish exactly one node in every target layer.
                @test next_cache.next_node_id[] == 1
                @test all(==(first(next_node_ids)), next_node_ids)

                previous = current_cache
                previous_directory = previous.directory
                BS._cleanup_prefix_cache!(previous)
                @test isempty(previous.resident)
                @test all(isnothing, previous.nodes)
                if previous_directory !== nothing
                    push!(retired_directories, previous_directory)
                    @test !ispath(previous_directory)
                end
                @test all(path -> !ispath(path), retired_directories)
                if next_cache.directory !== nothing
                    @test isdir(next_cache.directory)
                end

                current_cache = next_cache
                next_cache = nothing
                current_node_ids, next_node_ids = next_node_ids, current_node_ids
            end

            BS._finish_cached_layer!(
                shot_rngs,
                sampler,
                current_cache,
                current_node_ids,
                configuration,
                log_probability,
                length(sampler.state),
                nshots,
                worker_count,
            )
            @test all(isone, configuration)
            @test all(iszero, log_probability)
        finally
            next_cache === nothing || BS._cleanup_prefix_cache!(next_cache)
            final_directory = current_cache.directory
            BS._cleanup_prefix_cache!(current_cache)
            if final_directory !== nothing
                @test !ispath(final_directory)
            end
        end
        @test all(path -> !ispath(path), retired_directories)
    end

    @testset "last-prefix metadata omits dead factor state" begin
        one_site = BS.BornSampler(deterministic_rank3_state(length=1))
        one_site_cache = BS._new_prefix_cache(
            typeof(one_site.initial_factor),
            Float64,
            4;
            disk=true,
            maxsize=1,
        )
        try
            root = BS._initialize_prefix_cache!(
                one_site,
                first(one_site.workspaces),
                one_site_cache,
            )
            @test isempty(root.children)
            @test root.factor_space === nothing
            @test root.branch_factor_spaces === nothing
            @test isempty(one_site_cache.resident)
        finally
            BS._cleanup_prefix_cache!(one_site_cache)
        end

        two_site = BS.BornSampler(deterministic_rank3_state(length=2))
        source_cache = BS._new_prefix_cache(
            typeof(two_site.initial_factor),
            Float64,
            1;
            disk=true,
            maxsize=2,
        )
        target_cache = BS._new_prefix_cache(
            typeof(two_site.initial_factor),
            Float64,
            4;
            disk=false,
            maxsize=2,
        )
        try
            workspace = first(two_site.workspaces)
            root = BS._initialize_prefix_cache!(two_site, workspace, source_cache)
            source_node_ids = [root.id]
            target_node_ids = Vector{Int}(undef, 1)
            configuration = Matrix{Int}(undef, 2, 1)
            shot_rng = MersenneTwister(0x6c65_6166)
            BS._advance_cached_layer_shot!(
                shot_rng,
                two_site,
                workspace,
                source_cache,
                target_cache,
                source_node_ids,
                target_node_ids,
                configuration,
                1,
                1,
            )
            leaf = BS._published_prefix_node(target_cache, only(target_node_ids))
            @test isempty(leaf.children)
            @test leaf.factor_space === nothing
            @test leaf.branch_factor_spaces === nothing
            @test Set(keys(source_cache.resident)) == Set((root.id,))
            @test isempty(target_cache.resident)

            log_probability = Vector{Float64}(undef, 1)
            BS._finish_cached_layer_shot!(
                shot_rng,
                two_site,
                workspace,
                target_cache,
                target_node_ids,
                configuration,
                log_probability,
                1,
                2,
            )
            @test all(isone, configuration)
            @test iszero(only(log_probability))
        finally
            BS._cleanup_prefix_cache!(target_cache)
            BS._cleanup_prefix_cache!(source_cache)
        end
    end
end

@testset "prefix metadata and probability-ranked factor storage" begin
    factor_tensor(data::AbstractMatrix{T}) where {T} = TK.TensorMap(
        copy(data), TK.ComplexSpace(size(data, 1)), TK.ComplexSpace(size(data, 2)),
    )
    root_factor = factor_tensor(reshape(ComplexF64[1 + 2im, 3 - 4im], 2, 1))
    cache = BS._new_prefix_cache(
        typeof(root_factor),
        Float64,
        8;
        disk=true,
        maxsize=2,
    )
    directory = cache.directory
    @test directory !== nothing
    @test isdir(directory)

    function register_node!(
        cache,
        log_probability,
        factor,
    )
        id = BS._allocate_node_id!(cache)
        node = BS.PrefixNode(
            id=id,
            log_probability=log_probability,
            q=Float64[],
            factor_space=TK.space(factor),
        )
        BS._set_node!(cache, node)
        BS._admit_owned_prefix_factor!(cache, node, factor)
        return node
    end

    try
        root_id = BS._allocate_node_id!(cache)
        root = BS.PrefixNode(
            id=root_id,
            log_probability=0.0,
            q=[0.6, 0.4],
            factor_space=TK.space(root_factor),
        )
        BS._set_node!(cache, root)
        root_reference = copy(root_factor)
        BS._insert_resident!(cache, root, root_factor)
        @test haskey(cache.resident, root.id)
        @test BS._prefix_factor(cache, root.id) == root_reference

        warm_factor = factor_tensor(ComplexF64[1 2; 3 4])
        warm = register_node!(cache, -0.1, warm_factor)
        @test haskey(cache.resident, warm.id)
        @test Set(keys(cache.resident)) == Set((root.id, warm.id))
        @test cache.min_resident_id == warm.id

        cold_factor = factor_tensor(ComplexF64[5 6; 7 8])
        cold = register_node!(cache, -0.2, cold_factor)
        @test !haskey(cache.resident, cold.id)
        @test BS._prefix_factor(cache, cold.id) == cold_factor
        @test !haskey(cache.resident, cold.id) # disk reads never promote

        # Admission is strict: an exact tie with the resident minimum stays cold.
        tied_factor = factor_tensor(ComplexF64[9 10; 11 12])
        tied = register_node!(cache, -0.1, tied_factor)
        @test !haskey(cache.resident, tied.id)
        @test cache.min_resident_id == warm.id
        @test BS._prefix_factor(cache, tied.id) == tied_factor

        # A strictly more probable node evicts the unique resident minimum.
        hot_factor = factor_tensor(ComplexF64[13 14; 15 16])
        hot = register_node!(cache, -0.05, hot_factor)
        @test haskey(cache.resident, root.id)
        @test haskey(cache.resident, hot.id)
        @test !haskey(cache.resident, warm.id)
        @test BS._prefix_factor(cache, warm.id) == warm_factor
        @test Set(keys(cache.resident)) == Set((root.id, hot.id))
        @test cache.min_resident_id == hot.id

        # All metadata in the current frontier, including cold-node weights,
        # remains available until the layer is released.
        @test BS._published_prefix_node(cache, cold.id) === cold
        @test BS._published_prefix_node(cache, root.id).q == [0.6, 0.4]

        BS._publish_child_id!(root.children[1], warm.id)
        @test BS._child_id(root.children[1]) == warm.id
    finally
        BS._cleanup_prefix_cache!(cache)
    end

    @test !ispath(directory)
    @test isempty(cache.resident)
    @test all(isnothing, cache.nodes)
    # Without disk storage, maxsize is intentionally ignored and every factor
    # remains resident for the duration of the batch.
    memory_root_factor = factor_tensor(ones(1, 1))
    memory_cache = BS._new_prefix_cache(
        typeof(memory_root_factor),
        Float64,
        2;
        disk=false,
        maxsize=0,
    )
    try
        root_id = BS._allocate_node_id!(memory_cache)
        root = BS.PrefixNode(
            id=root_id,
            log_probability=0.0,
            q=[1.0],
            factor_space=TK.space(memory_root_factor),
        )
        BS._set_node!(memory_cache, root)
        BS._insert_resident!(memory_cache, root, memory_root_factor)
        child_id = BS._allocate_node_id!(memory_cache)
        child_factor = factor_tensor(fill(2.0, 1, 1))
        child = BS.PrefixNode(
            id=child_id,
            log_probability=-1.0,
            q=Float64[],
            factor_space=TK.space(child_factor),
        )
        BS._set_node!(memory_cache, child)
        BS._admit_owned_prefix_factor!(
            memory_cache, child, child_factor,
        )
        @test haskey(memory_cache.resident, child.id)
        @test memory_cache.directory === nothing
        @test length(memory_cache.resident) == 2
        @test BS._prefix_factor(memory_cache, child.id) == child_factor
    finally
        BS._cleanup_prefix_cache!(memory_cache)
    end
end
