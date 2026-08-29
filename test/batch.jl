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
            path -> startswith(basename(path), "Bornsampling-prefix-"),
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
            path -> startswith(basename(path), "Bornsampling-prefix-"),
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

    @testset "one shared physical prefix under contention" begin
        sampler = BS.BornSampler(deterministic_rank3_state(length=5))
        result = BS.bornsample!(
            MersenneTwister(0x7072_6566), sampler, 64;
            ntasks=Threads.nthreads() + 11,
            disk=true,
            maxsize=1,
        )
        @test all(isone, result.configuration)
        @test all(iszero, result.log_probability)

        # Exercise the edge-level double check directly: all contenders must
        # observe one published child rather than allocate duplicate nodes.
        contenders = Threads.nthreads() + 11
        BS._ensure_workspaces!(sampler, contenders)
        cache = BS._new_prefix_cache(
            typeof(sampler.initial_factor),
            Float64,
            contenders,
            length(sampler.state);
            disk=true,
            maxsize=1,
        )
        try
            BS._initialize_prefix_cache!(
                sampler, first(sampler.workspaces), cache,
            )
            root = BS._published_prefix_node(cache, 1)
            child_ids = Vector{Int}(undef, contenders)
            @sync for worker in 1:contenders
                Threads.@spawn begin
                    child_ids[worker] = BS._get_or_build_prefix_child!(
                        sampler,
                        sampler.workspaces[worker],
                        cache,
                        root,
                        1,
                        1,
                    ).id
                end
            end
            @test all(==(first(child_ids)), child_ids)
            @test cache.next_node_id[] == 2

            child = BS._published_prefix_node(cache, first(child_ids))
            @test !child.logical_resident[]
            grandchild_ids = Vector{Int}(undef, contenders)
            @sync for worker in 1:contenders
                Threads.@spawn begin
                    grandchild_ids[worker] = BS._get_or_build_prefix_child!(
                        sampler,
                        sampler.workspaces[worker],
                        cache,
                        child,
                        1,
                        2,
                    ).id
                end
            end
            @test all(==(first(grandchild_ids)), grandchild_ids)
            @test cache.next_node_id[] == 3
        finally
            BS._cleanup_prefix_cache!(cache)
        end
    end

    @testset "last-prefix metadata omits dead factor state" begin
        one_site = BS.BornSampler(deterministic_rank3_state(length=1))
        one_site_cache = BS._new_prefix_cache(
            typeof(one_site.initial_factor),
            Float64,
            4,
            1;
            disk=true,
            maxsize=1,
        )
        try
            BS._initialize_prefix_cache!(
                one_site,
                first(one_site.workspaces),
                one_site_cache,
            )
            root = BS._published_prefix_node(one_site_cache, 1)
            @test isempty(root.children)
            @test root.factor_space === nothing
            @test !root.logical_resident[]
            @test isempty(one_site_cache.resident)
        finally
            BS._cleanup_prefix_cache!(one_site_cache)
        end

        two_site = BS.BornSampler(deterministic_rank3_state(length=2))
        two_site_cache = BS._new_prefix_cache(
            typeof(two_site.initial_factor),
            Float64,
            4,
            2;
            disk=true,
            maxsize=2,
        )
        try
            workspace = first(two_site.workspaces)
            BS._initialize_prefix_cache!(two_site, workspace, two_site_cache)
            root = BS._published_prefix_node(two_site_cache, 1)
            leaf = BS._get_or_build_prefix_child!(
                two_site,
                workspace,
                two_site_cache,
                root,
                1,
                1,
            )
            @test isempty(leaf.children)
            @test leaf.factor_space === nothing
            @test !leaf.logical_resident[]
            @test Set(keys(two_site_cache.resident)) == Set((root.id,))
            @test !ispath(BS._factor_path(two_site_cache, leaf.id))

            configuration = Matrix{Int}(undef, 2, 1)
            log_probability = BS._sample_cached_shot!(
                MersenneTwister(0x6c65_6166),
                two_site,
                workspace,
                two_site_cache,
                configuration,
                1,
            )
            @test all(isone, configuration)
            @test iszero(log_probability)
        finally
            BS._cleanup_prefix_cache!(two_site_cache)
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
        8,
        3;
        disk=true,
        maxsize=2,
    )
    directory = cache.directory
    @test directory !== nothing
    @test isdir(directory)

    function register_node!(
        cache,
        parent,
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
        BS._admit_owned_prefix_factor!(cache, node, parent.id, factor)
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
        @test root.logical_resident[]
        @test BS._prefix_factor(cache, root.id) == root_reference

        warm_factor = factor_tensor(ComplexF64[1 2; 3 4])
        warm = register_node!(cache, root, -0.1, warm_factor)
        @test warm.logical_resident[]
        @test Set(keys(cache.resident)) == Set((root.id, warm.id))
        @test cache.min_resident_id == warm.id

        cold_factor = factor_tensor(ComplexF64[5 6; 7 8])
        cold = register_node!(cache, root, -0.2, cold_factor)
        @test !cold.logical_resident[]
        @test BS._prefix_factor(cache, cold.id) == cold_factor
        @test !cold.logical_resident[] # disk reads never promote

        # Admission is strict: an exact tie with the resident minimum stays cold.
        tied_factor = factor_tensor(ComplexF64[9 10; 11 12])
        tied = register_node!(cache, root, -0.1, tied_factor)
        @test !tied.logical_resident[]
        @test cache.min_resident_id == warm.id
        @test BS._prefix_factor(cache, tied.id) == tied_factor

        # A strictly more probable node evicts the unique resident minimum.
        hot_factor = factor_tensor(ComplexF64[13 14; 15 16])
        hot = register_node!(cache, root, -0.05, hot_factor)
        @test root.logical_resident[]
        @test hot.logical_resident[]
        @test !warm.logical_resident[]
        @test BS._prefix_factor(cache, warm.id) == warm_factor
        @test Set(keys(cache.resident)) == Set((root.id, hot.id))
        @test cache.min_resident_id == hot.id

        # A descendant of a cold parent is cold without changing the top-K set.
        resident_ids_before = Set(keys(cache.resident))
        descendant_factor = factor_tensor(ComplexF64[17 18; 19 20])
        descendant = register_node!(
            cache,
            cold,
            -0.3,
            descendant_factor,
        )
        @test !descendant.logical_resident[]
        @test Set(keys(cache.resident)) == resident_ids_before
        @test BS._prefix_factor(cache, descendant.id) == descendant_factor

        # All node metadata, including cold-node branch weights, remains resident.
        @test BS._published_prefix_node(cache, cold.id) === cold
        @test BS._published_prefix_node(cache, root.id).q == [0.6, 0.4]

        BS._publish_child_id!(root.children[1], warm.id)
        @test BS._child_id(root.children[1]) == warm.id
    finally
        BS._cleanup_prefix_cache!(cache)
    end

    @test !ispath(directory)
    @test isempty(cache.resident)
    # Without disk storage, maxsize is intentionally ignored and every factor
    # remains resident for the duration of the batch.
    memory_root_factor = factor_tensor(ones(1, 1))
    memory_cache = BS._new_prefix_cache(
        typeof(memory_root_factor),
        Float64,
        2,
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
            memory_cache, child, root.id, child_factor,
        )
        @test child.logical_resident[]
        @test memory_cache.directory === nothing
        @test length(memory_cache.resident) == 2
        @test BS._prefix_factor(memory_cache, child.id) == child_factor
    finally
        BS._cleanup_prefix_cache!(memory_cache)
    end
end
