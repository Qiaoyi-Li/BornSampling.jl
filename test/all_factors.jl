@testset "traced-MPO all-physical factor bank" begin
    @testset "constructs every branch exactly once" begin
        sampler = BS.BornSampler(residual_route_rank4_state())
        workspace = first(sampler.workspaces)
        factor = sampler.initial_factor
        original_plan = first(sampler.plans)

        reduced_views = Ref(0)
        counted_plan = view_counting_plan(original_plan, reduced_views)
        factors = BS._compute_weights_and_factors!(
            workspace,
            factor,
            counted_plan,
        )

        @test length(factors) == original_plan.physical.fulldim
        @test reduced_views[] == all_branch_transition_count(counted_plan, factor)

        reference_workspace = BS._clone_workspace(workspace)
        BS._compute_weights!(reference_workspace, factor, original_plan)
        branch_count = original_plan.physical.fulldim
        @test workspace.q[1:branch_count] ≈
              reference_workspace.q[1:branch_count] rtol=8e-13 atol=8e-13

        for selected in 1:branch_count
            reference = BS._build_selected_factor!(
                reference_workspace,
                factor,
                original_plan,
                selected,
            )
            @test factors[selected] * adjoint(factors[selected]) ≈
                  reference * adjoint(reference) rtol=1e-12 atol=1e-12
            @test norm(factors[selected])^2 ≈
                  workspace.q[selected] rtol=8e-13 atol=8e-13
        end
    end

    @testset "different prefix children consume the prebuilt bank" begin
        for disk in (false, true)
            sampler = BS.BornSampler(residual_route_rank4_state())
            workspace = first(sampler.workspaces)
            factor = sampler.initial_factor
            original_plan = first(sampler.plans)
            reduced_views = Ref(0)
            counted_plan = view_counting_plan(original_plan, reduced_views)
            sampler.plans[1] = counted_plan

            Environment = BS._prefix_environment_type(sampler, typeof(factor))
            cache = BS._new_prefix_cache(
                Environment,
                typeof(factor),
                eltype(workspace.q),
                8,
                length(sampler.state);
                disk=disk,
                maxsize=1,
            )
            try
                BS._initialize_prefix_cache!(sampler, workspace, cache)
                root = BS._published_prefix_node(cache, 1)
                expected_views = all_branch_transition_count(counted_plan, factor)
                @test reduced_views[] == expected_views

                bundle = cache.resident[root.id]
                @test bundle isa BS.TracedBranchBundle{typeof(factor)}
                for selected in eachindex(root.q)
                    stored = bundle.factors[selected]
                    @test stored !== nothing
                    @test norm(stored)^2 ≈ root.q[selected] rtol=8e-13 atol=8e-13
                end

                positive = findall(>(zero(eltype(root.q))), root.q)
                @test length(positive) >= 2
                selected_children = positive[1:2]

                for selected in selected_children
                    reference_workspace = BS._clone_workspace(workspace)
                    reference_factor = advance_test_factor!(
                        reference_workspace,
                        factor,
                        original_plan,
                        selected,
                        root.q[selected],
                    )
                    next_plan = sampler.plans[2]
                    BS._compute_weights!(
                        reference_workspace,
                        reference_factor,
                        next_plan,
                    )
                    next_count = BS._outcome_count(reference_workspace, next_plan)
                    reference_q = copy(@view reference_workspace.q[1:next_count])

                    stored = bundle.factors[selected]
                    reference_G = BS._build_selected_factor!(
                        reference_workspace,
                        factor,
                        original_plan,
                        selected,
                    )
                    @test stored * adjoint(stored) ≈
                          reference_G * adjoint(reference_G) rtol=1e-12 atol=1e-12

                    child = BS._get_or_build_prefix_child!(
                        sampler,
                        workspace,
                        cache,
                        root,
                        selected,
                        1,
                    )
                    @test reduced_views[] == expected_views
                    @test bundle.factors[selected] === nothing
                    @test child.q ≈ reference_q rtol=2e-12 atol=2e-12
                    @test child.log_probability ≈
                          log(root.q[selected]) - log(sum(root.q)) rtol=8e-13 atol=8e-13

                    # Re-reading an already published edge must neither rebuild
                    # its factor nor consume another bank entry.
                    same_child = BS._get_or_build_prefix_child!(
                        sampler,
                        workspace,
                        cache,
                        root,
                        selected,
                        1,
                    )
                    @test same_child === child
                    @test reduced_views[] == expected_views
                end
            finally
                BS._cleanup_prefix_cache!(cache)
            end
        end
    end

    @testset "one traced edge is built once under contention" begin
        sampler = BS.BornSampler(residual_route_rank4_state())
        contenders = Threads.nthreads() + 11
        BS._ensure_workspaces!(sampler, contenders)
        workspace = first(sampler.workspaces)
        factor = sampler.initial_factor
        original_plan = first(sampler.plans)
        reduced_views = Ref(0)
        counted_plan = view_counting_plan(original_plan, reduced_views)
        sampler.plans[1] = counted_plan

        Environment = BS._prefix_environment_type(sampler, typeof(factor))
        cache = BS._new_prefix_cache(
            Environment,
            typeof(factor),
            eltype(workspace.q),
            contenders,
            length(sampler.state);
            disk=false,
            maxsize=contenders,
        )
        try
            BS._initialize_prefix_cache!(sampler, workspace, cache)
            root = BS._published_prefix_node(cache, 1)
            selected = argmax(root.q)
            expected_views = all_branch_transition_count(counted_plan, factor)
            child_ids = Vector{Int}(undef, contenders)
            @sync for worker in 1:contenders
                Threads.@spawn begin
                    child_ids[worker] = BS._get_or_build_prefix_child!(
                        sampler,
                        sampler.workspaces[worker],
                        cache,
                        root,
                        selected,
                        1,
                    ).id
                end
            end
            @test all(==(first(child_ids)), child_ids)
            @test cache.next_node_id[] == 2
            @test reduced_views[] == expected_views
            @test cache.resident[root.id].factors[selected] === nothing
        finally
            BS._cleanup_prefix_cache!(cache)
        end
    end

    @testset "cold traced branch bank round-trips through disk" begin
        sampler = BS.BornSampler(mixed_rank_three_site_mpo_state())
        workspace = first(sampler.workspaces)
        factor = sampler.initial_factor
        Factor = typeof(factor)
        Environment = BS._prefix_environment_type(sampler, Factor)
        cache = BS._new_prefix_cache(
            Environment,
            Factor,
            eltype(workspace.q),
            8,
            length(sampler.state);
            disk=true,
            maxsize=1,
        )
        directory = cache.directory::String
        try
            BS._initialize_prefix_cache!(sampler, workspace, cache)
            root = BS._published_prefix_node(cache, 1)
            selected_root = argmax(root.q)
            child = BS._get_or_build_prefix_child!(
                sampler,
                workspace,
                cache,
                root,
                selected_root,
                1,
            )
            @test !child.logical_resident[]
            @test child.branch_factor_spaces !== nothing

            reference_workspace = BS._clone_workspace(workspace)
            reference_factor = advance_test_factor!(
                reference_workspace,
                factor,
                sampler.plans[1],
                selected_root,
                root.q[selected_root],
            )
            positive = findall(>(zero(eltype(child.q))), child.q)
            @test length(positive) >= 2
            for selected in positive[1:2]
                path = BS._branch_factor_path(cache, child.id, selected)
                @test ispath(path)
                stored = BS._take_traced_branch!(cache, child, selected)
                @test !ispath(path)
                reference = BS._build_selected_factor!(
                    reference_workspace,
                    reference_factor,
                    sampler.plans[2],
                    selected,
                )
                @test stored * adjoint(stored) ≈
                      reference * adjoint(reference) rtol=2e-12 atol=2e-12
                @test norm(stored)^2 ≈ child.q[selected] rtol=1e-12 atol=1e-12
            end
        finally
            BS._cleanup_prefix_cache!(cache)
        end
        @test !ispath(directory)
    end
end
