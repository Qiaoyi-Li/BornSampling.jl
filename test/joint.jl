@testset "joint physical-purification sampling" begin
    @testset "rank-4 layout and complete dense probabilities" begin
        sampler = BS.BornSampler(residual_route_rank4_state(); purified=false)
        chain_length = length(sampler.state)
        @test chain_length == 2

        reference = normalize_weights!(dense_joint_weights(sampler.state))
        @test length(reference) == prod(
            plan.physical.fulldim * BS.purification_dimension(plan) for
            plan in sampler.plans
        )
        @test sum(values(reference)) ≈ 1 atol=8e-13

        physical_dimensions = map(plan -> plan.physical.fulldim, sampler.plans)
        purification_dimensions = map(BS.purification_dimension, sampler.plans)
        for physical in all_configurations(physical_dimensions)
            for purification in all_configurations(purification_dimensions)
                target = vcat(physical, purification)
                iszero(reference[Tuple(target)]) && continue
                uniforms = uniforms_for_joint_configuration(
                    reference,
                    target,
                    purification_dimensions,
                )
                configuration = Vector{Int}(undef, 2 * chain_length)
                log_probability = BS.bornsample!(
                    SequenceRNG(uniforms),
                    sampler,
                    configuration,
                )
                @test configuration == target
                @test exp(log_probability) ≈
                      reference[Tuple(target)] rtol=4e-11 atol=8e-13
            end
        end

        shot = BS.bornsample!(MersenneTwister(0x6a6f_696e), sampler)
        @test keys(shot) == (:configuration, :log_probability)
        @test length(shot.configuration) == 2 * chain_length
        @test all(
            site -> 1 <= shot.configuration[site] <= physical_dimensions[site],
            1:chain_length,
        )
        @test all(
            site -> 1 <= shot.configuration[chain_length + site] <=
                    purification_dimensions[site],
            1:chain_length,
        )
        @test exp(shot.log_probability) ≈
              reference[Tuple(shot.configuration)] rtol=4e-11 atol=8e-13

        @test_throws DimensionMismatch BS.bornsample!(
            MersenneTwister(0x7368_6f72),
            sampler,
            Vector{Int}(undef, chain_length),
        )
        @test_throws DimensionMismatch BS.bornsample!(
            MersenneTwister(0x6c6f_6e67),
            sampler,
            Vector{Int}(undef, 2 * chain_length + 1),
        )

        # The default purified mode remains a physical-only marginal.
        traced = BS.BornSampler(residual_route_rank4_state())
        traced_reference = normalize_weights!(dense_physical_weights(traced.state))
        for physical in all_configurations(physical_dimensions)
            physical_key = Tuple(physical)
            marginal = sum(reference) do (configuration, probability)
                configuration[1:chain_length] == physical_key ? probability : 0.0
            end
            @test marginal ≈ traced_reference[physical_key] rtol=4e-11 atol=8e-13

            traced_configuration = Vector{Int}(undef, chain_length)
            traced_log_probability = BS.bornsample!(
                SequenceRNG(uniforms_for_configuration(traced_reference, physical)),
                traced,
                traced_configuration,
            )
            @test traced_configuration == physical
            @test exp(traced_log_probability) ≈ marginal rtol=4e-11 atol=8e-13
        end
    end

    @testset "rank-4 batch scheduling and factor storage" begin
        sampler = BS.BornSampler(residual_route_rank4_state(); purified=false)
        chain_length = length(sampler.state)
        reference = normalize_weights!(dense_joint_weights(sampler.state))
        nshots = 40
        seed = 0x7061_6972

        workspace = first(sampler.workspaces)
        plan = first(sampler.plans)
        C = sampler.initial_factor
        BS._compute_weights!(workspace, C, plan)
        branch_count = plan.physical.fulldim * BS.purification_dimension(plan)
        selected = argmax(@view workspace.q[1:branch_count])
        selected_physical = div(selected - 1, BS.purification_dimension(plan)) + 1
        selected_purification = mod1(selected, BS.purification_dimension(plan))
        G = BS._build_selected_factor!(workspace, C, plan, selected)
        @test Int(TK.dim(TK.domain(G))) == 1
        @test size(workspace.route_output, 2) == 1
        @test norm(G)^2 ≈ workspace.q[selected] rtol=8e-13 atol=8e-13
        dense_local = convert(Array, sampler.state[1].A)
        dense_C = convert(Array, C)
        dense_selected = transpose(@view(
            dense_local[:, selected_physical, selected_purification, :],
        )) * dense_C
        @test workspace.q[selected] ≈ norm(dense_selected)^2 rtol=8e-13 atol=8e-13
        active_routes = filter(
            route -> TK.hasblock(C, route_left_sector(plan, route)),
            basis_routes(plan, selected_physical, selected_purification),
        )
        @test length(active_routes) == 1
        selected_route = only(active_routes)
        @test TK.block(G, route_right_sector(plan, selected_route)) ≈
              route_channel_result(
                  plan,
                  C,
                  selected_physical,
                  selected_purification,
                  selected_route,
              ) rtol=8e-13 atol=8e-13
        @test 1 <= selected_physical <= plan.physical.fulldim
        @test 1 <= selected_purification <= BS.purification_dimension(plan)
        Cnext = BS._advance_factor!(
            workspace,
            C,
            plan,
            selected,
            workspace.q[selected],
        )
        @test Int(TK.dim(TK.domain(Cnext))) == 1
        @test norm(Cnext) ≈ 1 rtol=8e-13 atol=8e-13

        serial = BS.bornsample!(
            MersenneTwister(seed),
            sampler,
            nshots;
            ntasks=1,
            disk=false,
        )
        parallel = BS.bornsample!(
            MersenneTwister(seed),
            sampler,
            nshots;
            ntasks=Threads.nthreads() + 5,
            disk=false,
        )
        disk = BS.bornsample!(
            MersenneTwister(seed),
            sampler,
            nshots;
            ntasks=Threads.nthreads() + 7,
            disk=true,
            maxsize=1,
        )

        @test size(serial.configuration) == (2 * chain_length, nshots)
        @test length(serial.log_probability) == nshots
        @test parallel.configuration == serial.configuration
        @test parallel.log_probability == serial.log_probability
        @test disk.configuration == serial.configuration
        @test disk.log_probability == serial.log_probability
        for shot in 1:nshots
            configuration = Tuple(@view serial.configuration[:, shot])
            @test exp(serial.log_probability[shot]) ≈
                  reference[configuration] rtol=4e-11 atol=8e-13
        end
    end

    @testset "mixed-rank MPO keeps outer sampling semantics" begin
        traced = BS.BornSampler(mixed_rank_mpo_state())
        joint = BS.BornSampler(mixed_rank_mpo_state(); purified=false)
        chain_length = length(traced.state)

        @test traced.state[1] isa FiniteMPS.MPSTensor{4}
        @test traced.state[2] isa FiniteMPS.MPSTensor{3}
        @test map(BS.purification_dimension, traced.plans) == [2, 1]

        traced_reference = normalize_weights!(dense_physical_weights(traced.state))
        joint_reference = normalize_weights!(dense_joint_weights(joint.state))
        physical_dimensions = map(plan -> plan.physical.fulldim, traced.plans)
        purification_dimensions = map(BS.purification_dimension, joint.plans)

        @test sum(values(traced_reference)) ≈ 1 atol=3e-13
        @test sum(values(joint_reference)) ≈ 1 atol=3e-13
        @test all(configuration -> configuration[chain_length + 2] == 1,
                  keys(joint_reference))

        for physical in all_configurations(physical_dimensions)
            key = Tuple(physical)
            marginal = sum(joint_reference) do (configuration, probability)
                configuration[1:chain_length] == key ? probability : 0.0
            end
            @test marginal ≈ traced_reference[key] rtol=2e-12 atol=3e-13

            configuration = Vector{Int}(undef, chain_length)
            log_probability = BS.bornsample!(
                SequenceRNG(uniforms_for_configuration(traced_reference, physical)),
                traced,
                configuration,
            )
            @test configuration == physical
            @test exp(log_probability) ≈ traced_reference[key] rtol=2e-12 atol=3e-13
        end

        for (target_tuple, probability) in joint_reference
            iszero(probability) && continue
            target = collect(target_tuple)
            configuration = Vector{Int}(undef, 2 * chain_length)
            log_probability = BS.bornsample!(
                SequenceRNG(uniforms_for_joint_configuration(
                    joint_reference,
                    target,
                    purification_dimensions,
                )),
                joint,
                configuration,
            )
            @test configuration == target
            @test configuration[chain_length + 2] == 1
            @test exp(log_probability) ≈ probability rtol=2e-12 atol=3e-13
        end

        # In traced MPO mode, the first rank-4 site produces a multi-column
        # factor. The following rank-3 site still compresses it according to
        # the global MPO mode, despite its synthetic purification dimension.
        traced_workspace = first(traced.workspaces)
        first_plan, second_plan = traced.plans
        factor = traced.initial_factor
        BS._compute_weights!(traced_workspace, factor, first_plan)
        selected = argmax(@view traced_workspace.q[1:first_plan.physical.fulldim])
        factor = advance_test_factor!(
            traced_workspace,
            factor,
            first_plan,
            selected,
            traced_workspace.q[selected],
        )
        @test Int(TK.dim(TK.domain(factor))) > 1

        BS._compute_weights!(traced_workspace, factor, second_plan)
        selected = argmax(@view traced_workspace.q[1:second_plan.physical.fulldim])
        uncompressed = BS._build_selected_factor!(
            traced_workspace, factor, second_plan, selected,
        )
        @test Int(TK.dim(TK.domain(uncompressed))) >
              Int(TK.dim(second_plan.residual_right.space))
        factor = advance_test_factor!(
            traced_workspace,
            factor,
            second_plan,
            selected,
            traced_workspace.q[selected],
        )
        @test Int(TK.dim(TK.domain(factor))) <=
              Int(TK.dim(second_plan.residual_right.space))
        @test norm(factor) ≈ 1 rtol=3e-13 atol=3e-13

        # Joint MPO mode keeps one pure factor column at both local ranks.
        joint_workspace = first(joint.workspaces)
        factor = joint.initial_factor
        @test size(joint_workspace.route_output, 2) == 1
        for plan in joint.plans
            BS._compute_weights!(joint_workspace, factor, plan)
            count = plan.physical.fulldim * BS.purification_dimension(plan)
            selected = argmax(@view joint_workspace.q[1:count])
            factor = BS._advance_factor!(
                joint_workspace,
                factor,
                plan,
                selected,
                joint_workspace.q[selected],
            )
            @test Int(TK.dim(TK.domain(factor))) == 1
        end

        # Mixed local ranks must use the same deterministic prefix tree in
        # serial, parallel, memory-resident, and disk-spilled batch schedules.
        nshots = 20
        seed = 0x6d69_7862
        for (sampler, expected_rows) in (
            (traced, chain_length),
            (joint, 2 * chain_length),
        )
            serial = BS.bornsample!(
                MersenneTwister(seed),
                sampler,
                nshots;
                ntasks=1,
                disk=false,
            )
            parallel = BS.bornsample!(
                MersenneTwister(seed),
                sampler,
                nshots;
                ntasks=Threads.nthreads() + 3,
                disk=false,
            )
            disk = BS.bornsample!(
                MersenneTwister(seed),
                sampler,
                nshots;
                ntasks=Threads.nthreads() + 5,
                disk=true,
                maxsize=1,
            )

            @test size(serial.configuration) == (expected_rows, nshots)
            @test parallel.configuration == serial.configuration
            @test parallel.log_probability == serial.log_probability
            @test disk.configuration == serial.configuration
            @test disk.log_probability == serial.log_probability
        end
        joint_batch = BS.bornsample!(
            MersenneTwister(seed),
            joint,
            nshots;
            ntasks=1,
            disk=false,
        )
        @test all(==(1), @view joint_batch.configuration[chain_length + 2, :])
    end

    @testset "MPS ignores the MPO purification switch" begin
        source = rank3_state(length=4, bonddim=3)
        default_sampler = BS.BornSampler(deepcopy(source))
        joint_flag_sampler = BS.BornSampler(deepcopy(source); purified=false)
        chain_length = length(source)
        seed = 0x7261_6e6b

        default_configuration = Vector{Int}(undef, chain_length)
        flagged_configuration = Vector{Int}(undef, chain_length)
        default_log_probability = BS.bornsample!(
            MersenneTwister(seed),
            default_sampler,
            default_configuration,
        )
        flagged_log_probability = BS.bornsample!(
            MersenneTwister(seed),
            joint_flag_sampler,
            flagged_configuration,
        )
        @test flagged_configuration == default_configuration
        @test flagged_log_probability == default_log_probability
        @test length(flagged_configuration) == chain_length

        default_batch = BS.bornsample!(
            MersenneTwister(seed),
            default_sampler,
            24;
            ntasks=Threads.nthreads() + 2,
            disk=true,
            maxsize=1,
        )
        flagged_batch = BS.bornsample!(
            MersenneTwister(seed),
            joint_flag_sampler,
            24;
            ntasks=Threads.nthreads() + 2,
            disk=true,
            maxsize=1,
        )
        @test flagged_batch.configuration == default_batch.configuration
        @test flagged_batch.log_probability == default_batch.log_probability
        @test size(flagged_batch.configuration, 1) == chain_length

        workspace = first(joint_flag_sampler.workspaces)
        plan = first(joint_flag_sampler.plans)
        C = joint_flag_sampler.initial_factor
        BS._compute_weights!(workspace, C, plan)
        selected = argmax(@view workspace.q[1:plan.physical.fulldim])
        G = BS._build_selected_factor!(workspace, C, plan, selected)
        @test Int(TK.dim(TK.domain(G))) == 1
        @test size(workspace.route_output, 2) == 1
    end
end
