"Force the public `[x..., b]` layout while the boundary is drawn before the sites."
function uniforms_for_mps_boundary_configuration(probabilities, target; purified)
    purified && return uniforms_for_configuration(probabilities, target)
    boundary_first = Dict(
        Tuple(vcat(last(configuration), collect(configuration[1:(end - 1)]))) =>
        probability for (configuration, probability) in probabilities
    )
    return uniforms_for_configuration(
        boundary_first,
        vcat(last(target), target[1:(end - 1)]),
    )
end

function charged_spin_one_left_state()
    sector_type = TK.Irrep[TK.:×(TK.U₁, TK.SU₂)]
    physical = TK.GradedSpace(sector_type(1, 1 // 2) => 1)
    left = TK.GradedSpace(sector_type(-2, 1) => 1)
    bond = TK.GradedSpace(sector_type(-1, 1 // 2) => 1)
    right = TK.GradedSpace(sector_type(0, 0) => 1)
    return FiniteMPS.MPS(FiniteMPS.MPSTensor.([
        TK.randn(ComplexF64, ⊗(left, physical), bond),
        TK.randn(ComplexF64, ⊗(bond, physical), right),
    ]))
end

@testset "MPS left boundary as a global purification leg" begin
    @testset "non-Abelian trace and complete joint probabilities: $name" for
        (name, source) in (
            ("SU(2)", nontrivial_irrep_left_state()),
            ("U(1) x SU(2)", charged_spin_one_left_state()),
        )
        chain_length = length(source)
        boundary_dimension = Int(TK.dim(BS.leftspace(first(source.A))))
        @test boundary_dimension == 3
        traced_reference = normalize_weights!(dense_mps_boundary_weights(source))
        joint_reference = normalize_weights!(dense_mps_boundary_weights(
            source; purified=false,
        ))
        @test length(traced_reference) == 4
        @test length(joint_reference) == 12
        @test sum(values(traced_reference)) ≈ 1 atol=3e-13
        @test sum(values(joint_reference)) ≈ 1 atol=3e-13

        for (physical, probability) in traced_reference
            marginal = sum(joint_reference) do (configuration, weight)
                configuration[1:chain_length] == physical ? weight : 0.0
            end
            @test marginal ≈ probability atol=3e-13
        end
        for boundary in 1:boundary_dimension
            marginal = sum(joint_reference) do (configuration, probability)
                last(configuration) == boundary ? probability : 0.0
            end
            @test marginal ≈ 1 / boundary_dimension atol=3e-13
        end

        for purified in (true, false)
            sampler = BS.BornSampler(deepcopy(source); purified)
            reference = purified ? traced_reference : joint_reference
            configuration_length = chain_length + Int(!purified)
            for (target_key, probability) in reference
                # Zero branches have no interval that a uniform draw can select.
                iszero(probability) && continue
                target = collect(target_key)
                uniforms = uniforms_for_mps_boundary_configuration(
                    reference, target; purified,
                )
                configuration = Vector{Int}(undef, configuration_length)
                log_probability = BS.bornsample!(
                    SequenceRNG(uniforms), sampler, configuration,
                )
                @test configuration == target
                @test exp(log_probability) ≈ probability rtol=3e-11 atol=3e-13
            end

            seed = 0x6d70_7362
            shot = BS.bornsample!(MersenneTwister(seed), sampler)
            direct = BS.bornsample!(
                MersenneTwister(seed), deepcopy(source); purified,
            )
            @test length(shot.configuration) == configuration_length
            @test direct.configuration == shot.configuration
            @test direct.log_probability ≈ shot.log_probability atol=3e-13
            @test exp(shot.log_probability) ≈
                  reference[Tuple(shot.configuration)] rtol=3e-11 atol=3e-13
            @test_throws DimensionMismatch BS.bornsample!(
                MersenneTwister(seed),
                sampler,
                Vector{Int}(undef, chain_length + Int(purified)),
            )
        end
    end

    @testset "traced boundary rank compresses without changing the density" begin
        sampler = BS.BornSampler(nontrivial_irrep_left_state())
        plan = first(sampler.plans)
        workspace = first(sampler.workspaces)
        initial = sampler.initial_factor
        initial_array = convert(Array, initial)
        @test size(initial_array) == (3, 3)
        @test initial_array * adjoint(initial_array) ≈
              Matrix{ComplexF64}(I, 3, 3) / 3 atol=3e-13
        BS._compute_weights!(workspace, initial, plan)
        tensor = convert(Array, first(sampler.state.A).A)
        for selected in 1:plan.physical.fulldim
            K = transpose(@view tensor[:, selected, :])
            expected_density = K * adjoint(K) / 3
            expected_density /= real(tr(expected_density))
            next_factor = BS._advance_factor!(
                workspace, initial, plan, selected, workspace.q[selected],
            )
            next_array = convert(Array, next_factor)
            @test size(next_array, 1) == 2
            @test size(next_array, 2) <= 2
            @test norm(next_factor) ≈ 1 atol=3e-13
            @test next_array * adjoint(next_array) ≈
                  expected_density rtol=3e-11 atol=3e-13
        end
    end

    @testset "single-site non-Abelian boundary reaches the terminal layer" begin
        physical = FiniteMPS.SU2Spin.pspace
        left = TK.Rep[TK.SU₂](1 // 2 => 1)
        right = TK.Rep[TK.SU₂](0 => 1)
        source = FiniteMPS.MPS([
            FiniteMPS.MPSTensor(TK.randn(ComplexF64, ⊗(left, physical), right)),
        ])
        for purified in (true, false)
            sampler = BS.BornSampler(deepcopy(source); purified)
            reference = normalize_weights!(dense_mps_boundary_weights(source; purified))
            @test length(reference) == (purified ? 2 : 4)
            for (target_key, probability) in reference
                iszero(probability) && continue
                target = collect(target_key)
                shot = BS.bornsample!(
                    SequenceRNG(uniforms_for_mps_boundary_configuration(
                        reference, target; purified,
                    )),
                    sampler,
                )
                @test shot.configuration == target
                @test length(shot.configuration) == 1 + Int(!purified)
                @test exp(shot.log_probability) ≈ probability atol=3e-13
            end
            for disk in (false, true)
                batch = BS.bornsample!(
                    MersenneTwister(0x6f6e_6573), sampler, 8;
                    ntasks=2, disk, maxsize=1,
                )
                @test size(batch.configuration) == (1 + Int(!purified), 8)
                for shot in eachindex(batch.log_probability)
                    @test exp(batch.log_probability[shot]) ≈
                          reference[Tuple(@view batch.configuration[:, shot])]
                end
            end
        end
    end

    @testset "boundary probabilities survive prefix caching and task scheduling" begin
        source = nontrivial_irrep_left_state()
        nshots = 24
        seed = 0x626f_756e
        for purified in (true, false)
            sampler = BS.BornSampler(deepcopy(source); purified)
            reference = normalize_weights!(dense_mps_boundary_weights(source; purified))
            configuration_length = length(source) + Int(!purified)
            seed_rng = MersenneTwister(seed)
            shot_seeds = [rand(seed_rng, UInt64) for _ in 1:nshots]
            expected_configuration = Matrix{Int}(undef, configuration_length, nshots)
            expected_log_probability = Vector{Float64}(undef, nshots)
            for shot in 1:nshots
                expected_log_probability[shot] = BS.bornsample!(
                    Random.Xoshiro(shot_seeds[shot]),
                    sampler,
                    @view(expected_configuration[:, shot]),
                )
                @test exp(expected_log_probability[shot]) ≈
                      reference[Tuple(@view expected_configuration[:, shot])]
            end
            expected_next_caller_value = rand(seed_rng, UInt64)

            baseline = nothing
            for disk in (false, true), ntasks in (1, Threads.nthreads() + 2)
                caller_rng = MersenneTwister(seed)
                result = BS.bornsample!(
                    caller_rng, sampler, nshots; ntasks, disk, maxsize=1,
                )
                @test size(result.configuration) == (configuration_length, nshots)
                @test result.configuration == expected_configuration
                @test result.log_probability ≈
                      expected_log_probability rtol=3e-11 atol=3e-13
                @test rand(caller_rng, UInt64) == expected_next_caller_value
                if baseline === nothing
                    baseline = result
                else
                    @test result.configuration == baseline.configuration
                    @test result.log_probability == baseline.log_probability
                end
            end

            direct = BS.bornsample!(
                MersenneTwister(seed), deepcopy(source), nshots;
                purified, ntasks=2, disk=true, maxsize=1,
            )
            @test direct.configuration == expected_configuration
            @test direct.log_probability ≈
                  expected_log_probability rtol=3e-11 atol=3e-13
            empty_batch = BS.bornsample!(
                MersenneTwister(seed), sampler, 0; disk=true, maxsize=1,
            )
            @test size(empty_batch.configuration) == (configuration_length, 0)
            @test isempty(empty_batch.log_probability)
        end
    end

    @testset "singleton boundary layout and supported input contract" begin
        source = rank3_state(length=3, bonddim=2)
        joint = BS.BornSampler(deepcopy(source); purified=false)
        reference = normalize_weights!(dense_mps_boundary_weights(source; purified=false))
        for (target_key, probability) in reference
            iszero(probability) && continue
            target = collect(target_key)
            shot = BS.bornsample!(
                SequenceRNG(uniforms_for_mps_boundary_configuration(
                    reference, target; purified=false,
                )),
                joint,
            )
            @test shot.configuration == target
            @test length(shot.configuration) == length(source) + 1
            @test last(shot.configuration) == 1
            @test exp(shot.log_probability) ≈ probability rtol=3e-11 atol=3e-13
        end
        result = BS.bornsample!(
            MersenneTwister(0x7369_6e67), joint, 12;
            ntasks=Threads.nthreads() + 2, disk=true, maxsize=1,
        )
        @test size(result.configuration) == (length(source) + 1, 12)
        @test all(==(1), @view result.configuration[end, :])
        for shot in eachindex(result.log_probability)
            @test exp(result.log_probability[shot]) ≈
                  reference[Tuple(@view result.configuration[:, shot])]
        end

        # Each original sector must have multiplicity one, and there must be
        # exactly one original sector, even if projection merges residual charges.
        physical = FiniteMPS.SU2Spin.pspace
        left = TK.Rep[TK.SU₂](0 => 1, 1 => 1)
        bond = TK.Rep[TK.SU₂](1 // 2 => 1)
        right = TK.Rep[TK.SU₂](0 => 1)
        multi_sector = FiniteMPS.MPS(FiniteMPS.MPSTensor.([
            TK.randn(ComplexF64, ⊗(left, physical), bond),
            TK.randn(ComplexF64, ⊗(bond, physical), right),
        ]))
        for purified in (true, false)
            @test_throws ArgumentError BS.BornSampler(nontrivial_left_state(); purified)
            @test_throws ArgumentError BS.BornSampler(deepcopy(multi_sector); purified)
            @test_throws ArgumentError BS.BornSampler(
                nontrivial_irrep_left_state();
                purified, left_boundary=ComplexF64[1, 0, 0],
            )
            @test_throws ArgumentError BS.BornSampler(
                deepcopy(source); purified, left_boundary=ComplexF64[1],
            )
        end
    end
end
