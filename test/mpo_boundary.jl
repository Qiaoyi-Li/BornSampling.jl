"Small SU(2) MPOs with one spin-one left multiplet and a singlet right boundary."
function nontrivial_mpo_boundary_state(kind::Symbol)
    rng = MersenneTwister(0x6d70_6f71)
    physical = FiniteMPS.SU2Spin.pspace
    left = TK.Rep[TK.SU₂](1 => 1)
    right = TK.Rep[TK.SU₂](0 => 1)
    tensor3(l, r) = TK.randn(rng, ComplexF64, ⊗(l, physical), r)
    tensor4(l, y, r) = TK.randn(rng, ComplexF64, ⊗(l, physical), ⊗(y, r))
    tensors = if kind === :rank4
        bond = TK.Rep[TK.SU₂](0 => 1, 1 => 1)
        [tensor4(left, physical, bond), tensor4(bond, physical, right)]
    elseif kind === :single_site
        [tensor4(left, physical, right)]
    else
        bond = TK.Rep[TK.SU₂](1 // 2 => 1)
        if kind === :first_rank3
            [tensor3(left, bond), tensor4(bond, left, right)]
        elseif kind === :first_rank4
            [tensor4(left, left, bond), tensor3(bond, right)]
        elseif kind === :singleton_purification
            [tensor4(left, right, bond), tensor3(bond, right)]
        elseif kind === :rank3_only
            [tensor3(left, bond), tensor3(bond, right)]
        else
            error("unknown MPO boundary fixture: $kind")
        end
    end
    return FiniteMPS.MPO(FiniteMPS.MPSTensor.(tensors))
end

@testset "MPO global boundary and local purification legs" begin
    @testset "complete dense distributions and layout: $kind" for kind in (
        :rank4, :first_rank3, :first_rank4, :singleton_purification,
        :rank3_only, :single_site,
    )
        source = nontrivial_mpo_boundary_state(kind)
        chain_length = length(source)
        layout = mpo_joint_layout(source)
        traced_reference = normalize_weights!(dense_mpo_boundary_weights(source))
        joint_reference = normalize_weights!(dense_mpo_boundary_weights(source; purified=false))
        @test Int(TK.dim(BS.leftspace(source[1]))) == 3
        for (physical, probability) in traced_reference
            marginal = sum(joint_reference) do (configuration, weight)
                configuration[1:chain_length] == physical ? weight : 0.0
            end
            @test marginal ≈ probability atol=8e-13
        end
        for boundary in 1:3
            marginal = sum(joint_reference) do (configuration, probability)
                last(configuration) == boundary ? probability : 0.0
            end
            @test marginal ≈ 1 / 3 atol=8e-13
        end

        for purified in (true, false)
            sampler = BS.BornSampler(deepcopy(source); purified)
            reference = purified ? traced_reference : joint_reference
            output_length = purified ? chain_length : 2 * chain_length + 1
            for (target_key, probability) in reference
                iszero(probability) && continue
                target = collect(target_key)
                uniforms = purified ? uniforms_for_configuration(reference, target) :
                    uniforms_for_mpo_boundary_configuration(reference, target, source)
                configuration = Vector{Int}(undef, output_length)
                log_probability = BS.bornsample!(SequenceRNG(uniforms), sampler, configuration)
                @test configuration == target
                @test exp(log_probability) ≈ probability rtol=4e-11 atol=8e-13
            end
            seed = 0x7175_616e
            shot = BS.bornsample!(MersenneTwister(seed), sampler)
            direct = BS.bornsample!(MersenneTwister(seed), deepcopy(source); purified)
            @test length(shot.configuration) == output_length
            @test direct.configuration == shot.configuration
            @test direct.log_probability ≈ shot.log_probability atol=8e-13
            @test exp(shot.log_probability) ≈ reference[Tuple(shot.configuration)]
            if !purified
                @test 1 <= last(shot.configuration) <= 3
                for site in 1:chain_length
                    y = shot.configuration[chain_length + site]
                    if site in layout.purification_sites
                        @test 1 <= y <= layout.purification_dimensions[site]
                    else
                        @test y == 0
                    end
                end
                if kind === :singleton_purification
                    @test shot.configuration[(chain_length + 1):(2 * chain_length)] == [1, 0]
                elseif kind === :rank3_only
                    @test all(iszero, shot.configuration[(chain_length + 1):(2 * chain_length)])
                end
            end
        end
    end

    @testset "traced boundary and local purification preserve dense density" begin
        sampler = BS.BornSampler(nontrivial_mpo_boundary_state(:rank4))
        workspace = first(sampler.workspaces)
        plan = first(sampler.plans)
        initial = convert(Array, sampler.initial_factor)
        @test size(initial) == (3, 3)
        @test initial * adjoint(initial) ≈ Matrix{ComplexF64}(I, 3, 3) / 3 atol=8e-13
        factors = BS._compute_weights_and_factors!(workspace, sampler.initial_factor, plan)
        tensor = convert(Array, sampler.state[1].A)
        for physical in 1:size(tensor, 2)
            density = zeros(ComplexF64, size(tensor, 4), size(tensor, 4))
            for purification in 1:size(tensor, 3)
                K = transpose(@view tensor[:, physical, purification, :])
                density += K * adjoint(K) / 3
            end
            weight = real(tr(density))
            @test workspace.q[physical] ≈ weight atol=8e-13
            factor = BS._advance_built_traced_factor!(factors[physical], plan, weight)
            array = convert(Array, factor)
            @test size(array, 2) <= size(array, 1)
            @test norm(factor) ≈ 1 atol=8e-13
            @test array * adjoint(array) ≈ density / weight rtol=4e-11 atol=8e-13
        end
    end

    @testset "boundary probabilities across cached batches: $kind" for kind in (
        :rank4, :first_rank3, :first_rank4, :single_site,
    )
        source = nontrivial_mpo_boundary_state(kind)
        nshots = 20
        seed = 0x6361_6368
        for purified in (true, false)
            sampler = BS.BornSampler(deepcopy(source); purified)
            reference = normalize_weights!(dense_mpo_boundary_weights(source; purified))
            output_length = purified ? length(source) : 2 * length(source) + 1
            seed_rng = MersenneTwister(seed)
            shot_seeds = [rand(seed_rng, UInt64) for _ in 1:nshots]
            expected_configuration = Matrix{Int}(undef, output_length, nshots)
            expected_log_probability = Vector{Float64}(undef, nshots)
            for shot in 1:nshots
                expected_log_probability[shot] = BS.bornsample!(
                    Random.Xoshiro(shot_seeds[shot]), sampler,
                    @view(expected_configuration[:, shot]),
                )
            end
            for disk in (false, true), ntasks in (1, Threads.nthreads() + 2)
                result = BS.bornsample!(
                    MersenneTwister(seed), sampler, nshots; disk, ntasks, maxsize=1,
                )
                @test size(result.configuration) == (output_length, nshots)
                @test result.configuration == expected_configuration
                @test result.log_probability ≈ expected_log_probability atol=8e-13
                for shot in 1:nshots
                    @test exp(result.log_probability[shot]) ≈
                          reference[Tuple(@view result.configuration[:, shot])]
                end
            end
            empty_batch = BS.bornsample!(MersenneTwister(seed), sampler, 0; disk=true)
            @test size(empty_batch.configuration) == (output_length, 0)
            @test isempty(empty_batch.log_probability)
        end
    end
end
