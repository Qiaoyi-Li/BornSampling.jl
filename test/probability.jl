@testset "short-chain complete probabilities" begin
    @testset "rank 3" begin
        sampler = BS.BornSampler(rank3_state(length=3, bonddim=3))
        reference = normalize_weights!(dense_physical_weights(sampler.state))
        @test sum(values(reference)) ≈ 1 atol=2e-13

        for physical in all_configurations(map(plan -> plan.physical.fulldim, sampler.plans))
            log_probability = sequential_log_probability!(sampler, physical)
            @test exp(log_probability) ≈ reference[Tuple(physical)] rtol=2e-11 atol=2e-13

            forced = Vector{Int}(undef, length(physical))
            uniforms = uniforms_for_configuration(reference, physical)
            log_probability = BS.bornsample!(SequenceRNG(uniforms), sampler, forced)
            @test forced == physical
            @test exp(log_probability) ≈ reference[Tuple(physical)] rtol=2e-11 atol=2e-13
        end
    end

    @testset "rank 4 with explicit purification trace" begin
        sampler = BS.BornSampler(rank4_state())
        reference = normalize_weights!(dense_physical_weights(sampler.state))
        @test sum(values(reference)) ≈ 1 atol=5e-13

        physical_dimensions = map(plan -> plan.physical.fulldim, sampler.plans)
        for physical in all_configurations(physical_dimensions)
            log_probability = sequential_log_probability!(sampler, physical)
            @test exp(log_probability) ≈ reference[Tuple(physical)] rtol=3e-11 atol=5e-13

            if !iszero(reference[Tuple(physical)])
                forced = Vector{Int}(undef, length(physical))
                uniforms = uniforms_for_configuration(reference, physical)
                log_probability = BS.bornsample!(SequenceRNG(uniforms), sampler, forced)
                @test forced == physical
                @test exp(log_probability) ≈ reference[Tuple(physical)] rtol=3e-11 atol=5e-13
            end
        end
    end
end
