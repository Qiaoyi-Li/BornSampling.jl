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

    @testset "MPO probabilities across task and storage configurations" begin
        for state in (rank4_state(), mixed_rank_three_site_mpo_state())
            sampler = BS.BornSampler(state)
            reference = normalize_weights!(dense_physical_weights(sampler.state))
            nshots = 24
            seed = 0x6c61_7965
            execution_options = (
                (; ntasks=1, disk=false),
                (; ntasks=Threads.nthreads() + 1, disk=false),
                (; ntasks=Threads.nthreads() + 1, disk=true, maxsize=1),
            )
            directories_before = Set(filter(
                path -> startswith(basename(path), "BornSampling-prefix-"),
                readdir(tempdir(); join=true),
            ))
            baseline = nothing
            for options in execution_options
                result = BS.bornsample!(MersenneTwister(seed), sampler, nshots; options...)
                if baseline === nothing
                    baseline = result
                else
                    @test result == baseline
                end
                for shot in eachindex(result.log_probability)
                    configuration = Tuple(@view result.configuration[:, shot])
                    @test exp(result.log_probability[shot]) ≈
                          reference[configuration] rtol=3e-11 atol=5e-13
                end
            end
            directories_after = Set(filter(
                path -> startswith(basename(path), "BornSampling-prefix-"),
                readdir(tempdir(); join=true),
            ))
            @test directories_after == directories_before
        end
    end
end
