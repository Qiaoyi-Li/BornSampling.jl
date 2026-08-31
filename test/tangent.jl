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

function tangent_temporary_directories(prefix)
    return Set(filter(readdir(tempdir(); join=true)) do path
        startswith(basename(path), prefix)
    end)
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
            ordinary_joint_result = BS.bornsample!(
                MersenneTwister(0x6a6f_696e),
                ordinary_joint,
                8;
                ntasks=2,
            )
            @test tangent_joint_result.configuration ==
                  ordinary_joint_result.configuration
            @test tangent_joint_result.log_probability ≈
                  ordinary_joint_result.log_probability atol=2e-12 rtol=2e-12

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
        @test size(joint.configuration) == (3, 12)
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
            @test (first(sampler.plans) isa BS.TangentGlobalQPlan) == !purified
            @test length(sampler.plans) == length(state) + Int(!purified)
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
                    site = purified ? layer : layer - 1
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
                  (purified ? 3 : 4, 16)
            @test all(isfinite, serial.log_probability)
            if !purified
                @test all(q -> 1 <= q <= 2, serial.configuration[4, :])
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
        @test size(joint.configuration) == (6, 4)
        @test all(view(joint.configuration, 5:6, :) .== 1)
        @test all(isfinite, joint.log_probability)
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
            step = (plan::BS.TangentSitePlan).step
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
        ordinary_result = BS.bornsample!(
            MersenneTwister(0x7265_7369),
            ordinary,
            8;
            ntasks=2,
        )
        @test tangent_result.configuration == ordinary_result.configuration
        @test tangent_result.log_probability ≈
              ordinary_result.log_probability atol=2e-12 rtol=2e-12
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
            rank_four;
            left_boundary=ComplexF64[1, 0],
        )
        @test length(result_four.configuration) == 1
        @test isfinite(result_four.log_probability)
        joint_four = BS.bornsample!(
            MersenneTwister(0x7134),
            rank_four;
            purified=false,
            left_boundary=ComplexF64[1, 0],
        )
        @test length(joint_four.configuration) == 2
        @test 1 <= joint_four.configuration[2] <= Int(TK.dim(symmetry))
        @test isfinite(joint_four.log_probability)

        operator_state = FiniteMPS.identityMPO(ComplexF64, 1, physical)
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
        @test length(joint_five.configuration) == 3
        @test 1 <= joint_five.configuration[2] <= Int(TK.dim(physical))
        @test 1 <= joint_five.configuration[3] <= Int(TK.dim(symmetry))
        @test isfinite(joint_five.log_probability)
        @test BS._tensor_rank.(rank_four.B) == [4]
        @test BS._tensor_rank.(rank_five.B) == [5]
    end
end
