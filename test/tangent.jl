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
        @test_throws ArgumentError BS.BornSampler(tangent; purified=false)
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

        operator_state = FiniteMPS.identityMPO(ComplexF64, 1, physical)
        rank_five = tangent_with_persistent_symmetry(
            FiniteMPSTangents.BaseMPS(operator_state),
            symmetry;
            seed=0x7135,
        )
        result_five = BS.bornsample!(MersenneTwister(0x7135), rank_five)
        @test length(result_five.configuration) == 1
        @test isfinite(result_five.log_probability)
        @test BS._tensor_rank.(rank_four.B) == [4]
        @test BS._tensor_rank.(rank_five.B) == [5]
    end
end
