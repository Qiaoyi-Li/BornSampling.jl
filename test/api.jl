@testset "public API and validation" begin
    # The outer FiniteMPS type selects the sampling semantics. MPS is the
    # rank-one pure-amplitude path and therefore accepts rank-3 sites only.
    rank4_in_mps = FiniteMPS.MPS(deepcopy(rank4_state().A))
    @test rank4_in_mps isa FiniteMPS.MPS
    @test_throws ArgumentError BS.BornSampler(rank4_in_mps)

    # MPO accepts rank-3 sites as synthetic one-dimensional purification legs.
    # Its outer mode still controls the public configuration layout.
    rank3_source = rank3_state(length=2, bonddim=2)
    traced_rank3_mpo = BS.BornSampler(FiniteMPS.MPO(deepcopy(rank3_source.A)))
    joint_rank3_mpo = BS.BornSampler(
        FiniteMPS.MPO(deepcopy(rank3_source.A)); purified=false,
    )
    traced_rank3_config = Vector{Int}(undef, 2)
    joint_rank3_config = Vector{Int}(undef, 4)
    traced_rank3_logp = BS.bornsample!(
        MersenneTwister(10), traced_rank3_mpo, traced_rank3_config,
    )
    joint_rank3_logp = BS.bornsample!(
        MersenneTwister(10), joint_rank3_mpo, joint_rank3_config,
    )
    @test joint_rank3_config[1:2] == traced_rank3_config
    @test joint_rank3_config[3:4] == [1, 1]
    @test joint_rank3_logp == traced_rank3_logp

    state = rank3_state(length=3, bonddim=2)
    sampler = BS.BornSampler(state)
    @test sampler.state === state

    config = Vector{Int}(undef, length(state))
    logp = BS.bornsample!(MersenneTwister(11), sampler, config)
    @test isfinite(logp)
    @test all(i -> 1 <= config[i] <= sampler.plans[i].physical.fulldim, eachindex(config))

    shot = BS.bornsample!(MersenneTwister(12), sampler)
    @test keys(shot) == (:configuration, :log_probability)
    @test length(shot.configuration) == length(state)
    @test isfinite(shot.log_probability)

    # NoSymSpinOneHalf is one trivial sector with degeneracy two. This also
    # checks that flat ordering is sector, then degeneracy, with irrep fastest.
    first_plan = sampler.plans[1]
    @test length(first_plan.physical_basis) == first_plan.physical.fulldim == 2
    @test map(info -> info.degeneracy, first_plan.physical_basis) == [1, 2]
    @test all(info -> info.irrep == 1, first_plan.physical_basis)

    direct_state = rank3_state(length=2, bonddim=2)
    direct_shot = BS.bornsample!(MersenneTwister(13), direct_state)
    @test length(direct_shot.configuration) == length(direct_state)

    # Direct MPO conveniences preserve the network-level joint mode, including
    # heterogeneous local ranks and every batch-cache keyword.
    joint_source = mixed_rank_three_site_mpo_state()
    joint_sampler = BS.BornSampler(deepcopy(joint_source); purified=false)
    joint_expected = BS.bornsample!(MersenneTwister(0x6469_7265), joint_sampler)
    joint_direct = BS.bornsample!(
        MersenneTwister(0x6469_7265),
        deepcopy(joint_source);
        purified=false,
    )
    @test joint_direct == joint_expected
    @test joint_direct.configuration[5:6] == [1, 1]

    joint_batch_sampler = BS.BornSampler(deepcopy(joint_source); purified=false)
    joint_batch_expected = BS.bornsample!(
        MersenneTwister(0x6261_7463),
        joint_batch_sampler,
        12;
        ntasks=3,
        disk=true,
        maxsize=1,
    )
    joint_batch_direct = BS.bornsample!(
        MersenneTwister(0x6261_7463),
        deepcopy(joint_source),
        12;
        purified=false,
        ntasks=3,
        disk=true,
        maxsize=1,
    )
    @test joint_batch_direct == joint_batch_expected
    @test size(joint_batch_direct.configuration) == (6, 12)
    @test all(==(1), @view joint_batch_direct.configuration[5:6, :])

    # `purified` does not change MPS semantics.
    flagged_state = rank3_state(length=2, bonddim=2)
    flagged_sampler = BS.BornSampler(flagged_state; purified=false)
    @test length(BS.bornsample!(MersenneTwister(13), flagged_sampler).configuration) ==
          length(flagged_state)

    # A non-Abelian irrep may have full dimension greater than one while its
    # reduced boundary multiplicity D* remains exactly one.
    left = ComplexF64[3 + 4im, -2im, 1 - im]
    left_sampler = BS.BornSampler(nontrivial_irrep_left_state(); left_boundary=left)
    @test vec(convert(Array, left_sampler.initial_factor)) ≈ left / norm(left)
    @test norm(left_sampler.initial_factor) ≈ 1
    left_config = Vector{Int}(undef, 2)
    @test isfinite(BS.bornsample!(MersenneTwister(14), left_sampler, left_config))

    @test_throws ArgumentError BS.BornSampler(nontrivial_irrep_left_state())
    # Reduced boundary multiplicity greater than one is outside the supported
    # contract even when a full pure-state vector is supplied.
    @test_throws ArgumentError BS.BornSampler(nontrivial_left_state())
    @test_throws ArgumentError BS.BornSampler(
        nontrivial_left_state(); left_boundary=ones(ComplexF64, 2),
    )
    @test_throws ArgumentError BS.BornSampler(
        nontrivial_left_state(); left_boundary=ones(ComplexF64, 2, 2),
    )
    @test_throws DimensionMismatch BS.BornSampler(
        nontrivial_left_state(); left_boundary=ones(ComplexF64, 3),
    )
    @test_throws DimensionMismatch BS.bornsample!(
        MersenneTwister(15), sampler, Vector{Int}(undef, length(state) - 1),
    )
end
