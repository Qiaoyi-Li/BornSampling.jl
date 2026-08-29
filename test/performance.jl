@testset "workspace reuse and allocation smoke checks" begin
    rank3_sampler = BS.BornSampler(rank3_state(length=4, bonddim=3))
    rank3_config = Vector{Int}(undef, length(rank3_sampler.state))
    rng3 = MersenneTwister(21)
    BS.bornsample!(rng3, rank3_sampler, rank3_config) # warm up

    workspace3 = first(rank3_sampler.workspaces)
    plans3 = rank3_sampler.plans
    initial3 = rank3_sampler.initial_factor
    route_output3 = workspace3.route_output
    scratch3, q3 = workspace3.scratch, workspace3.q
    branch_columns3 = workspace3.branch_columns
    allocated3 = @allocated BS.bornsample!(rng3, rank3_sampler, rank3_config)
    @test rank3_sampler.plans === plans3
    @test rank3_sampler.initial_factor === initial3
    @test workspace3.route_output === route_output3
    @test workspace3.scratch === scratch3
    @test workspace3.q === q3
    @test workspace3.branch_columns === branch_columns3
    @test size(route_output3, 2) == 1
    # Residual TensorMaps are created for selected branches; keep this as a
    # smoke bound against accidentally materializing dense local operators.
    @test allocated3 < 64_000

    rank4_sampler = BS.BornSampler(rank4_state())
    rank4_config = Vector{Int}(undef, length(rank4_sampler.state))
    rng4 = MersenneTwister(22)
    BS.bornsample!(rng4, rank4_sampler, rank4_config) # warm up

    workspace4 = first(rank4_sampler.workspaces)
    plans4 = rank4_sampler.plans
    initial4 = rank4_sampler.initial_factor
    route_output4 = workspace4.route_output
    scratch4, q4 = workspace4.scratch, workspace4.q
    allocated4 = @allocated BS.bornsample!(rng4, rank4_sampler, rank4_config)
    @test rank4_sampler.plans === plans4
    @test rank4_sampler.initial_factor === initial4
    @test workspace4.route_output === route_output4
    @test workspace4.scratch === scratch4
    @test workspace4.q === q4
    @test size(route_output4, 1) >= maximum(rank4_sampler.plans) do plan
        maximum(plan.residual_right.dimensions; init=0)
    end
    # TensorKit rightorth! allocates its exact L/Q factors in the first version.
    @test allocated4 < 128_000
end
