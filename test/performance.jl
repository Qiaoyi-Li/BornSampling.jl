@testset "per-shot allocation smoke checks" begin
    rank3_sampler = BS.BornSampler(rank3_state(length=4, bonddim=3))
    rank3_config = Vector{Int}(undef, length(rank3_sampler.state))
    rng3 = MersenneTwister(21)
    BS.bornsample!(rng3, rank3_sampler, rank3_config) # warm up
    allocated3 = @allocated BS.bornsample!(rng3, rank3_sampler, rank3_config)
    # Residual TensorMaps are created for selected branches; keep this as a
    # smoke bound against accidentally materializing dense local operators.
    @test allocated3 < 64_000

    rank4_sampler = BS.BornSampler(rank4_state())
    rank4_config = Vector{Int}(undef, length(rank4_sampler.state))
    rng4 = MersenneTwister(22)
    BS.bornsample!(rng4, rank4_sampler, rank4_config) # warm up
    allocated4 = @allocated BS.bornsample!(rng4, rank4_sampler, rank4_config)
    # Traced sampling includes the exact L/Q factors from TK.right_orth!.
    @test allocated4 < 128_000
end
