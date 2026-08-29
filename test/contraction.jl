function test_compiled_routes(tensor; rank=2, rtol=8e-13, atol=8e-13)
    plan = compile_plan(tensor)
    C = residual_factor(plan; rank=rank)
    for x in 1:plan.physical.fulldim
        for y in 1:BS.purification_dimension(plan)
            for route in basis_routes(plan, x, y)
                actual = route_channel_result(plan, C, x, y, route)
                Cblock = TK.block(C, route_left_sector(plan, route))
                embedded = embed_route_input(plan, route, Cblock)
                dense_output = dense_channel_result(tensor, embedded, x, y)
                reference = gather_route_output(plan, route, dense_output)
                @test actual ≈ reference rtol=rtol atol=atol
            end
        end
    end
    return plan
end

@testset "local residual-route contraction" begin
    @testset "UniqueFusion rank 3" begin
        tensor = nosym_rank3_tensor()
        @test BS._style_type(tensor) === BS.UniqueStyle
        plan = test_compiled_routes(tensor; rank=1, rtol=2e-13, atol=2e-13)
        @test BS.purification_dimension(plan) == 1
        @test TK.sectortype(plan.residual_left.space) === TK.Trivial
    end

    @testset "SU(2) fusion trees, rank 3" begin
        tensor = su2_rank3_tensor()
        @test BS._style_type(tensor) === BS.FusionTreeStyle
        plan = test_compiled_routes(tensor; rank=2, rtol=5e-13, atol=5e-13)
        @test BS.purification_dimension(plan) == 1
        @test TK.sectortype(plan.residual_left.space) === TK.Trivial
    end

    @testset "U(1) × SU(2) fusion trees, rank 4" begin
        tensor = product_su2_rank4_tensor()
        @test BS._style_type(tensor) === BS.FusionTreeStyle
        plan = test_compiled_routes(tensor; rank=2)
        @test BS.purification_dimension(plan) == plan.purification.fulldim == 4
        @test TK.sectortype(plan.residual_left.space) === TK.Irrep[TK.U₁]
    end
end
