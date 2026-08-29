@testset "representative FiniteMPS LocalSpace routes" begin
    local_spaces = (
        :NoSymSpinOneHalf,
        :U1Spin,
        :SU2Spin,
        :U1SpinlessFermion,
        :U1SU2Fermion,
        :Z2SU2Fermion,
        :U1U1Fermion,
        :U1SU2tJFermion,
        :Z2SU2tJFermion,
        :U1U1tJFermion,
    )

    for (seed, name) in pairs(local_spaces)
        local_space = getproperty(FiniteMPS, name)
        physical = local_space.pspace
        tensor_map = TK.randn(
            MersenneTwister(seed),
            ComplexF64,
            ⊗(physical, physical),
            ⊗(physical, physical),
        )
        tensor = FiniteMPS.MPSTensor(tensor_map)
        plan = compile_plan(tensor)
        C = residual_factor(plan; rank=2, seed=seed)
        dense = convert(Array, tensor_map)
        scratch = zeros(ComplexF64, BS.scratch_length(plan))
        route_count = 0

        for x in 1:plan.physical.fulldim, y in 1:plan.purification.fulldim
            K = transpose(@view dense[:, x, y, :])
            for route in basis_routes(plan, x, y)
                route_count += 1
                Cblock = TK.block(C, route_left_sector(plan, route))
                actual = zeros(
                    ComplexF64,
                    plan.residual_right.dimensions[route.right_slot],
                    size(Cblock, 2),
                )
                BS._apply_route!(
                    actual,
                    Cblock,
                    plan,
                    route,
                    plan.physical_basis[x],
                    plan.purification_basis[y],
                    scratch,
                )
                reference = gather_route_output(
                    plan,
                    route,
                    K * embed_route_input(plan, route, Cblock),
                )
                @test actual ≈ reference rtol=3e-12 atol=3e-12
            end
        end
        @test route_count > 0
    end
end
