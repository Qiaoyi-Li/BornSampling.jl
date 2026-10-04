@testset "factorized physical branch" begin
    sampler = BS.BornSampler(rank4_state())
    plan = first(sampler.plans)
    workspace = first(sampler.workspaces)
    C = sampler.initial_factor

    BS._compute_weights!(workspace, C, plan)
    selected = argmax(@view workspace.q[1:plan.physical.fulldim])
    G = BS._build_selected_factor!(workspace, C, plan, selected)

    # Within one residual output charge, separate (purification, input-charge)
    # routes are concatenated, while the transitions inside each route have
    # already been coherently accumulated by `_apply_route!`.
    expected_blocks = Dict(
        sector => Matrix{eltype(C)}[] for sector in plan.residual_right.sectors
    )
    for y in 1:BS.purification_dimension(plan)
        for route in basis_routes(plan, selected, y)
            left_sector = route_left_sector(plan, route)
            TK.hasblock(C, left_sector) || continue
            push!(expected_blocks[route_right_sector(plan, route)],
                route_channel_result(plan, C, selected, y, route))
        end
    end
    for (slot, sector) in pairs(plan.residual_right.sectors)
        reference = hcat(expected_blocks[sector]...)
        @test TK.block(G, sector) ≈ reference rtol=8e-13 atol=8e-13
    end
    @test norm(G)^2 ≈ workspace.q[selected] rtol=8e-13 atol=8e-13

    # Exercise the production exact-compression helper on a deliberately wide
    # residual TensorMap and verify preservation of G*G'.
    residual_type = eltype(plan.residual_right.sectors)
    wide_dimensions = [dimension + 2 for dimension in plan.residual_right.dimensions]
    wide_domain = BS._residual_space(
        residual_type,
        Dict(plan.residual_right.sectors[i] => wide_dimensions[i]
             for i in eachindex(wide_dimensions)),
    )
    wide_G = TK.randn(
        MersenneTwister(0x7769_6465),
        ComplexF64,
        plan.residual_right.space,
        wide_domain,
    )
    reference_density = wide_G * adjoint(wide_G)
    Cnext = BS._compress_factor!(wide_G, plan)
    @test Cnext * adjoint(Cnext) ≈ reference_density rtol=2e-12 atol=2e-12
    @test Int(TK.dim(TK.domain(Cnext))) <= Int(TK.dim(plan.residual_right.space))

end
