@testset "traced-MPO all-physical factor bank" begin
    @testset "factor bank agrees with selected branches" begin
        sampler = BS.BornSampler(residual_route_rank4_state())
        workspace = first(sampler.workspaces)
        factor = sampler.initial_factor
        original_plan = first(sampler.plans)

        factors = BS._compute_weights_and_factors!(
            workspace,
            factor,
            original_plan,
        )

        @test length(factors) == original_plan.physical.fulldim

        reference_workspace = BS._clone_workspace(workspace)
        BS._compute_weights!(reference_workspace, factor, original_plan)
        branch_count = original_plan.physical.fulldim
        @test workspace.q[1:branch_count] ≈
              reference_workspace.q[1:branch_count] rtol=8e-13 atol=8e-13

        for selected in 1:branch_count
            reference = BS._build_selected_factor!(
                reference_workspace,
                factor,
                original_plan,
                selected,
            )
            @test factors[selected] * adjoint(factors[selected]) ≈
                  reference * adjoint(reference) rtol=1e-12 atol=1e-12
            @test norm(factors[selected])^2 ≈
                  workspace.q[selected] rtol=8e-13 atol=8e-13
        end
    end
end
