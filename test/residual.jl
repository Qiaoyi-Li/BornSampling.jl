@testset "residual symmetry" begin
    @testset "structural component inference" begin
        spaces = residual_component_spaces()

        atomic = BS._infer_residual_symmetry(
            TK.sectortype(spaces.atomic_unique),
            (spaces.atomic_unique,),
        )
        @test atomic isa BS.ResidualSymmetry{TK.Irrep[TK.U₁]}
        for sector in TK.sectors(spaces.atomic_unique)
            @test BS._project_sector(atomic, sector) == sector
        end

        nonabelian = BS._infer_residual_symmetry(
            TK.sectortype(spaces.pure_nonabelian),
            (spaces.pure_nonabelian,),
        )
        @test nonabelian isa BS.ResidualSymmetry{TK.Trivial}
        for sector in TK.sectors(spaces.pure_nonabelian)
            @test BS._project_sector(nonabelian, sector) == TK.Trivial()
        end

        second = BS._infer_residual_symmetry(
            TK.sectortype(spaces.unique_second),
            (spaces.unique_second,),
        )
        @test second isa BS.ResidualSymmetry{TK.Irrep[TK.U₁]}
        for sector in TK.sectors(spaces.unique_second)
            @test BS._project_sector(second, sector) == sector[2]
        end

        nonadjacent = BS._infer_residual_symmetry(
            TK.sectortype(spaces.unique_nonadjacent),
            (spaces.unique_nonadjacent,),
        )
        expected_nonadjacent = TK.ProductSector{Tuple{
            TK.Irrep[TK.U₁],
            TK.Irrep[TK.ℤ₂],
        }}
        @test nonadjacent isa BS.ResidualSymmetry{expected_nonadjacent}
        for sector in TK.sectors(spaces.unique_nonadjacent)
            @test BS._project_sector(nonadjacent, sector) ==
                  TK.deligneproduct(sector[1], sector[3])
        end

        alternate = BS._infer_residual_symmetry(
            TK.sectortype(spaces.unique_alternate),
            (spaces.unique_alternate,),
        )
        expected_alternate = TK.ProductSector{Tuple{
            TK.Irrep[TK.U₁],
            TK.Irrep[TK.ℤ₂],
        }}
        @test alternate isa BS.ResidualSymmetry{expected_alternate}
        for sector in TK.sectors(spaces.unique_alternate)
            @test BS._project_sector(alternate, sector) ==
                  TK.deligneproduct(sector[2], sector[4])
        end
    end

    @testset "compiled U(1) x SU(2) routes" begin
        tensor = residual_route_rank4_tensor()
        plan = compile_plan(tensor)
        residual_type = TK.Irrep[TK.U₁]
        original_type = TK.sectortype(tensor.A)
        project_sector = sector -> sector[1]

        @test BS._style_type(tensor) === BS.FusionTreeStyle
        @test TK.sectortype(plan.residual_left.space) === residual_type
        @test TK.sectortype(plan.residual_right.space) === residual_type

        left_layout = reference_residual_layout(plan.left.space, project_sector)
        right_layout = reference_residual_layout(plan.right.space, project_sector)
        compiled_left_dimensions = Dict(
            plan.residual_left.sectors[slot] => plan.residual_left.dimensions[slot]
            for slot in eachindex(plan.residual_left.sectors)
        )
        compiled_right_dimensions = Dict(
            plan.residual_right.sectors[slot] => plan.residual_right.dimensions[slot]
            for slot in eachindex(plan.residual_right.sectors)
        )
        @test compiled_left_dimensions == left_layout.dimensions
        @test compiled_right_dimensions == right_layout.dimensions
        @test left_layout.dimensions == Dict(
            residual_type(0) => 4,
            residual_type(1) => 2,
            residual_type(-1) => 2,
        )

        for (original_slot, embedding) in pairs(plan.residual_left.embeddings)
            original_sector = plan.left.sectors[original_slot]
            @test embedding_sector(plan.residual_left, embedding) ==
                  project_sector(original_sector)
            @test embedding.rows == left_layout.offsets[original_sector]
        end
        for (original_slot, embedding) in pairs(plan.residual_right.embeddings)
            original_sector = plan.right.sectors[original_slot]
            @test embedding_sector(plan.residual_right, embedding) ==
                  project_sector(original_sector)
            @test embedding.rows == right_layout.offsets[original_sector]
        end

        ranks = Dict{residual_type,Int}(sector => 2 for sector in plan.residual_left.sectors)
        factor_domain = BS._residual_space(residual_type, ranks)
        C = TK.randn(
            MersenneTwister(0x435f_726f),
            ComplexF64,
            plan.residual_left.space,
            factor_domain,
        )
        dense = convert(Array, tensor.A)
        scratch = zeros(ComplexF64, BS.scratch_length(plan))

        for x in 1:plan.physical.fulldim, y in 1:plan.purification.fulldim
            K = transpose(@view dense[:, x, y, :])
            for route in basis_routes(plan, x, y)
                left_sector = route_left_sector(plan, route)
                right_sector = route_right_sector(plan, route)
                Cblock = TK.block(C, left_sector)
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

                embedded = embed_residual_input_block(
                    plan.left.space,
                    left_layout,
                    project_sector,
                    left_sector,
                    Cblock,
                )
                reference = gather_residual_output_block(
                    K * embedded,
                    plan.right.space,
                    right_layout,
                    project_sector,
                    right_sector,
                )
                @test actual ≈ reference rtol=2e-12 atol=2e-12
            end
        end

        # Isolate the route containing two SU(2) fusion channels with identical
        # original legs. Its transition contributions must interfere before a
        # branch norm is taken.
        physical_sector = original_type(0, 1 // 2)
        x = first(TK.axes(plan.physical.space, physical_sector))
        y = first(TK.axes(plan.purification.space, physical_sector))
        zero_charge = residual_type(0)
        route = only(filter(
            route -> route_left_sector(plan, route) == zero_charge &&
                     route_right_sector(plan, route) == zero_charge,
            basis_routes(plan, x, y),
        ))
        @test length(route.transition_indices) > 1

        coherent_domain = BS._residual_space(
            residual_type,
            Dict{residual_type,Int}(sector => 3 for sector in plan.residual_left.sectors),
        )
        coherent_C = TK.zeros(ComplexF64, plan.residual_left.space, coherent_domain)
        left_spin_one = original_type(0, 1)
        spin_rows = left_layout.offsets[left_spin_one]
        TK.block(coherent_C, zero_charge)[spin_rows, :] .=
            Matrix{ComplexF64}(LinearAlgebra.I, 3, 3)
        actual = zeros(
            ComplexF64,
            plan.residual_right.dimensions[route.right_slot],
            3,
        )
        BS._apply_route!(
            actual,
            TK.block(coherent_C, zero_charge),
            plan,
            route,
            plan.physical_basis[x],
            plan.purification_basis[y],
            scratch,
        )

        embedded = embed_residual_input_block(
            plan.left.space,
            left_layout,
            project_sector,
            zero_charge,
            TK.block(coherent_C, zero_charge),
        )
        tree_pairs = collect(TK.fusiontrees(tensor.A))
        parts = map(route.transition_indices) do transition_index
            partial = zero(tensor.A)
            pair = tree_pairs[transition_index]
            partial[pair...] .= tensor.A[pair...]
            partial_dense = convert(Array, partial)
            gather_residual_output_block(
                transpose(@view(partial_dense[:, x, y, :])) * embedded,
                plan.right.space,
                right_layout,
                project_sector,
                zero_charge,
            )
        end
        @test actual ≈ sum(parts) rtol=2e-12 atol=2e-12
        @test abs(norm(actual)^2 - sum(part -> norm(part)^2, parts)) > 1e-6

        # Different purification outcomes and input residual sectors represent
        # an incoherent mixture. Build the same horizontal factor prescribed by
        # the production algorithm and compare its Gram matrix with a direct
        # full-basis density update.
        selected = x
        pieces = Dict(
            sector => Matrix{ComplexF64}[] for sector in plan.residual_right.sectors
        )
        reference_density = zeros(
            ComplexF64,
            plan.right.fulldim,
            plan.right.fulldim,
        )
        for purification in 1:plan.purification.fulldim
            K = transpose(@view dense[:, selected, purification, :])
            for left_sector in plan.residual_left.sectors
                Cblock = TK.block(C, left_sector)
                embedded = embed_residual_input_block(
                    plan.left.space,
                    left_layout,
                    project_sector,
                    left_sector,
                    Cblock,
                )
                Yfull = K * embedded
                reference_density .+= Yfull * adjoint(Yfull)
            end
            for route in basis_routes(plan, selected, purification)
                left_sector = route_left_sector(plan, route)
                right_sector = route_right_sector(plan, route)
                Cblock = TK.block(C, left_sector)
                piece = zeros(
                    ComplexF64,
                    plan.residual_right.dimensions[route.right_slot],
                    size(Cblock, 2),
                )
                BS._apply_route!(
                    piece,
                    Cblock,
                    plan,
                    route,
                    plan.physical_basis[selected],
                    plan.purification_basis[purification],
                    scratch,
                )
                push!(pieces[right_sector], piece)
            end
        end

        factor_density = zeros(ComplexF64, size(reference_density))
        widths = Dict{residual_type,Int}()
        for right_sector in plan.residual_right.sectors
            branch_block = hcat(pieces[right_sector]...)
            widths[right_sector] = size(branch_block, 2)
            scatter_residual_gram!(
                factor_density,
                branch_block,
                plan.right.space,
                right_layout,
                project_sector,
                right_sector,
            )
        end
        @test widths == Dict(
            residual_type(0) => 8,
            residual_type(1) => 6,
            residual_type(-1) => 6,
        )
        @test factor_density ≈ reference_density rtol=3e-12 atol=3e-12
        @test real(tr(factor_density)) ≈
              sum(sum(abs2, piece) for blocks in values(pieces) for piece in blocks)
    end

    @testset "TensorKit exact residual compression" begin
        residual_type = TK.Irrep[TK.U₁]
        right_space = TK.GradedSpace(
            residual_type(0) => 2,
            residual_type(1) => 3,
        )
        branch_space = TK.GradedSpace(
            residual_type(0) => 5,
            residual_type(1) => 4,
        )
        G = TK.randn(
            MersenneTwister(0x4c51_706f),
            ComplexF64,
            right_space,
            branch_space,
        )
        reference = copy(G)
        L, Q = TK.rightorth!(G; alg=TK.LQpos())

        @test reference ≈ L * Q rtol=2e-13 atol=2e-13
        @test reference * adjoint(reference) ≈
              L * adjoint(L) rtol=5e-13 atol=5e-13
        @test size(TK.block(L, residual_type(0))) == (2, 2)
        @test size(TK.block(L, residual_type(1))) == (3, 3)
        @test size(TK.block(Q, residual_type(0))) == (2, 5)
        @test size(TK.block(Q, residual_type(1))) == (3, 4)

        trivial = TK.Trivial()
        trivial_right = BS._residual_space(TK.Trivial, Dict(trivial => 3))
        trivial_branch = BS._residual_space(TK.Trivial, Dict(trivial => 5))
        dense_G = TK.randn(
            MersenneTwister(0x7472_6976),
            ComplexF64,
            trivial_right,
            trivial_branch,
        )
        dense_reference = copy(dense_G)
        dense_L, dense_Q = TK.rightorth!(dense_G; alg=TK.LQpos())
        @test dense_reference ≈ dense_L * dense_Q rtol=2e-13 atol=2e-13
        @test dense_reference * adjoint(dense_reference) ≈
              dense_L * adjoint(dense_L) rtol=5e-13 atol=5e-13
        @test size(TK.block(dense_L, trivial)) == (3, 3)
        @test size(TK.block(dense_Q, trivial)) == (3, 5)
    end
end
