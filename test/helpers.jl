const ⊗ = TK.:⊗

"Return a freshly compiled plan without constructing a whole sampler."
function compile_plan(tensor)
    style = BS._style_type(tensor)
    spaces = if tensor isa FiniteMPS.MPSTensor{3}
        (BS.leftspace(tensor), BS.physspace(tensor), BS.rightspace(tensor))
    else
        (
            BS.leftspace(tensor),
            BS.physspace(tensor),
            BS.purspace(tensor),
            BS.rightspace(tensor),
        )
    end
    residual = BS._infer_residual_symmetry(TK.sectortype(tensor.A), spaces)
    return BS._compile_site(tensor, style, BS._kernel_cache(style), residual)
end

route_left_sector(plan, route) = plan.residual_left.sectors[route.left_slot]
route_right_sector(plan, route) = plan.residual_right.sectors[route.right_slot]
embedding_sector(residual, embedding) = residual.sectors[embedding.residual_slot]

function basis_routes(plan, physical::Int, purification::Int)
    return BS._channel_routes(
        plan,
        plan.physical_basis[physical],
        plan.purification_basis[purification],
    )
end

function nosym_rank3_tensor(; T=ComplexF64)
    left = TK.ComplexSpace(2)
    physical = FiniteMPS.NoSymSpinOneHalf.pspace
    right = TK.ComplexSpace(3)
    A = TK.randn(T, ⊗(left, physical), right)
    return FiniteMPS.MPSTensor(A)
end

function su2_rank3_tensor(; T=ComplexF64)
    physical = FiniteMPS.SU2Spin.pspace
    left = TK.Rep[TK.SU₂](0 => 1, 1 => 1)
    right = TK.Rep[TK.SU₂](1 // 2 => 2, 3 // 2 => 1)
    A = TK.randn(T, ⊗(left, physical), right)
    return FiniteMPS.MPSTensor(A)
end

function product_su2_spaces()
    physical = FiniteMPS.U1SU2Fermion.pspace
    sector_type = TK.sectortype(physical)
    vacuum = one(sector_type)
    virtual = TK.GradedSpace(vacuum => 1, sector_type((0, 1)) => 1)
    boundary = TK.GradedSpace(vacuum => 1)
    return (; physical, virtual, boundary)
end

function product_su2_rank4_tensor(; T=ComplexF64)
    spaces = product_su2_spaces()
    X = TK.randn(
        T,
        ⊗(spaces.virtual, spaces.physical),
        ⊗(spaces.physical, spaces.virtual),
    )
    return FiniteMPS.MPSTensor(X)
end

"Custom spaces used to verify structural residual-component inference."
function residual_component_spaces()
    times = TK.:×

    atomic_type = TK.Irrep[TK.U₁]
    atomic_unique = TK.GradedSpace(
        atomic_type(-2) => 1,
        atomic_type(1) => 2,
    )

    nonabelian_type = TK.Irrep[TK.SU₂]
    pure_nonabelian = TK.GradedSpace(
        nonabelian_type(0) => 1,
        nonabelian_type(1 // 2) => 1,
    )

    second_type = TK.Irrep[times(TK.SU₂, TK.U₁)]
    unique_second = TK.GradedSpace(
        second_type(0, 0) => 1,
        second_type(1 // 2, 1) => 1,
        second_type(1, 2) => 1,
    )

    nonadjacent_type = TK.Irrep[times(TK.U₁, TK.SU₂, TK.ℤ₂)]
    unique_nonadjacent = TK.GradedSpace(
        nonadjacent_type(0, 0, 0) => 1,
        nonadjacent_type(1, 1 // 2, 1) => 2,
        nonadjacent_type(-1, 1, 0) => 1,
    )

    alternate_type = TK.Irrep[times(TK.SU₂, TK.U₁, TK.SU₂, TK.ℤ₂)]
    unique_alternate = TK.GradedSpace(
        alternate_type(0, 0, 0, 0) => 1,
        alternate_type(1 // 2, 1, 1, 1) => 1,
    )

    return (;
        atomic_unique,
        pure_nonabelian,
        unique_second,
        unique_nonadjacent,
        unique_alternate,
    )
end

"A generic U(1) x SU(2) rank-4 tensor with several residual charges and fusion paths."
function residual_route_rank4_tensor(; T=ComplexF64, seed=3)
    sector_type = TK.Irrep[TK.:×(TK.U₁, TK.SU₂)]
    physical = TK.GradedSpace(
        sector_type(-1, 0) => 1,
        sector_type(0, 1 // 2) => 1,
        sector_type(1, 0) => 1,
    )
    virtual = TK.GradedSpace(
        sector_type(0, 0) => 1,
        sector_type(0, 1) => 1,
        sector_type(1, 1 // 2) => 1,
        sector_type(-1, 1 // 2) => 1,
    )
    tensor = TK.randn(
        MersenneTwister(seed),
        T,
        ⊗(virtual, physical),
        ⊗(physical, virtual),
    )
    return FiniteMPS.MPSTensor(tensor)
end

"Two-site U(1) x SU(2) purification state exercising several residual charges."
function residual_route_rank4_state(; T=ComplexF64, seed=41)
    sector_type = TK.Irrep[TK.:×(TK.U₁, TK.SU₂)]
    physical = TK.GradedSpace(
        sector_type(-1, 0) => 1,
        sector_type(0, 1 // 2) => 1,
        sector_type(1, 0) => 1,
    )
    boundary = TK.GradedSpace(one(sector_type) => 1)
    virtual = TK.GradedSpace(
        sector_type(0, 0) => 1,
        sector_type(0, 1) => 1,
        sector_type(1, 1 // 2) => 1,
        sector_type(-1, 1 // 2) => 1,
    )
    rng = MersenneTwister(seed)
    first_tensor = TK.randn(
        rng,
        T,
        ⊗(boundary, physical),
        ⊗(physical, virtual),
    )
    second_tensor = TK.randn(
        rng,
        T,
        ⊗(virtual, physical),
        ⊗(physical, boundary),
    )
    return FiniteMPS.MPO(FiniteMPS.MPSTensor.([first_tensor, second_tensor]))
end

"Independent original-basis to residual-block layout used by dense test oracles."
function reference_residual_layout(space, project_sector)
    offsets = Dict{Any,UnitRange{Int}}()
    dimensions = Dict{Any,Int}()
    for sector in TK.sectors(space)
        residual_sector = project_sector(sector)
        width = Int(TK.dim(space, sector)) * Int(TK.dim(sector))
        first_index = get(dimensions, residual_sector, 0) + 1
        offsets[sector] = first_index:(first_index + width - 1)
        dimensions[residual_sector] = first_index + width - 1
    end
    return (; offsets, dimensions)
end

function embed_residual_input_block(space, layout, project_sector, sector, block)
    embedded = zeros(eltype(block), Int(TK.dim(space)), size(block, 2))
    for original_sector in TK.sectors(space)
        project_sector(original_sector) == sector || continue
        embedded[TK.axes(space, original_sector), :] .=
            block[layout.offsets[original_sector], :]
    end
    return embedded
end

function gather_residual_output_block(
    output,
    space,
    layout,
    project_sector,
    sector,
)
    gathered = zeros(eltype(output), layout.dimensions[sector], size(output, 2))
    for original_sector in TK.sectors(space)
        project_sector(original_sector) == sector || continue
        gathered[layout.offsets[original_sector], :] .=
            output[TK.axes(space, original_sector), :]
    end
    return gathered
end

function scatter_residual_gram!(
    output,
    block,
    space,
    layout,
    project_sector,
    sector,
)
    gram = block * adjoint(block)
    matching = filter(c -> project_sector(c) == sector, collect(TK.sectors(space)))
    for row_sector in matching, column_sector in matching
        output[
            TK.axes(space, row_sector),
            TK.axes(space, column_sector),
        ] .= gram[
            layout.offsets[row_sector],
            layout.offsets[column_sector],
        ]
    end
    return output
end

function rank3_state(; T=ComplexF64, length=3, bonddim=3)
    physical = FiniteMPS.NoSymSpinOneHalf.pspace
    virtual = TK.ComplexSpace(bonddim)
    return FiniteMPS.randMPS(T, length, physical, virtual)
end

function rank4_state(; T=ComplexF64)
    spaces = product_su2_spaces()
    X1 = TK.randn(
        T,
        ⊗(spaces.boundary, spaces.physical),
        ⊗(spaces.physical, spaces.virtual),
    )
    X2 = TK.randn(
        T,
        ⊗(spaces.virtual, spaces.physical),
        ⊗(spaces.physical, spaces.boundary),
    )
    return FiniteMPS.MPO(FiniteMPS.MPSTensor.([X1, X2]))
end

"A mixed-rank MPO whose traced factor must be compressed at its rank-3 site."
function mixed_rank_mpo_state(; T=ComplexF64, seed=0x6d69_7865)
    physical = FiniteMPS.NoSymSpinOneHalf.pspace
    purification = TK.ComplexSpace(2)
    boundary = TK.ComplexSpace(1)
    virtual = TK.ComplexSpace(2)
    rng = MersenneTwister(seed)
    X1 = TK.randn(
        rng,
        T,
        ⊗(boundary, physical),
        ⊗(purification, virtual),
    )
    A2 = TK.randn(rng, T, ⊗(virtual, physical), boundary)
    return FiniteMPS.MPO(FiniteMPS.MPSTensor.([X1, A2]))
end

"Three-site rank-4 → rank-3 → rank-3 MPO for public convenience methods."
function mixed_rank_three_site_mpo_state(; T=ComplexF64, seed=0x6d69_7833)
    physical = FiniteMPS.NoSymSpinOneHalf.pspace
    purification = TK.ComplexSpace(2)
    boundary = TK.ComplexSpace(1)
    virtual = TK.ComplexSpace(2)
    rng = MersenneTwister(seed)
    X1 = TK.randn(
        rng,
        T,
        ⊗(boundary, physical),
        ⊗(purification, virtual),
    )
    A2 = TK.randn(rng, T, ⊗(virtual, physical), virtual)
    A3 = TK.randn(rng, T, ⊗(virtual, physical), boundary)
    return FiniteMPS.MPO(FiniteMPS.MPSTensor.([X1, A2, A3]))
end

function nontrivial_left_state(; T=ComplexF64)
    left = TK.ComplexSpace(2)
    physical = FiniteMPS.NoSymSpinOneHalf.pspace
    right = TK.ComplexSpace(1)
    A = TK.randn(T, ⊗(left, physical), right)
    return FiniteMPS.MPS([FiniteMPS.MPSTensor(A)])
end

function nontrivial_irrep_left_state(; T=ComplexF64)
    physical = FiniteMPS.SU2Spin.pspace
    left = TK.Rep[TK.SU₂](1 => 1)
    bond = TK.Rep[TK.SU₂](1 // 2 => 1)
    right = TK.Rep[TK.SU₂](0 => 1)
    A1 = TK.randn(T, ⊗(left, physical), bond)
    A2 = TK.randn(T, ⊗(bond, physical), right)
    return FiniteMPS.MPS(FiniteMPS.MPSTensor.([A1, A2]))
end

function residual_factor(plan; rank=2, T=ComplexF64, seed=0x7265_7369)
    residual_type = eltype(plan.residual_left.sectors)
    dimensions = Dict{residual_type,Int}(
        sector => rank for sector in plan.residual_left.sectors
    )
    domain = BS._residual_space(residual_type, dimensions)
    C = TK.randn(MersenneTwister(seed), T, plan.residual_left.space, domain)
    rmul!(C, inv(norm(C)))
    return C
end

function route_channel_result(plan, C, x::Int, y::Int, route)
    Cblock = TK.block(C, route_left_sector(plan, route))
    rows = plan.residual_right.dimensions[route.right_slot]
    Y = zeros(eltype(C), rows, size(Cblock, 2))
    scratch = zeros(eltype(C), BS.scratch_length(plan))
    BS._apply_route!(
        Y,
        Cblock,
        plan,
        route,
        plan.physical_basis[x],
        plan.purification_basis[y],
        scratch,
    )
    return Y
end

function embed_route_input(plan, route, Cblock)
    embedded = zeros(eltype(Cblock), plan.left.fulldim, size(Cblock, 2))
    for (original_slot, embedding) in pairs(plan.residual_left.embeddings)
        embedding.residual_slot == route.left_slot || continue
        original_rows = TK.axes(plan.left.space, plan.left.sectors[original_slot])
        embedded[original_rows, :] .= Cblock[embedding.rows, :]
    end
    return embedded
end

function gather_route_output(plan, route, output)
    gathered = zeros(
        eltype(output),
        plan.residual_right.dimensions[route.right_slot],
        size(output, 2),
    )
    for (original_slot, embedding) in pairs(plan.residual_right.embeddings)
        embedding.residual_slot == route.right_slot || continue
        original_rows = TK.axes(plan.right.space, plan.right.sectors[original_slot])
        gathered[embedding.rows, :] .= output[original_rows, :]
    end
    return gathered
end

function dense_channel_result(tensor::FiniteMPS.MPSTensor{3}, C, x::Int, ::Int=1)
    dense = convert(Array, tensor.A)
    return transpose(@view(dense[:, x, :])) * C
end

function dense_channel_result(tensor::FiniteMPS.MPSTensor{4}, C, x::Int, y::Int)
    dense = convert(Array, tensor.A)
    return transpose(@view(dense[:, x, y, :])) * C
end

function all_configurations(dimensions::AbstractVector{Int})
    iterators = map(d -> 1:d, dimensions)
    return [collect(configuration) for configuration in Iterators.product(iterators...)]
end

function dense_physical_weights(state; left_boundary=nothing)
    tensors = map(site -> convert(Array, site.A), state.A)
    physical_dimensions = map(A -> size(A, 2), tensors)
    purification_dimensions = map(A -> ndims(A) == 3 ? 1 : size(A, 3), tensors)
    initial = left_boundary === nothing ? ones(eltype(first(tensors)), size(first(tensors), 1)) :
              Vector{eltype(first(tensors))}(left_boundary)
    weights = Dict{Tuple{Vararg{Int}},Float64}()
    purification_configurations = all_configurations(purification_dimensions)
    for physical in all_configurations(physical_dimensions)
        weight = 0.0
        for purification in purification_configurations
            factor = initial
            for i in eachindex(tensors)
                A = tensors[i]
                factor = if ndims(A) == 3
                    transpose(@view(A[:, physical[i], :])) * factor
                else
                    transpose(
                        @view(A[:, physical[i], purification[i], :]),
                    ) * factor
                end
            end
            weight += real(sum(abs2, factor))
        end
        weights[Tuple(physical)] = weight
    end
    return weights
end

"Dense joint physical-purification probabilities in external `[x..., y...]` layout."
function dense_joint_weights(state; left_boundary=nothing)
    tensors = map(site -> convert(Array, site.A), state.A)
    physical_dimensions = map(A -> size(A, 2), tensors)
    purification_dimensions = map(A -> ndims(A) == 3 ? 1 : size(A, 3), tensors)
    initial = left_boundary === nothing ?
              ones(eltype(first(tensors)), size(first(tensors), 1)) :
              Vector{eltype(first(tensors))}(left_boundary)
    weights = Dict{Tuple{Vararg{Int}},Float64}()
    for physical in all_configurations(physical_dimensions)
        for purification in all_configurations(purification_dimensions)
            factor = initial
            for site in eachindex(tensors)
                A = tensors[site]
                factor = if ndims(A) == 3
                    transpose(@view(A[:, physical[site], :])) * factor
                else
                    transpose(
                        @view(A[:, physical[site], purification[site], :]),
                    ) * factor
                end
            end
            configuration = Tuple(vcat(physical, purification))
            weights[configuration] = real(sum(abs2, factor))
        end
    end
    return weights
end

function normalize_weights!(weights)
    normalization = sum(values(weights))
    normalization > 0 || error("oracle state has zero norm")
    for configuration in keys(weights)
        weights[configuration] /= normalization
    end
    return weights
end

"Advance one test factor through the mode-specific production update."
function advance_test_factor!(
    workspace::BS.SamplingWorkspace{BS.TracedMPOMode},
    factor,
    plan,
    selected::Int,
    selected_weight,
)
    G = BS._build_selected_factor!(workspace, factor, plan, selected)
    return BS._advance_built_traced_factor!(G, plan, selected_weight)
end

function advance_test_factor!(
    workspace,
    factor,
    plan,
    selected::Int,
    selected_weight,
)
    return BS._advance_factor!(
        workspace,
        factor,
        plan,
        selected,
        selected_weight,
    )
end

"Evaluate the log of the sampler's sequential conditionals on a prescribed branch."
function sequential_log_probability!(sampler, physical::AbstractVector{Int})
    workspace = first(sampler.workspaces)
    factor = sampler.initial_factor
    log_probability = zero(eltype(workspace.q))
    for (site, plan) in enumerate(sampler.plans)
        physical_dimension = plan.physical.fulldim
        BS._compute_weights!(workspace, factor, plan)
        normalization = BS._total_weight(workspace.q, physical_dimension, site)
        selected_weight = workspace.q[physical[site]]
        iszero(selected_weight) && return oftype(log_probability, -Inf)
        log_probability += log(selected_weight) - log(normalization)
        factor = advance_test_factor!(
            workspace,
            factor,
            plan,
            physical[site],
            selected_weight,
        )
        @test Int(TK.dim(TK.domain(factor))) <=
              Int(TK.dim(plan.residual_right.space))
    end
    return log_probability
end

"A bond-one product state whose only nonzero physical configuration is all ones."
function deterministic_rank3_state(; T=ComplexF64, length=4)
    physical = FiniteMPS.NoSymSpinOneHalf.pspace
    boundary = TK.ComplexSpace(1)
    data = reshape(T[one(T), zero(T)], 1, 2, 1)
    tensor = TK.TensorMap(data, ⊗(boundary, physical), boundary)
    return FiniteMPS.MPS([
        FiniteMPS.MPSTensor(copy(tensor)) for _ in 1:length
    ])
end

mutable struct SequenceRNG <: Random.AbstractRNG
    values::Vector{Float64}
    index::Int
end

SequenceRNG(values) = SequenceRNG(collect(Float64, values), 1)

function Random.rand(rng::SequenceRNG, ::Type{Float64})
    rng.index <= length(rng.values) || error("SequenceRNG exhausted")
    value = rng.values[rng.index]
    rng.index += 1
    return value
end

function uniforms_for_configuration(probabilities, target::AbstractVector{Int})
    uniforms = Float64[]
    for site in eachindex(target)
        prefix = Tuple(target[1:(site - 1)])
        prefix_mass = 0.0
        lower_mass = 0.0
        selected_mass = 0.0
        for (configuration, probability) in probabilities
            configuration[1:(site - 1)] == prefix || continue
            prefix_mass += probability
            if configuration[site] < target[site]
                lower_mass += probability
            elseif configuration[site] == target[site]
                selected_mass += probability
            end
        end
        selected_mass > 0 || error("cannot force a zero-probability branch")
        push!(uniforms, (lower_mass + selected_mass / 2) / prefix_mass)
    end
    return uniforms
end


"Force `[x..., y...]` while the sampler draws one flattened `(x,y)` per site."
function uniforms_for_joint_configuration(
    probabilities,
    target::AbstractVector{Int},
    purification_dimensions::AbstractVector{Int},
)
    chain_length = length(purification_dimensions)
    length(target) == 2 * chain_length || throw(DimensionMismatch(
        "joint target length must be twice the chain length",
    ))
    uniforms = Float64[]
    for site in 1:chain_length
        purification_dimension = purification_dimensions[site]
        target_outcome =
            (target[site] - 1) * purification_dimension + target[chain_length + site]
        prefix_mass = 0.0
        lower_mass = 0.0
        selected_mass = 0.0
        for (configuration, probability) in probabilities
            matches_prefix = all(1:(site - 1)) do previous
                configuration[previous] == target[previous] &&
                    configuration[chain_length + previous] ==
                    target[chain_length + previous]
            end
            matches_prefix || continue
            prefix_mass += probability
            outcome = (configuration[site] - 1) * purification_dimension +
                      configuration[chain_length + site]
            if outcome < target_outcome
                lower_mass += probability
            elseif outcome == target_outcome
                selected_mass += probability
            end
        end
        selected_mass > 0 || error("cannot force a zero-probability joint branch")
        push!(uniforms, (lower_mass + selected_mass / 2) / prefix_mass)
    end
    return uniforms
end


"""
An array wrapper that counts the reduced-block views taken by a compiled
contraction plan. `_apply_route!` takes exactly one such view for every
transition application, while route discovery and factor-space bookkeeping do
not touch the wrapped data. This lets performance-structure tests detect a
second selected-branch contraction without adding instrumentation to the
library.
"""
struct ViewCountingArray{T,N,A} <: AbstractArray{T,N}
    parent::A
    views::Base.RefValue{Int}
end

ViewCountingArray(parent::A, views::Base.RefValue{Int}) where {A} =
    ViewCountingArray{eltype(A),ndims(A),A}(parent, views)

Base.size(array::ViewCountingArray) = size(array.parent)
Base.axes(array::ViewCountingArray) = axes(array.parent)
Base.IndexStyle(::Type{<:ViewCountingArray{T,N,A}}) where {T,N,A} =
    Base.IndexStyle(A)
Base.getindex(array::ViewCountingArray, indices...) =
    getindex(array.parent, indices...)

function Base.view(array::ViewCountingArray, indices...)
    array.views[] += 1
    return view(array.parent, indices...)
end

"Return `plan` with every reduced transition block instrumented for views."
function view_counting_plan(
    plan::BS.SitePlan{R,S},
    views::Base.RefValue{Int},
) where {R,S}
    transitions = map(plan.transitions) do transition
        BS.Transition{R,S}(
            transition.left_slot,
            transition.right_slot,
            ViewCountingArray(transition.B, views),
            transition.kernel,
        )
    end
    return BS.SitePlan{R,S}(
        plan.left,
        plan.physical,
        plan.purification,
        plan.right,
        plan.residual_left,
        plan.residual_right,
        plan.physical_basis,
        plan.purification_basis,
        transitions,
        plan.routes,
    )
end

"Count transition contractions needed to construct every local `(x,y)` branch."
function all_branch_transition_count(plan, factor)
    count = 0
    for physical in plan.physical_basis
        for purification in plan.purification_basis
            for route in BS._channel_routes(plan, physical, purification)
                left_sector = plan.residual_left.sectors[route.left_slot]
                TK.hasblock(factor, left_sector) || continue
                count += length(route.transition_indices)
            end
        end
    end
    return count
end
