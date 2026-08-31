struct TracedTangentMode <: PhysicalSamplingMode end
struct JointTangentMode <: AbstractSamplingMode end

const TangentSamplingMode = Union{TracedTangentMode,JointTangentMode}

"""One local state-machine step for the coherent single-insertion tangent state."""
struct TangentLocalPlan{AL,AR,B,P,LocalBasis,SymmetryBasis}
    left::AL
    right::AR
    insertion::B
    physical::P
    local_purification_basis::LocalBasis
    symmetry_basis::SymmetryBasis
end

struct TangentSitePlan{P} <: AbstractSitePlan
    step::P
    physical_row::Int
    purification_row::Int
end

struct TangentGlobalQPlan{B} <: AbstractSitePlan
    symmetry_basis::B
    output_row::Int
end

"""One deterministic sparse matrix over pairs of residual-sector blocks."""
struct TangentBlockMatrix{T}
    keys::Vector{NTuple{2,Int}}
    blocks::Vector{Matrix{T}}
    index::Dict{NTuple{2,Int},Int}
end

TangentBlockMatrix{T}() where {T} = TangentBlockMatrix{T}(
    NTuple{2,Int}[],
    Matrix{T}[],
    Dict{NTuple{2,Int},Int}(),
)

"""One residual input/output block of a fixed local-basis channel."""
struct TangentChannelBlock{T}
    left_slot::Int
    right_slot::Int
    data::Matrix{T}
end

"""
The q-resolved suffix Gram layers for a tangent prefix.

The fields are `I = R'R`, `K_q = R'T_q`, and `N_q = T_q'T_q`. Each quadratic
map stores only its compatible residual-sector pairs; in particular, `K_q` may
contain off-diagonal sector pairs carrying the fixed q charge. The q family is
kept as a vector and is never fused with a virtual space.
"""
struct TangentCompletionMetric{T}
    ranges::Vector{UnitRange{Int}}
    inserted::TangentBlockMatrix{T}
    cross::Vector{TangentBlockMatrix{T}}
    uninserted::Vector{TangentBlockMatrix{T}}
end

"""A sampled tangent prefix in one common purification-history factor basis."""
struct TangentPrefixFactor{T,M<:Matrix{T},A<:Array{T,3},I<:Vector{Int}}
    uninserted::M
    inserted::A
    symmetry_indices::I
end

mutable struct TangentSamplingWorkspace{
    M<:AbstractSamplingMode,T,R,Scratch,
}
    q::Vector{R}
    route_output::Matrix{T}
    identity::Matrix{T}
    matrix_scratch::Matrix{T}
    factor_scratch::Matrix{T}
    scratch::Scratch
end

"""A complete bank of uncompressed tangent branch factors for one prefix."""
mutable struct TangentBranchBundle{F}
    factors::Vector{Union{Nothing,F}}
end

function TangentBranchBundle(factors::AbstractVector{F}) where {F}
    owned = Vector{Union{Nothing,F}}(undef, length(factors))
    copyto!(owned, factors)
    return TangentBranchBundle{F}(owned)
end

mutable struct TangentSamplingRun{C,S}
    store::S
    root_completion::Union{Nothing,C}
    has_global_q_root::Bool
end

@inline _tensor_rank(::FiniteMPS.MPSTensor{R}) where {R} = R

@inline _local_purification_basis(plan::SitePlan{3}) =
    _TRIVIAL_PURIFICATION_BASIS
@inline _local_purification_basis(plan::SitePlan{4}) =
    plan.purification_basis

@inline _symmetry_basis(::SitePlan{3}, ::SitePlan{3}) =
    _TRIVIAL_PURIFICATION_BASIS
@inline _symmetry_basis(::SitePlan{3}, plan::SitePlan{4}) =
    plan.purification_basis
@inline _symmetry_basis(::SitePlan{4}, ::SitePlan{4}) =
    _TRIVIAL_PURIFICATION_BASIS
@inline _symmetry_basis(::SitePlan{4}, plan::SitePlan{5}) =
    last(plan.purification_basis)

@inline _base_auxiliary(plan::SitePlan{3}, ::Int) =
    only(plan.purification_basis)
@inline _base_auxiliary(plan::SitePlan{4}, local_index::Int) =
    plan.purification_basis[local_index]

@inline _insertion_auxiliary(
    ::SitePlan{3},
    plan::SitePlan{3},
    ::Int,
    ::Int,
) = only(plan.purification_basis)

@inline _insertion_auxiliary(
    ::SitePlan{3},
    plan::SitePlan{4},
    ::Int,
    symmetry::Int,
) = plan.purification_basis[symmetry]

@inline _insertion_auxiliary(
    ::SitePlan{4},
    plan::SitePlan{4},
    local_index::Int,
    ::Int,
) = plan.purification_basis[local_index]

@inline _insertion_auxiliary(
    ::SitePlan{4},
    plan::SitePlan{5},
    local_index::Int,
    symmetry::Int,
) = (
    first(plan.purification_basis)[local_index],
    last(plan.purification_basis)[symmetry],
)

@inline _residual_dimension(info::ResidualSpaceInfo) = sum(info.dimensions)

@inline function _residual_range(info::ResidualSpaceInfo, slot::Int)
    first_index = 1
    @inbounds for preceding in 1:(slot - 1)
        first_index += info.dimensions[preceding]
    end
    return first_index:(first_index + info.dimensions[slot] - 1)
end

function _same_space(actual, expected, description::AbstractString, site::Int)
    actual == expected || throw(DimensionMismatch(
        "site $site has incompatible $description spaces",
    ))
    return nothing
end

function _validate_tangent_tensors(state::FiniteMPSTangents.TangentMPS)
    L = length(state.B)
    L > 0 || throw(ArgumentError("cannot sample an empty TangentMPS"))
    length(state.base.Al) == L || throw(DimensionMismatch(
        "base.Al length $(length(state.base.Al)) does not match tangent length $L",
    ))
    length(state.base.Ar) == L || throw(DimensionMismatch(
        "base.Ar length $(length(state.base.Ar)) does not match tangent length $L",
    ))
    all(site -> isassigned(state.B, site), 1:L) || throw(ArgumentError(
        "all tangent tensors must be assigned before sampling",
    ))

    symmetry_space = nothing
    symmetry_presence = nothing
    operator_like = false
    for site in 1:L
        Al = state.base.Al[site]
        Ar = state.base.Ar[site]
        B = state.B[site]
        base_rank = _tensor_rank(Al)
        base_rank in (3, 4) || throw(ArgumentError(
            "site $site requires a rank-three or rank-four base tensor",
        ))
        operator_like |= base_rank == 4
        _tensor_rank(Ar) == base_rank || throw(ArgumentError(
            "site $site has mismatched Al/Ar ranks",
        ))
        insertion_rank = _tensor_rank(B)
        insertion_rank in (base_rank, base_rank + 1) || throw(ArgumentError(
            "site $site has unsupported base/tangent ranks " *
            "($base_rank, $insertion_rank)",
        ))

        for (tensor, name) in ((Ar, "Ar"), (B, "B"))
            _same_space(leftspace(tensor), leftspace(Al), "$name left virtual", site)
            _same_space(physspace(tensor), physspace(Al), "$name physical", site)
            _same_space(rightspace(tensor), rightspace(Al), "$name right virtual", site)
        end

        if base_rank == 4
            base_purification = purspace(Al)
            _same_space(purspace(Ar), base_purification, "Ar purification", site)
            insertion_purification = insertion_rank == 4 ?
                purspace(B) : first(purspaces(B))
            _same_space(
                insertion_purification,
                base_purification,
                "B local purification",
                site,
            )
        end

        has_symmetry = insertion_rank == base_rank + 1
        if symmetry_presence === nothing
            symmetry_presence = has_symmetry
        elseif has_symmetry != symmetry_presence
            throw(ArgumentError(
                "all tangent tensors must either carry the persistent symmetry " *
                "leg or omit it; site $site has inconsistent rank",
            ))
        end
        if has_symmetry
            current_symmetry = insertion_rank == 4 ?
                purspace(B) : last(purspaces(B))
            if symmetry_space === nothing
                symmetry_space = current_symmetry
            else
                _same_space(
                    current_symmetry,
                    symmetry_space,
                    "persistent tangent symmetry",
                    site,
                )
            end
        end
    end
    return (
        symmetry_space=symmetry_space,
        has_symmetry=something(symmetry_presence, false),
        operator_like=operator_like,
    )
end

function _tangent_sampling_spaces(state::FiniteMPSTangents.TangentMPS)
    spaces = Any[]
    for tensors in (state.base.Al, state.base.Ar, state.B)
        for tensor in tensors
            _append_sampling_spaces!(spaces, tensor)
        end
    end
    return spaces
end

function _compile_tangent_local_plans(
    state::FiniteMPSTangents.TangentMPS,
    ::Type{S},
    kernel_cache,
    residual,
) where {S}
    plans = Vector{TangentLocalPlan}(undef, length(state.B))
    for site in eachindex(state.B)
        left_plan = _compile_site(state.base.Al[site], S, kernel_cache, residual)
        right_plan = _compile_site(state.base.Ar[site], S, kernel_cache, residual)
        insertion_plan = _compile_site(state.B[site], S, kernel_cache, residual)
        plans[site] = TangentLocalPlan(
            left_plan,
            right_plan,
            insertion_plan,
            left_plan.physical,
            _local_purification_basis(left_plan),
            _symmetry_basis(left_plan, insertion_plan),
        )
    end
    return plans
end

function _tangent_scalar_type(state, local_plans)
    T = Union{}
    for tensors in (state.base.Al, state.base.Ar, state.B)
        for tensor in tensors
            T = promote_type(T, eltype(tensor.A))
        end
    end
    for step in local_plans
        T = promote_type(
            T,
            kernel_scalartype(step.left),
            kernel_scalartype(step.right),
            kernel_scalartype(step.insertion),
        )
    end
    T in (Float64, ComplexF64) || throw(ArgumentError(
        "only Float64 and ComplexF64 tangent workspaces are supported",
    ))
    return T
end

function _allocate_tangent_workspace(
    ::Type{M},
    ::Type{T},
    ::Type{R},
    ::Type{S},
    local_plans,
    has_global_q_root::Bool,
) where {M<:TangentSamplingMode,T,R,S}
    all_site_plans = Any[]
    for step in local_plans
        append!(all_site_plans, (step.left, step.right, step.insertion))
    end
    maximum_dimension = maximum(all_site_plans) do plan
        max(
            _residual_dimension(plan.residual_left),
            _residual_dimension(plan.residual_right),
        )
    end
    maximum_block = maximum(all_site_plans) do plan
        max(
            maximum(plan.residual_left.dimensions; init=0),
            maximum(plan.residual_right.dimensions; init=0),
        )
    end
    maximum_scratch = maximum(scratch_length, all_site_plans)
    maximum_local_purification = maximum(
        step -> length(step.local_purification_basis),
        local_plans,
    )
    symmetry_dimension = length(first(local_plans).symmetry_basis)
    maximum_history = maximum_local_purification *
                      (symmetry_dimension + 1) * maximum_dimension
    qmax = maximum(local_plans) do step
        local_dimension = M === JointTangentMode ?
                          length(step.local_purification_basis) : 1
        step.physical.fulldim * local_dimension
    end
    has_global_q_root && (qmax = max(qmax, symmetry_dimension))
    scratch = _allocate_scratch(S, T, maximum_scratch)
    return TangentSamplingWorkspace{M,T,R,typeof(scratch)}(
        zeros(R, qmax),
        zeros(T, maximum_block, maximum_block),
        zeros(T, maximum_block, maximum_block),
        zeros(T, maximum_block, maximum_block),
        zeros(T, maximum_block, maximum_history),
        scratch,
    )
end

function _clone_workspace(
    workspace::TangentSamplingWorkspace{M,T,R},
) where {M,T,R}
    scratch = _zero_like(workspace.scratch)
    return TangentSamplingWorkspace{M,T,R,typeof(scratch)}(
        zeros(R, length(workspace.q)),
        zeros(T, size(workspace.route_output)),
        zeros(T, size(workspace.identity)),
        zeros(T, size(workspace.matrix_scratch)),
        zeros(T, size(workspace.factor_scratch)),
        scratch,
    )
end

# The quadratic right sweep uses the same compiled `ChannelRoute`s as the
# prefix hot path, but materializes each nonzero residual block separately.
# No matrix with the full concatenated residual dimension is formed.
function _materialize_block_channel(
    workspace::TangentSamplingWorkspace{M,T},
    plan::SitePlan,
    physical::BasisInfo,
    purification,
) where {M,T}
    blocks = TangentChannelBlock{T}[]
    for route in _channel_routes(plan, physical, purification)
        rows = plan.residual_right.dimensions[route.right_slot]
        columns = plan.residual_left.dimensions[route.left_slot]
        identity = view(workspace.identity, 1:columns, 1:columns)
        fill!(identity, zero(T))
        @inbounds for diagonal in 1:columns
            identity[diagonal, diagonal] = one(T)
        end
        route_output = view(workspace.route_output, 1:rows, 1:columns)
        fill!(route_output, zero(T))
        _apply_route!(
            route_output,
            identity,
            plan,
            route,
            physical,
            purification,
            workspace.scratch,
        )
        push!(blocks, TangentChannelBlock{T}(
            route.left_slot,
            route.right_slot,
            Matrix(route_output),
        ))
    end
    return blocks
end

@inline function _tangent_block(
    matrix::TangentBlockMatrix,
    row_slot::Int,
    column_slot::Int,
)
    slot = get(matrix.index, (row_slot, column_slot), 0)
    iszero(slot) && return nothing
    return @inbounds matrix.blocks[slot]
end

function _tangent_destination_block!(
    matrix::TangentBlockMatrix{T},
    row_slot::Int,
    column_slot::Int,
    rows::Int,
    columns::Int,
) where {T}
    key = (row_slot, column_slot)
    slot = get(matrix.index, key, 0)
    if iszero(slot)
        push!(matrix.keys, key)
        push!(matrix.blocks, zeros(T, rows, columns))
        slot = length(matrix.blocks)
        matrix.index[key] = slot
    end
    return @inbounds matrix.blocks[slot]
end

@inline _tangent_middle_block(
    matrix::TangentBlockMatrix,
    row_slot::Int,
    column_slot::Int,
    ::Val{false},
) = _tangent_block(matrix, row_slot, column_slot)

@inline function _tangent_middle_block(
    matrix::TangentBlockMatrix,
    row_slot::Int,
    column_slot::Int,
    ::Val{true},
)
    block = _tangent_block(matrix, column_slot, row_slot)
    return block === nothing ? nothing : adjoint(block)
end

function _add_completion_transfer!(
    destination::TangentBlockMatrix{T},
    bra_channel::Vector{TangentChannelBlock{T}},
    middle::TangentBlockMatrix{T},
    ket_channel::Vector{TangentChannelBlock{T}},
    workspace::TangentSamplingWorkspace{M,T},
    middle_adjoint::Val{A},
) where {M,T,A}
    for bra in bra_channel, ket in ket_channel
        middle_block = _tangent_middle_block(
            middle,
            bra.right_slot,
            ket.right_slot,
            middle_adjoint,
        )
        middle_block === nothing && continue
        result = _tangent_destination_block!(
            destination,
            bra.left_slot,
            ket.left_slot,
            size(bra.data, 2),
            size(ket.data, 2),
        )
        temporary = view(
            workspace.matrix_scratch,
            1:size(bra.data, 2),
            1:size(middle_block, 2),
        )
        mul!(temporary, adjoint(bra.data), middle_block)
        mul!(result, temporary, ket.data, one(T), one(T))
    end
    return nothing
end

@inline function _add_completion_transfer!(
    destination::TangentBlockMatrix{T},
    bra_channel::Vector{TangentChannelBlock{T}},
    middle::TangentBlockMatrix{T},
    ket_channel::Vector{TangentChannelBlock{T}},
    workspace::TangentSamplingWorkspace{M,T},
) where {M,T}
    return _add_completion_transfer!(
        destination,
        bra_channel,
        middle,
        ket_channel,
        workspace,
        Val(false),
    )
end

function _tangent_residual_ranges(dimensions)
    ranges = Vector{UnitRange{Int}}(undef, length(dimensions))
    first_row = 1
    @inbounds for slot in eachindex(dimensions)
        last_row = first_row + dimensions[slot] - 1
        ranges[slot] = first_row:last_row
        first_row = last_row + 1
    end
    return ranges
end

function _retreat_completion_metric(
    step::TangentLocalPlan,
    metric::TangentCompletionMetric{T},
    workspace::TangentSamplingWorkspace{M,T},
) where {M,T}
    symmetry_dimension = length(step.symmetry_basis)
    inserted = TangentBlockMatrix{T}()
    cross = [TangentBlockMatrix{T}() for _ in 1:symmetry_dimension]
    uninserted = [TangentBlockMatrix{T}() for _ in 1:symmetry_dimension]

    for physical in step.left.physical_basis
        for local_index in eachindex(step.local_purification_basis)
            left_channel = _materialize_block_channel(
                workspace,
                step.left,
                physical,
                _base_auxiliary(step.left, local_index),
            )
            right_channel = _materialize_block_channel(
                workspace,
                step.right,
                physical,
                _base_auxiliary(step.right, local_index),
            )
            _add_completion_transfer!(
                inserted,
                right_channel,
                metric.inserted,
                right_channel,
                workspace,
            )

            for symmetry_index in eachindex(step.symmetry_basis)
                insertion_channel = _materialize_block_channel(
                    workspace,
                    step.insertion,
                    physical,
                    _insertion_auxiliary(
                        step.left,
                        step.insertion,
                        local_index,
                        symmetry_index,
                    ),
                )
                metric_cross = metric.cross[symmetry_index]
                metric_uninserted = metric.uninserted[symmetry_index]

                # K'_q = R' I B_q + R' K_q L.
                _add_completion_transfer!(
                    cross[symmetry_index],
                    right_channel,
                    metric.inserted,
                    insertion_channel,
                    workspace,
                )
                _add_completion_transfer!(
                    cross[symmetry_index],
                    right_channel,
                    metric_cross,
                    left_channel,
                    workspace,
                )

                # Same-site B'I B occurs once. The two K terms are the two
                # orientations of every different-site cross term.
                _add_completion_transfer!(
                    uninserted[symmetry_index],
                    left_channel,
                    metric_uninserted,
                    left_channel,
                    workspace,
                )
                _add_completion_transfer!(
                    uninserted[symmetry_index],
                    insertion_channel,
                    metric.inserted,
                    insertion_channel,
                    workspace,
                )
                _add_completion_transfer!(
                    uninserted[symmetry_index],
                    insertion_channel,
                    metric_cross,
                    left_channel,
                    workspace,
                )
                _add_completion_transfer!(
                    uninserted[symmetry_index],
                    left_channel,
                    metric_cross,
                    insertion_channel,
                    workspace,
                    Val(true),
                )
            end
        end
    end
    return TangentCompletionMetric(
        _tangent_residual_ranges(step.left.residual_left.dimensions),
        inserted,
        cross,
        uninserted,
    )
end

function _right_boundary_metric(::Type{T}, step::TangentLocalPlan) where {T}
    residual = step.right.residual_right
    dimension = _residual_dimension(residual)
    dimension == 1 || throw(ArgumentError(
        "the final right virtual space must be one-dimensional",
    ))
    inserted = TangentBlockMatrix{T}()
    for (slot, block_dimension) in pairs(residual.dimensions)
        block = _tangent_destination_block!(
            inserted,
            slot,
            slot,
            block_dimension,
            block_dimension,
        )
        @inbounds for diagonal in 1:block_dimension
            block[diagonal, diagonal] = one(T)
        end
    end
    symmetry_dimension = length(step.symmetry_basis)
    return TangentCompletionMetric(
        _tangent_residual_ranges(residual.dimensions),
        inserted,
        [TangentBlockMatrix{T}() for _ in 1:symmetry_dimension],
        [TangentBlockMatrix{T}() for _ in 1:symmetry_dimension],
    )
end

function _build_initial_tangent_factor(
    ::Type{T},
    boundary::Vector{T},
    step::TangentLocalPlan,
) where {T}
    full_left = step.left.left
    residual_left = step.left.residual_left
    length(full_left.sectors) == 1 && only(full_left.multiplicities) == 1 ||
        throw(ArgumentError(
            "the left virtual space must have one original sector with reduced " *
            "multiplicity one",
        ))
    dimension = _residual_dimension(residual_left)
    uninserted = zeros(T, dimension, 1)
    rows = only(residual_left.embeddings).rows
    @inbounds for row in eachindex(boundary)
        uninserted[rows[row], 1] = boundary[row]
    end
    symmetry_dimension = length(step.symmetry_basis)
    inserted = zeros(T, dimension, symmetry_dimension, 1)
    return TangentPrefixFactor(
        uninserted,
        inserted,
        collect(1:symmetry_dimension),
    )
end

function _build_tangent_plans(
    local_plans,
    ::Type{M},
    operator_like::Bool,
    has_symmetry::Bool,
) where {M<:TangentSamplingMode}
    L = length(local_plans)
    joint = M === JointTangentMode
    has_q_root = joint && has_symmetry
    layers = Vector{AbstractSitePlan}(undef, L + Int(has_q_root))
    offset = Int(has_q_root)
    purification_offset = joint && operator_like ? L : 0
    if has_q_root
        qrow = L + purification_offset + 1
        layers[1] = TangentGlobalQPlan(first(local_plans).symmetry_basis, qrow)
    end
    for site in 1:L
        purification_row = purification_offset == 0 ? 0 : L + site
        layers[offset + site] = TangentSitePlan(
            local_plans[site],
            site,
            purification_row,
        )
    end
    return layers
end

"""
    BornSampler(tangent::FiniteMPSTangents.TangentMPS;
                left_boundary=nothing, purified=true)

Compile the Hilbert-space state represented by a finite-MPS tangent vector.
The state is the coherent sum over all single-insertion sites. Construction
compiles symmetry-aware local routes, while every nonempty sampling batch owns
one right-to-left completion sweep and releases it at batch completion.

With `purified=true`, all local purification indices and persistent global q
are traced. With `purified=false`, local purification indices are sampled and
a present q is sampled once at a synthetic root. Joint output is `[x; y; q]`,
omitting absent groups. Rank-three sites in a mixed operator-valued base emit
the synthetic local value `y=1`.
"""
function BornSampler(
    state::FiniteMPSTangents.TangentMPS;
    left_boundary=nothing,
    purified::Bool=true,
)
    layout = _validate_tangent_tensors(state)
    sector_type = TK.sectortype(first(state.base.Al).A)
    S = _style_type(first(state.base.Al))
    for tensors in (state.base.Al, state.base.Ar, state.B), tensor in tensors
        TK.sectortype(tensor.A) === sector_type || throw(ArgumentError(
            "all tangent and base tensors must have the same TensorKit sector type",
        ))
        _style_type(tensor) === S || throw(ArgumentError(
            "all tangent and base tensors must use the same fusion style",
        ))
    end

    residual = _infer_residual_symmetry(
        sector_type,
        _tangent_sampling_spaces(state),
    )
    local_plans = _compile_tangent_local_plans(
        state,
        S,
        _kernel_cache(S),
        residual,
    )
    symmetry_dimension = length(first(local_plans).symmetry_basis)
    all(step -> length(step.symmetry_basis) == symmetry_dimension, local_plans) ||
        throw(DimensionMismatch(
            "persistent tangent symmetry dimensions differ between sites",
        ))

    M = purified ? TracedTangentMode : JointTangentMode
    T = _tangent_scalar_type(state, local_plans)
    Rprob = typeof(real(zero(T)))
    plans = _build_tangent_plans(
        local_plans,
        M,
        layout.operator_like,
        layout.has_symmetry,
    )
    workspace = _allocate_tangent_workspace(
        M,
        T,
        Rprob,
        S,
        local_plans,
        first(plans) isa TangentGlobalQPlan,
    )
    boundary = _prepare_left_boundary(
        left_boundary,
        T,
        first(local_plans).left.left.fulldim,
    )
    initial_factor = _build_initial_tangent_factor(
        T,
        boundary,
        first(local_plans),
    )
    return BornSampler{
        M,
        typeof(state),
        typeof(plans),
        typeof(initial_factor),
        typeof(workspace),
    }(
        state,
        plans,
        initial_factor,
        typeof(workspace)[workspace],
    )
end

@inline function _tangent_local_steps(sampler::BornSampler{M}) where {
    M<:TangentSamplingMode,
}
    offset = first(sampler.plans) isa TangentGlobalQPlan ? 1 : 0
    return map((offset + 1):length(sampler.plans)) do layer
        (sampler.plans[layer]::TangentSitePlan).step
    end
end

function _begin_sampling_run(
    sampler::BornSampler{M};
    disk::Bool,
) where {M<:TangentSamplingMode}
    steps = _tangent_local_steps(sampler)
    workspace = first(sampler.workspaces)
    completion = _right_boundary_metric(
        eltype(workspace.route_output),
        last(steps),
    )
    C = typeof(completion)
    store = TangentCompletionStore{C}(length(steps); disk=disk)
    try
        has_global_q_root = first(sampler.plans) isa TangentGlobalQPlan
        for site in reverse(eachindex(steps))
            # The first sampling layer stays in RAM. Without a synthetic q
            # root this is E₁ itself; with a q root, E₁ is needed by layer 2
            # and is therefore stored while the fully retreated root stays in
            # RAM.
            if site > 1 || has_global_q_root
                _put_completion!(store, site, completion)
            end
            if site > 1 || has_global_q_root
                completion = _retreat_completion_metric(
                    steps[site],
                    completion,
                    workspace,
                )
            end
        end
        return TangentSamplingRun{C,typeof(store)}(
            store,
            completion,
            has_global_q_root,
        )
    catch
        _cleanup_completion_store!(store)
        rethrow()
    end
end

function _take_sampling_completion!(run::TangentSamplingRun{C}, layer::Int) where {C}
    if layer == 1
        completion = run.root_completion
        run.root_completion = nothing
        return completion::C
    end
    site = run.has_global_q_root ? layer - 1 : layer
    return _take_completion!(run.store, site)
end

function _cleanup_sampling_run!(run::TangentSamplingRun)
    run.root_completion = nothing
    _cleanup_completion_store!(run.store)
    return nothing
end

function _advance_traced_tangent_prefix(
    workspace::TangentSamplingWorkspace{TracedTangentMode,T},
    factor::TangentPrefixFactor{T},
    plan::TangentSitePlan,
    selected::Int,
) where {T}
    step = plan.step
    physical = step.left.physical_basis[selected]
    local_dimension = length(step.local_purification_basis)
    symmetry_dimension = length(factor.symmetry_indices)
    history_dimension = size(factor.uninserted, 2)
    next_history = local_dimension * history_dimension
    right_dimension = _residual_dimension(step.left.residual_right)
    uninserted = zeros(T, right_dimension, next_history)
    inserted = zeros(T, right_dimension, symmetry_dimension, next_history)

    for local_index in 1:local_dimension
        columns = (
            ((local_index - 1) * history_dimension + 1):
            (local_index * history_dimension)
        )
        _apply_routes_to_factor!(
            view(uninserted, :, columns),
            factor.uninserted,
            step.left,
            physical,
            _base_auxiliary(step.left, local_index),
            workspace.scratch,
        )
        for (axis, symmetry_index) in pairs(factor.symmetry_indices)
            destination = view(inserted, :, axis, columns)
            _apply_routes_to_factor!(
                destination,
                view(factor.inserted, :, axis, :),
                step.right,
                physical,
                _base_auxiliary(step.right, local_index),
                workspace.scratch,
            )
            _apply_routes_to_factor!(
                destination,
                factor.uninserted,
                step.insertion,
                physical,
                _insertion_auxiliary(
                    step.left,
                    step.insertion,
                    local_index,
                    symmetry_index,
                ),
                workspace.scratch,
                one(T),
            )
        end
    end
    return TangentPrefixFactor(
        uninserted,
        inserted,
        factor.symmetry_indices,
    )
end

@inline function _tangent_joint_coordinates(plan::TangentSitePlan, selected::Int)
    local_dimension = plan.purification_row == 0 ?
                      1 : length(plan.step.local_purification_basis)
    physical = div(selected - 1, local_dimension) + 1
    local_index = rem(selected - 1, local_dimension) + 1
    return physical, local_index
end

function _advance_joint_tangent_prefix(
    workspace::TangentSamplingWorkspace{JointTangentMode,T},
    factor::TangentPrefixFactor{T},
    plan::TangentSitePlan,
    selected::Int,
) where {T}
    length(factor.symmetry_indices) == 1 || throw(ArgumentError(
        "joint tangent sampling must fix the global q before sampling sites",
    ))
    physical_index, local_index = _tangent_joint_coordinates(plan, selected)
    step = plan.step
    physical = step.left.physical_basis[physical_index]
    right_dimension = _residual_dimension(step.left.residual_right)
    history_dimension = size(factor.uninserted, 2)
    uninserted = zeros(T, right_dimension, history_dimension)
    inserted = zeros(T, right_dimension, 1, history_dimension)

    _apply_routes_to_factor!(
        uninserted,
        factor.uninserted,
        step.left,
        physical,
        _base_auxiliary(step.left, local_index),
        workspace.scratch,
    )
    destination = view(inserted, :, 1, :)
    _apply_routes_to_factor!(
        destination,
        view(factor.inserted, :, 1, :),
        step.right,
        physical,
        _base_auxiliary(step.right, local_index),
        workspace.scratch,
    )
    symmetry_index = only(factor.symmetry_indices)
    _apply_routes_to_factor!(
        destination,
        factor.uninserted,
        step.insertion,
        physical,
        _insertion_auxiliary(
            step.left,
            step.insertion,
            local_index,
            symmetry_index,
        ),
        workspace.scratch,
        one(T),
    )
    return TangentPrefixFactor(
        uninserted,
        inserted,
        factor.symmetry_indices,
    )
end

@inline _advance_tangent_prefix(
    workspace::TangentSamplingWorkspace{TracedTangentMode},
    factor,
    plan::TangentSitePlan,
    selected::Int,
) = _advance_traced_tangent_prefix(workspace, factor, plan, selected)

@inline _advance_tangent_prefix(
    workspace::TangentSamplingWorkspace{JointTangentMode},
    factor,
    plan::TangentSitePlan,
    selected::Int,
) = _advance_joint_tangent_prefix(workspace, factor, plan, selected)

function _restrict_tangent_symmetry(
    factor::TangentPrefixFactor{T},
    symmetry_index::Int,
) where {T}
    axis = symmetry_index
    inserted = Array{T,3}(
        undef,
        size(factor.inserted, 1),
        1,
        size(factor.inserted, 3),
    )
    copyto!(view(inserted, :, 1, :), view(factor.inserted, :, axis, :))
    return TangentPrefixFactor(
        copy(factor.uninserted),
        inserted,
        Int[symmetry_index],
    )
end

function _tangent_block_bilinear(
    workspace::TangentSamplingWorkspace{M,T},
    bra::AbstractMatrix,
    matrix::TangentBlockMatrix{T},
    ket::AbstractMatrix,
    ranges::Vector{UnitRange{Int}},
) where {M,T}
    columns = size(bra, 2)
    value = zero(T)
    @inbounds for slot in eachindex(matrix.keys)
        row_slot, column_slot = matrix.keys[slot]
        row_range = ranges[row_slot]
        column_range = ranges[column_slot]
        block = matrix.blocks[slot]
        temporary = view(
            workspace.factor_scratch,
            1:length(row_range),
            1:columns,
        )
        mul!(temporary, block, view(ket, column_range, :))
        value += dot(view(bra, row_range, :), temporary)
    end
    return value
end

function _tangent_completion_weight(
    workspace::TangentSamplingWorkspace{M,T},
    factor::TangentPrefixFactor{T},
    metric::TangentCompletionMetric{T},
) where {M,T}
    value = zero(T)

    for (axis, symmetry_index) in pairs(factor.symmetry_indices)
        inserted = view(factor.inserted, :, axis, :)
        value += _tangent_block_bilinear(
            workspace,
            factor.uninserted,
            metric.uninserted[symmetry_index],
            factor.uninserted,
            metric.ranges,
        )
        value += _tangent_block_bilinear(
            workspace,
            inserted,
            metric.inserted,
            inserted,
            metric.ranges,
        )
        value += 2 * real(_tangent_block_bilinear(
            workspace,
            inserted,
            metric.cross[symmetry_index],
            factor.uninserted,
            metric.ranges,
        ))
    end
    result = real(value)
    scale = max(one(result), abs(result))
    tolerance = 64 * eps(one(result)) * scale
    if result < zero(result)
        result >= -tolerance && return zero(result)
        throw(ArgumentError(
            "tangent contraction produced a materially negative Born " *
            "weight $result (roundoff tolerance $tolerance)",
        ))
    end
    return result
end

function _compute_weights_and_factors!(
    workspace::TangentSamplingWorkspace,
    factor::TangentPrefixFactor,
    plan::TangentSitePlan,
    completion::TangentCompletionMetric,
)
    count = _outcome_count(workspace, plan)
    factors = Vector{typeof(factor)}(undef, count)
    @inbounds for selected in 1:count
        next_factor = _advance_tangent_prefix(
            workspace,
            factor,
            plan,
            selected,
        )
        factors[selected] = next_factor
        workspace.q[selected] = _tangent_completion_weight(
            workspace,
            next_factor,
            completion,
        )
    end
    return factors
end

function _compute_weights_and_factors!(
    workspace::TangentSamplingWorkspace{JointTangentMode},
    factor::TangentPrefixFactor,
    plan::TangentGlobalQPlan,
    completion::TangentCompletionMetric,
)
    count = length(plan.symmetry_basis)
    factors = Vector{typeof(factor)}(undef, count)
    @inbounds for symmetry_index in 1:count
        next_factor = _restrict_tangent_symmetry(factor, symmetry_index)
        factors[symmetry_index] = next_factor
        workspace.q[symmetry_index] = _tangent_completion_weight(
            workspace,
            next_factor,
            completion,
        )
    end
    return factors
end

function _compute_weights!(
    workspace::TangentSamplingWorkspace,
    factor::TangentPrefixFactor,
    plan::Union{TangentSitePlan,TangentGlobalQPlan},
    completion::TangentCompletionMetric,
)
    _compute_weights_and_factors!(workspace, factor, plan, completion)
    return nothing
end

function _compress_tangent_factor(factor::TangentPrefixFactor{T}) where {T}
    rows = size(factor.uninserted, 1)
    symmetry_dimension = size(factor.inserted, 2)
    history_dimension = size(factor.uninserted, 2)
    maximum_support = (symmetry_dimension + 1) * rows
    history_dimension <= maximum_support && return factor

    stacked = Matrix{T}(undef, maximum_support, history_dimension)
    copyto!(view(stacked, 1:rows, :), factor.uninserted)
    for symmetry_axis in 1:symmetry_dimension
        destination = (
            (symmetry_axis * rows + 1):((symmetry_axis + 1) * rows)
        )
        copyto!(
            view(stacked, destination, :),
            view(factor.inserted, :, symmetry_axis, :),
        )
    end
    decomposition = qr(adjoint(stacked))
    compressed = Matrix(adjoint(Matrix(decomposition.R)))
    support = size(compressed, 2)
    uninserted = Matrix(view(compressed, 1:rows, 1:support))
    inserted = Array{T,3}(undef, rows, symmetry_dimension, support)
    for symmetry_axis in 1:symmetry_dimension
        source = (
            (symmetry_axis * rows + 1):((symmetry_axis + 1) * rows)
        )
        copyto!(
            view(inserted, :, symmetry_axis, :),
            view(compressed, source, 1:support),
        )
    end
    return TangentPrefixFactor(
        uninserted,
        inserted,
        factor.symmetry_indices,
    )
end

function _advance_built_tangent_factor!(factor, qselected)
    next_factor = _compress_tangent_factor(factor)
    scale = inv(sqrt(qselected))
    rmul!(next_factor.uninserted, scale)
    rmul!(next_factor.inserted, scale)
    return next_factor
end

@inline _outcome_count(
    ::TangentSamplingWorkspace{TracedTangentMode},
    plan::TangentSitePlan,
) = plan.step.physical.fulldim

@inline function _outcome_count(
    ::TangentSamplingWorkspace{JointTangentMode},
    plan::TangentSitePlan,
)
    local_dimension = plan.purification_row == 0 ?
                      1 : length(plan.step.local_purification_basis)
    return plan.step.physical.fulldim * local_dimension
end

@inline _outcome_count(
    ::TangentSamplingWorkspace{JointTangentMode},
    plan::TangentGlobalQPlan,
) = length(plan.symmetry_basis)

@inline function _tangent_configuration_length(sampler)
    maximum_row = 0
    for plan in sampler.plans
        if plan isa TangentGlobalQPlan
            maximum_row = max(maximum_row, plan.output_row)
        else
            site_plan = plan::TangentSitePlan
            maximum_row = max(
                maximum_row,
                site_plan.physical_row,
                site_plan.purification_row,
            )
        end
    end
    return maximum_row
end

@inline _configuration_length(sampler::BornSampler{TracedTangentMode}) =
    _tangent_configuration_length(sampler)
@inline _configuration_length(sampler::BornSampler{JointTangentMode}) =
    _tangent_configuration_length(sampler)

@inline function _store_outcome!(
    ::TangentSamplingWorkspace,
    configuration::AbstractVector,
    ::Int,
    ::Int,
    plan::TangentGlobalQPlan,
    selected::Int,
)
    configuration[plan.output_row] = selected
    return nothing
end

@inline function _store_outcome!(
    ::TangentSamplingWorkspace{TracedTangentMode},
    configuration::AbstractVector,
    ::Int,
    ::Int,
    plan::TangentSitePlan,
    selected::Int,
)
    configuration[plan.physical_row] = selected
    return nothing
end

@inline function _store_outcome!(
    ::TangentSamplingWorkspace{JointTangentMode},
    configuration::AbstractVector,
    ::Int,
    ::Int,
    plan::TangentSitePlan,
    selected::Int,
)
    physical, local_index = _tangent_joint_coordinates(plan, selected)
    configuration[plan.physical_row] = physical
    iszero(plan.purification_row) ||
        (configuration[plan.purification_row] = local_index)
    return nothing
end

@inline function _store_outcome!(
    ::TangentSamplingWorkspace,
    configuration::AbstractMatrix,
    ::Int,
    ::Int,
    shot::Int,
    plan::TangentGlobalQPlan,
    selected::Int,
)
    configuration[plan.output_row, shot] = selected
    return nothing
end

@inline function _store_outcome!(
    ::TangentSamplingWorkspace{TracedTangentMode},
    configuration::AbstractMatrix,
    ::Int,
    ::Int,
    shot::Int,
    plan::TangentSitePlan,
    selected::Int,
)
    configuration[plan.physical_row, shot] = selected
    return nothing
end

@inline function _store_outcome!(
    ::TangentSamplingWorkspace{JointTangentMode},
    configuration::AbstractMatrix,
    ::Int,
    ::Int,
    shot::Int,
    plan::TangentSitePlan,
    selected::Int,
)
    physical, local_index = _tangent_joint_coordinates(plan, selected)
    configuration[plan.physical_row, shot] = physical
    iszero(plan.purification_row) ||
        (configuration[plan.purification_row, shot] = local_index)
    return nothing
end

@inline _prefix_environment_type(
    ::BornSampler{M},
    ::Type{F},
) where {M<:TangentSamplingMode,F} = TangentBranchBundle{F}

function _new_prefix_cache(
    ::Type{E},
    ::Type{F},
    ::Type{R},
    capacity::Int;
    disk::Bool,
    maxsize::Int=1024,
) where {T,R,F<:TangentPrefixFactor{T},E}
    nodes = Vector{Union{Nothing,PrefixNode{R}}}(undef, capacity)
    fill!(nodes, nothing)
    resident_limit = disk ? maxsize : capacity
    directory = disk ? mktempdir(; prefix="Bornsampling-tangent-prefix-") : nothing
    return PrefixCache{T,R,F,E}(
        nodes,
        Dict{Int,E}(),
        resident_limit,
        directory,
        Threads.Atomic{Int}(0),
        ReentrantLock(),
        0,
    )
end

function _write_serialized_atomic!(path::String, value)
    temporary = path * ".tmp"
    try
        open(temporary, "w") do io
            serialize(io, value)
            flush(io)
        end
        mv(temporary, path; force=false)
    catch
        ispath(temporary) && rm(temporary; force=true)
        rethrow()
    end
    return nothing
end

function _write_environment_atomic!(
    cache::PrefixCache{T,R,F,TangentBranchBundle{F}},
    node::PrefixNode,
    bundle::TangentBranchBundle{F},
) where {T,R,F}
    @inbounds for selected in eachindex(bundle.factors)
        factor = bundle.factors[selected]
        factor === nothing && continue
        _write_serialized_atomic!(
            _branch_factor_path(cache, node.id, selected),
            factor,
        )
    end
    return nothing
end

function _read_tangent_branch(
    cache::PrefixCache{T,R,F,TangentBranchBundle{F}},
    node::PrefixNode,
    selected::Int,
) where {T,R,F}
    path = _branch_factor_path(cache, node.id, selected)
    factor = open(deserialize, path)::F
    rm(path; force=true)
    return factor
end

function _take_tangent_branch!(
    cache::PrefixCache{T,R,F,TangentBranchBundle{F}},
    node::PrefixNode,
    selected::Int,
) where {T,R,F}
    bundle = get(cache.resident, node.id, nothing)
    if bundle !== nothing
        owned = bundle.factors[selected]::F
        bundle.factors[selected] = nothing
        return owned
    end
    return _read_tangent_branch(cache, node, selected)
end

function _initialize_prefix_cache_with_completion!(
    sampler::BornSampler{M},
    workspace::TangentSamplingWorkspace{M},
    cache::PrefixCache{T,R,F,TangentBranchBundle{F}},
    completion::TangentCompletionMetric,
) where {M<:TangentSamplingMode,T,R,F}
    factor = sampler.initial_factor
    plan = first(sampler.plans)
    extendable = length(sampler.plans) > 1
    factors = if extendable
        _compute_weights_and_factors!(workspace, factor, plan, completion)
    else
        _compute_weights!(workspace, factor, plan, completion)
        nothing
    end
    outcome_count = _outcome_count(workspace, plan)
    _total_weight(workspace.q, outcome_count, 1)
    q = _copy_active_weights(workspace, outcome_count)
    root_id = _allocate_node_id!(cache)
    root = PrefixNode(
        id=root_id,
        log_probability=zero(eltype(q)),
        q=q,
        factor_space=nothing,
        extendable=extendable,
    )
    _set_node!(cache, root)
    extendable && _insert_resident!(
        cache,
        root,
        TangentBranchBundle(factors::Vector{F}),
    )
    return root
end

function _build_prefix_child_with_completion!(
    sampler::BornSampler{M},
    workspace::TangentSamplingWorkspace{M},
    current_cache::PrefixCache{T,R,F,TangentBranchBundle{F}},
    next_cache::PrefixCache{T,R,F,TangentBranchBundle{F}},
    parent::PrefixNode,
    selected::Int,
    site::Int,
    next_completion::TangentCompletionMetric,
) where {M<:TangentSamplingMode,T,R,F}
    qselected = parent.q[selected]
    z = _total_weight(parent.q, length(parent.q), site)
    child_log_probability =
        parent.log_probability + (log(qselected) - log(z))
    child_id = _allocate_node_id!(next_cache)

    raw_factor = _take_tangent_branch!(current_cache, parent, selected)
    factor = _advance_built_tangent_factor!(raw_factor, qselected)
    next_plan = sampler.plans[site + 1]
    extendable = site + 1 < length(sampler.plans)
    factors = if extendable
        _compute_weights_and_factors!(
            workspace,
            factor,
            next_plan,
            next_completion,
        )
    else
        _compute_weights!(workspace, factor, next_plan, next_completion)
        nothing
    end
    outcome_count = _outcome_count(workspace, next_plan)
    _total_weight(workspace.q, outcome_count, site + 1)
    q = _copy_active_weights(workspace, outcome_count)
    child = PrefixNode(
        id=child_id,
        log_probability=child_log_probability,
        q=q,
        factor_space=nothing,
        extendable=extendable,
    )
    _set_node!(next_cache, child)
    extendable && _admit_owned_prefix_factor!(
        next_cache,
        child,
        TangentBranchBundle(factors::Vector{F}),
    )
    return child
end

# The scalar API uses the same batch-owned completion lifecycle and numerical
# branch builders. It stays serial, like the ordinary scalar API; all batched
# calls continue through the one shared prefix-tree/worker scheduler.
function bornsample!(
    rng::Random.AbstractRNG,
    sampler::BornSampler{M},
    config::AbstractVector{Int},
) where {M<:TangentSamplingMode}
    Base.require_one_based_indexing(config)
    expected_length = _configuration_length(sampler)
    length(config) == expected_length || throw(DimensionMismatch(
        "config has length $(length(config)); expected $expected_length",
    ))

    workspace = first(sampler.workspaces)
    factor = sampler.initial_factor
    log_probability = zero(eltype(workspace.q))
    run = _begin_sampling_run(sampler; disk=false)
    try
        chain_length = length(sampler.plans)
        for layer in eachindex(sampler.plans)
            plan = sampler.plans[layer]
            completion = _take_sampling_completion!(run, layer)
            factors = _compute_weights_and_factors!(
                workspace,
                factor,
                plan,
                completion,
            )
            outcome_count = _outcome_count(workspace, plan)
            z = _total_weight(workspace.q, outcome_count, layer)
            selected = _draw_outcome(rng, workspace.q, z, outcome_count)
            qselected = workspace.q[selected]
            _store_outcome!(
                workspace,
                config,
                layer,
                chain_length,
                plan,
                selected,
            )
            log_probability += log(qselected) - log(z)
            if layer < chain_length
                factor = _advance_built_tangent_factor!(
                    factors[selected],
                    qselected,
                )
            end
        end
    finally
        _cleanup_sampling_run!(run)
    end
    return log_probability
end

function bornsample!(
    rng::Random.AbstractRNG,
    sampler::BornSampler{M},
) where {M<:TangentSamplingMode}
    configuration = Vector{Int}(undef, _configuration_length(sampler))
    log_probability = bornsample!(rng, sampler, configuration)
    return (
        configuration=configuration,
        log_probability=log_probability,
    )
end

function bornsample!(
    rng::Random.AbstractRNG,
    state::FiniteMPSTangents.TangentMPS;
    left_boundary=nothing,
    purified::Bool=true,
)
    return bornsample!(
        rng,
        BornSampler(
            state;
            left_boundary=left_boundary,
            purified=purified,
        ),
    )
end

function bornsample!(
    rng::Random.AbstractRNG,
    state::FiniteMPSTangents.TangentMPS,
    nshots::Int;
    left_boundary=nothing,
    purified::Bool=true,
    kwargs...,
)
    sampler = BornSampler(
        state;
        left_boundary=left_boundary,
        purified=purified,
    )
    return bornsample!(rng, sampler, nshots; kwargs...)
end
