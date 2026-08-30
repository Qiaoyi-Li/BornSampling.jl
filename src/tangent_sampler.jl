struct TangentMode <: PhysicalSamplingMode end

"""
One local tensor-network state-machine step for a tangent MPS.

`left` propagates a prefix before its unique tangent insertion, `right`
propagates a prefix after the insertion, and `insertion` changes between those
two states.  Local purification and persistent symmetry bases are deliberately
separate.
"""
struct TangentLocalPlan{AL,AR,B,P,LocalBasis,SymmetryBasis}
    left::AL
    right::AR
    insertion::B
    physical::P
    local_purification_basis::LocalBasis
    symmetry_basis::SymmetryBasis
end

"""
The three suffix Gram layers needed to complete a factorized tangent prefix.

For suffix maps `R` (insertion already occurred) and `T_q` (insertion still in
the suffix), the fields represent `R'R`, `R'T_q`, and `sum_q T_q'T_q`.
The persistent symmetry index is the third axis of `cross` and is never fused
with a virtual space.
"""
struct TangentCompletionMetric{T,M<:Matrix{T},A<:Array{T,3}}
    inserted::M
    cross::A
    uninserted::M
end

struct TangentSitePlan{P,E} <: AbstractSitePlan
    step::P
    completion::E
end

"""
A sampled tangent prefix in a common purification-history factor basis.

`uninserted[:, h]` is the prefix amplitude before the tangent insertion.
`inserted[:, q, h]` is the coherent sum of all insertions already encountered.
Both fields always undergo the same history-space compression.
"""
struct TangentPrefixFactor{T,M<:Matrix{T},A<:Array{T,3}}
    uninserted::M
    inserted::A
end

mutable struct TangentSamplingWorkspace{T,R,Scratch}
    q::Vector{R}
    route_output::Matrix{T}
    identity::Matrix{T}
    left_channel::Matrix{T}
    right_channel::Matrix{T}
    insertion_channel::Matrix{T}
    matrix_scratch::Matrix{T}
    factor_scratch::Matrix{T}
    scratch::Scratch
end

"""A physical-branch bank for one tangent sampled prefix."""
mutable struct TangentBranchBundle{F}
    factors::Vector{Union{Nothing,F}}
end

function TangentBranchBundle(factors::AbstractVector{F}) where {F}
    owned = Vector{Union{Nothing,F}}(undef, length(factors))
    copyto!(owned, factors)
    return TangentBranchBundle{F}(owned)
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
) = (first(plan.purification_basis)[local_index],
     last(plan.purification_basis)[symmetry])

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
    for site in 1:L
        Al = state.base.Al[site]
        Ar = state.base.Ar[site]
        B = state.B[site]
        base_rank = _tensor_rank(Al)
        base_rank in (3, 4) || throw(ArgumentError(
            "site $site requires a rank-three or rank-four base tensor",
        ))
        _tensor_rank(Ar) == base_rank ||
            throw(ArgumentError(
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
            current_symmetry = insertion_rank == 4 ? purspace(B) : last(purspaces(B))
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
    return symmetry_space
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
    L = length(state.B)
    plans = Vector{TangentLocalPlan}(undef, L)
    for site in 1:L
        left_plan = _compile_site(state.base.Al[site], S, kernel_cache, residual)
        right_plan = _compile_site(state.base.Ar[site], S, kernel_cache, residual)
        insertion_plan = _compile_site(state.B[site], S, kernel_cache, residual)
        local_basis = _local_purification_basis(left_plan)
        symmetry_basis = _symmetry_basis(left_plan, insertion_plan)
        plans[site] = TangentLocalPlan(
            left_plan,
            right_plan,
            insertion_plan,
            left_plan.physical,
            local_basis,
            symmetry_basis,
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
    ::Type{T},
    ::Type{R},
    ::Type{S},
    local_plans,
) where {T,R,S}
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
    qmax = maximum(step -> step.physical.fulldim, local_plans)
    scratch = _allocate_scratch(S, T, maximum_scratch)
    return TangentSamplingWorkspace{T,R,typeof(scratch)}(
        zeros(R, qmax),
        zeros(T, maximum_block, maximum_block),
        zeros(T, maximum_block, maximum_block),
        zeros(T, maximum_dimension, maximum_dimension),
        zeros(T, maximum_dimension, maximum_dimension),
        zeros(T, maximum_dimension, maximum_dimension),
        zeros(T, maximum_dimension, maximum_dimension),
        zeros(T, maximum_dimension, maximum_history),
        scratch,
    )
end

function _clone_workspace(
    workspace::TangentSamplingWorkspace{T,R},
) where {T,R}
    scratch = _zero_like(workspace.scratch)
    return TangentSamplingWorkspace{T,R,typeof(scratch)}(
        zeros(R, length(workspace.q)),
        zeros(T, size(workspace.route_output)),
        zeros(T, size(workspace.identity)),
        zeros(T, size(workspace.left_channel)),
        zeros(T, size(workspace.right_channel)),
        zeros(T, size(workspace.insertion_channel)),
        zeros(T, size(workspace.matrix_scratch)),
        zeros(T, size(workspace.factor_scratch)),
        scratch,
    )
end

function _materialize_channel!(
    storage::Matrix,
    workspace::TangentSamplingWorkspace,
    plan::SitePlan,
    physical::BasisInfo,
    purification,
)
    left_dimension = _residual_dimension(plan.residual_left)
    right_dimension = _residual_dimension(plan.residual_right)
    channel = view(storage, 1:right_dimension, 1:left_dimension)
    fill!(channel, zero(eltype(channel)))

    for route in _channel_routes(plan, physical, purification)
        rows = plan.residual_right.dimensions[route.right_slot]
        columns = plan.residual_left.dimensions[route.left_slot]
        identity = view(workspace.identity, 1:columns, 1:columns)
        fill!(identity, zero(eltype(identity)))
        @inbounds for diagonal in 1:columns
            identity[diagonal, diagonal] = one(eltype(identity))
        end
        route_output = view(workspace.route_output, 1:rows, 1:columns)
        fill!(route_output, zero(eltype(route_output)))
        _apply_route!(
            route_output,
            identity,
            plan,
            route,
            physical,
            purification,
            workspace.scratch,
        )
        right_rows = _residual_range(plan.residual_right, route.right_slot)
        left_rows = _residual_range(plan.residual_left, route.left_slot)
        copyto!(view(channel, right_rows, left_rows), route_output)
    end
    return channel
end

function _add_completion_transfer!(
    destination,
    bra_channel,
    middle,
    ket_channel,
    workspace::TangentSamplingWorkspace,
)
    left_dimension = size(bra_channel, 2)
    middle_dimension = size(bra_channel, 1)
    right_dimension = size(ket_channel, 2)
    temporary = view(
        workspace.matrix_scratch,
        1:left_dimension,
        1:size(middle, 2),
    )
    mul!(temporary, adjoint(bra_channel), middle)
    mul!(destination, temporary, ket_channel, one(eltype(destination)), one(eltype(destination)))
    size(destination) == (left_dimension, right_dimension) || throw(DimensionMismatch(
        "completion-transfer destination has inconsistent dimensions",
    ))
    middle_dimension == size(middle, 1) || throw(DimensionMismatch(
        "completion metric and local channel have inconsistent dimensions",
    ))
    return nothing
end

function _retreat_completion_metric(
    step::TangentLocalPlan,
    metric::TangentCompletionMetric{T},
    workspace::TangentSamplingWorkspace{T},
) where {T}
    left_dimension = _residual_dimension(step.left.residual_left)
    symmetry_dimension = length(step.symmetry_basis)
    inserted = zeros(T, left_dimension, left_dimension)
    cross = zeros(T, left_dimension, left_dimension, symmetry_dimension)
    uninserted = zeros(T, left_dimension, left_dimension)

    # `cross[:, :, q]` is G₁₀(q) = R†T(q), rather than its adjoint.
    # `metric.uninserted` already contains the persistent-q trace, so the
    # L†G₀₀L transfer belongs outside the q loop below.

    for physical in step.left.physical_basis
        for local_index in eachindex(step.local_purification_basis)
            left_auxiliary = _base_auxiliary(step.left, local_index)
            right_auxiliary = _base_auxiliary(step.right, local_index)
            left_channel = _materialize_channel!(
                workspace.left_channel,
                workspace,
                step.left,
                physical,
                left_auxiliary,
            )
            right_channel = _materialize_channel!(
                workspace.right_channel,
                workspace,
                step.right,
                physical,
                right_auxiliary,
            )

            _add_completion_transfer!(
                inserted,
                right_channel,
                metric.inserted,
                right_channel,
                workspace,
            )
            _add_completion_transfer!(
                uninserted,
                left_channel,
                metric.uninserted,
                left_channel,
                workspace,
            )

            for symmetry_index in eachindex(step.symmetry_basis)
                insertion_auxiliary = _insertion_auxiliary(
                    step.left,
                    step.insertion,
                    local_index,
                    symmetry_index,
                )
                insertion_channel = _materialize_channel!(
                    workspace.insertion_channel,
                    workspace,
                    step.insertion,
                    physical,
                    insertion_auxiliary,
                )
                cross_slice = view(cross, :, :, symmetry_index)
                metric_cross = view(metric.cross, :, :, symmetry_index)

                _add_completion_transfer!(
                    cross_slice,
                    right_channel,
                    metric.inserted,
                    insertion_channel,
                    workspace,
                )
                _add_completion_transfer!(
                    cross_slice,
                    right_channel,
                    metric_cross,
                    left_channel,
                    workspace,
                )
                _add_completion_transfer!(
                    uninserted,
                    insertion_channel,
                    metric.inserted,
                    insertion_channel,
                    workspace,
                )
                _add_completion_transfer!(
                    uninserted,
                    insertion_channel,
                    metric_cross,
                    left_channel,
                    workspace,
                )
                _add_completion_transfer!(
                    uninserted,
                    left_channel,
                    adjoint(metric_cross),
                    insertion_channel,
                    workspace,
                )
            end
        end
    end
    return TangentCompletionMetric(inserted, cross, uninserted)
end

function _right_boundary_metric(::Type{T}, step::TangentLocalPlan) where {T}
    dimension = _residual_dimension(step.right.residual_right)
    dimension == 1 || throw(ArgumentError(
        "the final right virtual space must be one-dimensional",
    ))
    symmetry_dimension = length(step.symmetry_basis)
    return TangentCompletionMetric(
        Matrix{T}(I, dimension, dimension),
        zeros(T, dimension, dimension, symmetry_dimension),
        zeros(T, dimension, dimension),
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
    inserted = zeros(T, dimension, length(step.symmetry_basis), 1)
    return TangentPrefixFactor(uninserted, inserted)
end

"""
    BornSampler(tangent::FiniteMPSTangents.TangentMPS;
                left_boundary=nothing, purified=true)

Compile the Hilbert-space state represented by a finite-MPS tangent vector for
repeated physical Born sampling. The tangent state is the coherent sum of its
single-insertion terms. Local MPO purification legs and the persistent tangent
symmetry leg are traced, so every returned configuration contains exactly one
physical index per site; consequently `purified=false` is unsupported.

Unlike the direct `MPS`/`MPO` constructor, this method neither canonicalizes nor
mutates the tangent or its base point. The base must already satisfy the
canonical-form contract imposed by `FiniteMPSTangents.BaseMPS`. Compiled
contraction plans retain views into its tensor blocks, so neither object may be
modified while the sampler is in use.
"""
function BornSampler(
    state::FiniteMPSTangents.TangentMPS;
    left_boundary=nothing,
    purified::Bool=true,
)
    purified || throw(ArgumentError(
        "TangentMPS sampling always traces local purification and persistent " *
        "symmetry legs; purified=false is not supported",
    ))
    _validate_tangent_tensors(state)
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
    kernel_cache = _kernel_cache(S)
    local_plans = _compile_tangent_local_plans(
        state,
        S,
        kernel_cache,
        residual,
    )
    symmetry_dimension = length(first(local_plans).symmetry_basis)
    all(step -> length(step.symmetry_basis) == symmetry_dimension, local_plans) ||
        throw(DimensionMismatch(
            "persistent tangent symmetry dimensions differ between sites",
        ))

    T = _tangent_scalar_type(state, local_plans)
    Rprob = typeof(real(zero(T)))
    workspace = _allocate_tangent_workspace(T, Rprob, S, local_plans)

    completion = _right_boundary_metric(T, last(local_plans))
    completions = Vector{typeof(completion)}(undef, length(local_plans))
    for site in reverse(eachindex(local_plans))
        completions[site] = completion
        completion = _retreat_completion_metric(
            local_plans[site],
            completion,
            workspace,
        )
    end

    plans = Vector{AbstractSitePlan}(undef, length(local_plans))
    for site in eachindex(local_plans)
        plans[site] = TangentSitePlan(local_plans[site], completions[site])
    end
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
        TangentMode,
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

function _advance_tangent_prefix(
    workspace::TangentSamplingWorkspace{T},
    factor::TangentPrefixFactor{T},
    plan::TangentSitePlan,
    selected::Int,
) where {T}
    step = plan.step
    physical = step.left.physical_basis[selected]
    local_dimension = length(step.local_purification_basis)
    symmetry_dimension = length(step.symmetry_basis)
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
        left_channel = _materialize_channel!(
            workspace.left_channel,
            workspace,
            step.left,
            physical,
            _base_auxiliary(step.left, local_index),
        )
        mul!(view(uninserted, :, columns), left_channel, factor.uninserted)

        right_channel = _materialize_channel!(
            workspace.right_channel,
            workspace,
            step.right,
            physical,
            _base_auxiliary(step.right, local_index),
        )
        for symmetry_index in 1:symmetry_dimension
            destination = view(inserted, :, symmetry_index, columns)
            mul!(
                destination,
                right_channel,
                view(factor.inserted, :, symmetry_index, :),
            )
            insertion_channel = _materialize_channel!(
                workspace.insertion_channel,
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
            mul!(
                destination,
                insertion_channel,
                factor.uninserted,
                one(T),
                one(T),
            )
        end
    end
    return TangentPrefixFactor(uninserted, inserted)
end

function _tangent_completion_weight(
    workspace::TangentSamplingWorkspace{T},
    factor::TangentPrefixFactor{T},
    metric::TangentCompletionMetric{T},
) where {T}
    rows = size(factor.uninserted, 1)
    columns = size(factor.uninserted, 2)
    temporary = view(workspace.factor_scratch, 1:rows, 1:columns)
    value = zero(T)

    mul!(temporary, metric.uninserted, factor.uninserted)
    value += dot(factor.uninserted, temporary)
    for symmetry_index in axes(factor.inserted, 2)
        inserted = view(factor.inserted, :, symmetry_index, :)
        mul!(temporary, metric.inserted, inserted)
        value += dot(inserted, temporary)
        mul!(
            temporary,
            view(metric.cross, :, :, symmetry_index),
            factor.uninserted,
        )
        value += 2 * real(dot(inserted, temporary))
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
)
    count = plan.step.physical.fulldim
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
            plan.completion,
        )
    end
    return factors
end

function _compute_weights!(
    workspace::TangentSamplingWorkspace,
    factor::TangentPrefixFactor,
    plan::TangentSitePlan,
)
    _compute_weights_and_factors!(workspace, factor, plan)
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
    for symmetry_index in 1:symmetry_dimension
        destination = (
            (symmetry_index * rows + 1):((symmetry_index + 1) * rows)
        )
        copyto!(
            view(stacked, destination, :),
            view(factor.inserted, :, symmetry_index, :),
        )
    end
    decomposition = qr(adjoint(stacked))
    compressed = Matrix(adjoint(Matrix(decomposition.R)))
    support = size(compressed, 2)
    uninserted = Matrix(view(compressed, 1:rows, 1:support))
    inserted = Array{T,3}(undef, rows, symmetry_dimension, support)
    for symmetry_index in 1:symmetry_dimension
        source = (
            (symmetry_index * rows + 1):((symmetry_index + 1) * rows)
        )
        copyto!(
            view(inserted, :, symmetry_index, :),
            view(compressed, source, 1:support),
        )
    end
    return TangentPrefixFactor(uninserted, inserted)
end

function _advance_built_tangent_factor!(factor, qselected)
    next_factor = _compress_tangent_factor(factor)
    scale = inv(sqrt(qselected))
    rmul!(next_factor.uninserted, scale)
    rmul!(next_factor.inserted, scale)
    return next_factor
end

@inline _outcome_count(
    ::TangentSamplingWorkspace,
    plan::TangentSitePlan,
) = plan.step.physical.fulldim

@inline function _store_outcome!(
    ::TangentSamplingWorkspace,
    configuration::AbstractVector,
    site::Int,
    chain_length::Int,
    plan::TangentSitePlan,
    selected::Int,
)
    configuration[site] = selected
    return nothing
end

@inline function _store_outcome!(
    ::TangentSamplingWorkspace,
    configuration::AbstractMatrix,
    site::Int,
    chain_length::Int,
    shot::Int,
    plan::TangentSitePlan,
    selected::Int,
)
    configuration[site, shot] = selected
    return nothing
end

function _sample_site!(
    rng,
    workspace::TangentSamplingWorkspace,
    factor::TangentPrefixFactor,
    plan::TangentSitePlan,
    config,
    site::Int,
    chain_length::Int,
)
    factors = _compute_weights_and_factors!(workspace, factor, plan)
    outcome_count = _outcome_count(workspace, plan)
    z = _total_weight(workspace.q, outcome_count, site)
    selected = _draw_outcome(rng, workspace.q, z, outcome_count)
    qselected = workspace.q[selected]
    next_factor = _advance_built_tangent_factor!(
        factors[selected],
        qselected,
    )
    _store_outcome!(
        workspace,
        config,
        site,
        chain_length,
        plan,
        selected,
    )
    return log(qselected) - log(z), next_factor
end

@inline _prefix_environment_type(
    ::BornSampler{TangentMode},
    ::Type{F},
) where {F} = TangentBranchBundle{F}

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
    factor = open(deserialize, path)
    rm(path; force=true)
    factor isa F || throw(ArgumentError(
        "cached tangent branch has type $(typeof(factor)); expected $F",
    ))
    return factor::F
end

function _take_tangent_branch!(
    cache::PrefixCache{T,R,F,TangentBranchBundle{F}},
    node::PrefixNode,
    selected::Int,
) where {T,R,F}
    bundle = get(cache.resident, node.id, nothing)
    if bundle !== nothing
        owned = bundle.factors[selected]
        owned === nothing && error(
            "tangent branch $selected of prefix $(node.id) was already consumed",
        )
        bundle.factors[selected] = nothing
        return owned::F
    end
    return _read_tangent_branch(cache, node, selected)
end

function _initialize_prefix_cache!(
    sampler::BornSampler{TangentMode},
    workspace::TangentSamplingWorkspace,
    cache::PrefixCache{T,R,F,TangentBranchBundle{F}},
) where {T,R,F}
    factor = sampler.initial_factor
    plan = first(sampler.plans)::TangentSitePlan
    extendable = length(sampler.plans) > 1
    factors = if extendable
        _compute_weights_and_factors!(workspace, factor, plan)
    else
        _compute_weights!(workspace, factor, plan)
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

function _build_prefix_child!(
    sampler::BornSampler{TangentMode},
    workspace::TangentSamplingWorkspace,
    current_cache::PrefixCache{T,R,F,TangentBranchBundle{F}},
    next_cache::PrefixCache{T,R,F,TangentBranchBundle{F}},
    parent::PrefixNode,
    selected::Int,
    site::Int,
) where {T,R,F}
    plan = sampler.plans[site]::TangentSitePlan
    qselected = parent.q[selected]
    z = _total_weight(parent.q, length(parent.q), site)
    child_log_probability =
        parent.log_probability + (log(qselected) - log(z))
    child_id = _allocate_node_id!(next_cache)

    raw_factor = _take_tangent_branch!(current_cache, parent, selected)
    factor = _advance_built_tangent_factor!(raw_factor, qselected)
    next_plan = sampler.plans[site + 1]::TangentSitePlan
    extendable = site + 1 < length(sampler.plans)
    factors = if extendable
        _compute_weights_and_factors!(workspace, factor, next_plan)
    else
        _compute_weights!(workspace, factor, next_plan)
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

"""
    bornsample!(rng, tangent::FiniteMPSTangents.TangentMPS;
                left_boundary=nothing, purified=true)

Compile the tangent vector's naturally isomorphic Hilbert-space state and draw
one physical snapshot.  Local MPO purification legs and a persistent tangent
symmetry leg are traced.  The input tangent and its base point are not mutated.
`purified=false` is unsupported because these auxiliary legs are not sampling
outcomes in the tangent-state interface.
"""
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
