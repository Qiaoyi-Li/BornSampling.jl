abstract type AbstractSamplingMode end
abstract type PhysicalSamplingMode <: AbstractSamplingMode end

struct MPSMode <: PhysicalSamplingMode end
struct TracedMPOMode <: PhysicalSamplingMode end
struct JointMPOMode <: AbstractSamplingMode end

"""Reusable numerical scratch owned by one sampling worker task."""
mutable struct SamplingWorkspace{M<:AbstractSamplingMode,T,Rprob,Scratch}
    route_output::Matrix{T}
    scratch::Scratch
    q::Vector{Rprob}
    branch_columns::Vector{Int}
end

"""
    BornSampler(state::Union{FiniteMPS.MPS,FiniteMPS.MPO};
                left_boundary=nothing, purified=true)

Compile an in-memory `FiniteMPS.MPS` or `FiniteMPS.MPO` for repeated sequential
Born sampling. An `MPS` is sampled as a rank-three physical amplitude. For an
`MPO`, the default `purified=true` traces every rank-four tensor's first domain
leg and samples the exact physical marginal. With `purified=false`, that leg is
sampled jointly with the physical leg. Rank-three sites inside an `MPO` use a
one-dimensional synthetic purification leg in either mode.

Construction calls `FiniteMPS.canonicalize!(state, 1)` in place, compiles all
fusion-tree contraction plans, and allocates one reusable workspace. The
sampler keeps views into `state`, so the state and its tensor data must not be
modified after construction.

The left virtual space must contain one original sector with reduced
multiplicity one. If its full dimension exceeds one, `left_boundary` must
provide the pure state inside that sector. Mixed boundaries and coherent
superpositions across residual charges are not supported.

One public batch call may use multiple internal worker tasks, each with an
independent workspace. Concurrent external calls using the same sampler are
not synchronized and are the caller's responsibility.
"""
mutable struct BornSampler{M<:AbstractSamplingMode,State,Plans,Factor,Workspace}
    state::State
    plans::Plans
    initial_factor::Factor
    workspaces::Vector{Workspace}
end

function BornSampler(
    state::FiniteMPS.MPS;
    left_boundary=nothing,
    purified::Bool=true,
)
    # `purified` is accepted for a uniform public call signature and has no
    # effect on an MPS, whose local tensors have no purification leg.
    return _construct_sampler(state, MPSMode, left_boundary)
end

function BornSampler(
    state::FiniteMPS.MPO;
    left_boundary=nothing,
    purified::Bool=true,
)
    mode = purified ? TracedMPOMode : JointMPOMode
    return _construct_sampler(state, mode, left_boundary)
end


function _construct_sampler(
    state,
    ::Type{M},
    left_boundary,
) where {M<:AbstractSamplingMode}
    length(state) > 0 || throw(ArgumentError("cannot sample an empty DenseMPS"))
    state.A isa Vector || throw(ArgumentError(
        "BornSampler supports only in-memory DenseMPS storage",
    ))

    _validate_local_tensors(state, M)

    FiniteMPS.canonicalize!(state, 1)

    first_tensor = state[1]
    S = _style_type(first_tensor)
    return _build_sampler(M, S, state, left_boundary)
end

function _validate_local_tensors(state::FiniteMPS.MPS, ::Type{MPSMode})
    for tensor in state.A
        tensor isa FiniteMPS.MPSTensor{3} || throw(ArgumentError(
            "FiniteMPS.MPS sampling requires MPSTensor{3} at every site",
        ))
    end
    return nothing
end

function _validate_local_tensors(state::FiniteMPS.MPO, ::Type{M}) where {
    M<:Union{TracedMPOMode,JointMPOMode},
}
    for tensor in state.A
        tensor isa Union{FiniteMPS.MPSTensor{3},FiniteMPS.MPSTensor{4}} ||
            throw(ArgumentError(
                "FiniteMPS.MPO local tensors must be MPSTensor{3} or MPSTensor{4}",
            ))
    end
    return nothing
end

function _build_sampler(
    ::Type{M},
    ::Type{S},
    state::FiniteMPS.DenseMPS,
    left_boundary,
) where {M<:AbstractSamplingMode,S}
    L = length(state)
    sector_type = TK.sectortype(state[1].A)
    residual = _infer_residual_symmetry(
        sector_type,
        _sampling_spaces(state),
    )
    kernel_cache = _kernel_cache(S)

    first_plan = _compile_site(state[1], S, kernel_cache, residual)
    plans = Vector{AbstractSitePlan}(undef, L)
    plans[1] = first_plan
    for site in 2:L
        tensor = state[site]
        TK.sectortype(tensor.A) === sector_type || throw(ArgumentError(
            "all local tensors must have the same TensorKit sector type",
        ))
        plans[site] = _compile_site(tensor, S, kernel_cache, residual)
    end
    plans[end].right.fulldim == 1 || throw(ArgumentError(
        "the final right virtual space must be one-dimensional",
    ))

    T = FiniteMPS.scalartype(state)
    for plan in plans
        T = promote_type(T, kernel_scalartype(plan))
    end
    T in (Float64, ComplexF64) || throw(ArgumentError(
        "only Float64 and ComplexF64 contraction workspaces are supported",
    ))
    Rprob = typeof(real(zero(T)))

    qmax = maximum(plan -> _outcome_count(M, plan), plans)
    scratch_max = maximum(scratch_length, plans)
    residual_block_max = maximum(plans) do plan
        max(
            maximum(plan.residual_left.dimensions; init=0),
            maximum(plan.residual_right.dimensions; init=0),
        )
    end
    residual_sector_max = maximum(plans) do plan
        max(length(plan.residual_left.sectors), length(plan.residual_right.sectors))
    end

    boundary = _prepare_left_boundary(left_boundary, T, first_plan.left.fulldim)
    initial_factor = _build_initial_factor(
        T,
        boundary,
        first_plan.left,
        first_plan.residual_left,
    )
    workspace = _allocate_workspace(
        M,
        S,
        T,
        Rprob,
        residual_block_max,
        residual_sector_max,
        qmax,
        scratch_max,
    )

    return BornSampler{
        M,typeof(state),typeof(plans),typeof(initial_factor),typeof(workspace),
    }(
        state,
        plans,
        initial_factor,
        typeof(workspace)[workspace],
    )
end

function _prepare_left_boundary(left_boundary, ::Type{T}, dimension::Int) where {T}
    if left_boundary === nothing
        dimension == 1 || throw(ArgumentError(
            "left_boundary is required when the full left virtual dimension is $dimension",
        ))
        return T[one(T)]
    end
    left_boundary isa AbstractVector || throw(ArgumentError(
        "left_boundary must be a pure-state vector; mixed boundaries are not supported",
    ))
    length(left_boundary) == dimension || throw(DimensionMismatch(
        "left_boundary has length $(length(left_boundary)); expected $dimension",
    ))
    boundary = try
        Vector{T}(left_boundary)
    catch error
        error isa Union{InexactError,MethodError,ArgumentError} || rethrow()
        throw(ArgumentError("left_boundary entries must be convertible to workspace type $T"))
    end
    boundary_norm = norm(boundary)
    isfinite(boundary_norm) && !iszero(boundary_norm) || throw(ArgumentError(
        "left_boundary must have a finite, nonzero norm",
    ))
    boundary ./= boundary_norm
    return boundary
end

function _build_initial_factor(
    ::Type{T},
    boundary::Vector{T},
    full_left::SpaceInfo,
    residual_left::ResidualSpaceInfo,
) where {T}
    length(full_left.sectors) == 1 && only(full_left.multiplicities) == 1 ||
        throw(ArgumentError(
            "the left virtual space must have one original sector with reduced " *
            "multiplicity one",
        ))

    embedding = only(residual_left.embeddings)
    charge = residual_left.sectors[embedding.residual_slot]
    factor_space = _residual_space(typeof(charge), (charge => 1,))
    C = TK.TensorMap{T}(undef, residual_left.space, factor_space)
    fill!(C, zero(T))
    block = TK.block(C, charge)
    @inbounds for row in eachindex(boundary)
        block[embedding.rows[row], 1] = boundary[row]
    end
    return C
end

_allocate_scratch(::Type{UniqueStyle}, ::Type{T}, ::Int) where {T} = nothing
_allocate_scratch(::Type{FusionTreeStyle}, ::Type{T}, length::Int) where {T} =
    zeros(T, length)

_route_columns(::Type{MPSMode}, ::Int) = 1
_route_columns(::Type{TracedMPOMode}, residual_block_max::Int) = residual_block_max
_route_columns(::Type{JointMPOMode}, ::Int) = 1

function _allocate_workspace(
    ::Type{M},
    ::Type{S},
    ::Type{T},
    ::Type{Rprob},
    residual_block_max::Int,
    residual_sector_max::Int,
    qmax::Int,
    scratch_max::Int,
) where {M<:AbstractSamplingMode,S,T,Rprob}
    route_output = zeros(
        T,
        residual_block_max,
        _route_columns(M, residual_block_max),
    )
    scratch = _allocate_scratch(S, T, scratch_max)
    return SamplingWorkspace{M,T,Rprob,typeof(scratch)}(
        route_output,
        scratch,
        zeros(Rprob, qmax),
        zeros(Int, residual_sector_max),
    )
end

_zero_like(::Nothing) = nothing
_zero_like(A::AbstractArray{T}) where {T} = zeros(T, size(A))

function _clone_workspace(
    workspace::SamplingWorkspace{M,T,Rprob},
) where {M,T,Rprob}
    scratch = _zero_like(workspace.scratch)
    return SamplingWorkspace{M,T,Rprob,typeof(scratch)}(
        zeros(T, size(workspace.route_output)),
        scratch,
        zeros(Rprob, length(workspace.q)),
        zeros(Int, length(workspace.branch_columns)),
    )
end

function _ensure_workspaces!(sampler::BornSampler, count::Int)
    while length(sampler.workspaces) < count
        push!(sampler.workspaces, _clone_workspace(first(sampler.workspaces)))
    end
    return nothing
end

@inline function _zero_active!(A::AbstractMatrix{T}, rows::Int, columns::Int) where {T}
    @inbounds for column in 1:columns, row in 1:rows
        A[row, column] = zero(T)
    end
    return nothing
end

@inline function _active_norm2(A::AbstractMatrix{T}, rows::Int, columns::Int) where {T}
    value = zero(typeof(real(zero(T))))
    @inbounds for column in 1:columns, row in 1:rows
        value += abs2(A[row, column])
    end
    return value
end

@inline function _copy_active!(
    destination::AbstractMatrix,
    first_column::Int,
    source::AbstractMatrix,
    rows::Int,
    columns::Int,
)
    @inbounds for column in 1:columns, row in 1:rows
        destination[row, first_column + column - 1] = source[row, column]
    end
    return nothing
end

function _apply_route_to_workspace!(
    workspace::SamplingWorkspace,
    C,
    plan,
    route,
    physical,
    purification,
)
    left_sector = plan.residual_left.sectors[route.left_slot]
    Cblock = TK.block(C, left_sector)
    columns = size(Cblock, 2)
    rows = plan.residual_right.dimensions[route.right_slot]
    _zero_active!(workspace.route_output, rows, columns)
    output = view(workspace.route_output, 1:rows, 1:columns)
    _apply_route!(
        output,
        Cblock,
        plan,
        route,
        physical,
        purification,
        workspace.scratch,
    )
    return rows, columns
end

@inline _outcome_count(::Type{M}, plan) where {M<:PhysicalSamplingMode} =
    plan.physical.fulldim
@inline _outcome_count(::Type{JointMPOMode}, plan) =
    plan.physical.fulldim * purification_dimension(plan)
@inline _outcome_count(::SamplingWorkspace{M}, plan) where {M} =
    _outcome_count(M, plan)

@inline function _joint_coordinates(plan, selected::Int)
    dk = purification_dimension(plan)
    x = div(selected - 1, dk) + 1
    y = rem(selected - 1, dk) + 1
    return x, y
end

function _compute_weights!(
    workspace::SamplingWorkspace{M},
    C,
    plan,
) where {M<:PhysicalSamplingMode}
    q = workspace.q
    dp = plan.physical.fulldim
    dk = purification_dimension(plan)

    @inbounds for x in 1:dp
        qx = zero(eltype(q))
        physical = plan.physical_basis[x]
        for y in 1:dk
            purification = plan.purification_basis[y]
            for route in _channel_routes(plan, physical, purification)
                left_sector = plan.residual_left.sectors[route.left_slot]
                TK.hasblock(C, left_sector) || continue
                rows, columns = _apply_route_to_workspace!(
                    workspace,
                    C,
                    plan,
                    route,
                    physical,
                    purification,
                )
                qx += _active_norm2(workspace.route_output, rows, columns)
            end
        end
        q[x] = real(qx)
    end

    return nothing
end

function _compute_weights!(
    workspace::SamplingWorkspace{JointMPOMode},
    C,
    plan,
)
    q = workspace.q
    dk = purification_dimension(plan)

    @inbounds for x in 1:plan.physical.fulldim
        physical = plan.physical_basis[x]
        for y in 1:dk
            qxy = zero(eltype(q))
            purification = plan.purification_basis[y]
            for route in _channel_routes(plan, physical, purification)
                left_sector = plan.residual_left.sectors[route.left_slot]
                TK.hasblock(C, left_sector) || continue
                rows, columns = _apply_route_to_workspace!(
                    workspace,
                    C,
                    plan,
                    route,
                    physical,
                    purification,
                )
                qxy += _active_norm2(workspace.route_output, rows, columns)
            end
            q[(x - 1) * dk + y] = real(qxy)
        end
    end

    return nothing
end

@inline function _total_weight(q, count::Int, site::Int)
    z = zero(eltype(q))
    @inbounds for index in 1:count
        z += q[index]
    end
    isfinite(z) && z > zero(z) || throw(ArgumentError(
        "site $site has zero or non-finite total Born weight",
    ))
    return z
end

@inline function _draw_outcome(rng, q, z, dp::Int)
    threshold = rand(rng, eltype(q)) * z
    cumulative = zero(eltype(q))
    selected = dp
    @inbounds for x in 1:dp
        cumulative += q[x]
        if threshold < cumulative
            selected = x
            break
        end
    end
    return selected
end

function _count_branch_columns!(workspace, C, plan, physical, purifications)
    count = length(plan.residual_right.sectors)
    dimensions = workspace.branch_columns
    @inbounds for slot in 1:count
        dimensions[slot] = 0
    end
    for purification in purifications
        for route in _channel_routes(plan, physical, purification)
            left_sector = plan.residual_left.sectors[route.left_slot]
            TK.hasblock(C, left_sector) || continue
            dimensions[route.right_slot] += size(TK.block(C, left_sector), 2)
        end
    end
    return nothing
end

@inline function _selected_branch(
    ::SamplingWorkspace{M},
    plan,
    selected::Int,
) where {M<:PhysicalSamplingMode}
    return plan.physical_basis[selected], plan.purification_basis
end

@inline function _selected_branch(
    ::SamplingWorkspace{JointMPOMode},
    plan,
    selected::Int,
)
    x, y = _joint_coordinates(plan, selected)
    return plan.physical_basis[x], (plan.purification_basis[y],)
end

function _build_selected_factor_and_weight!(workspace, C, plan, selected::Int)
    physical, purifications = _selected_branch(workspace, plan, selected)
    _count_branch_columns!(workspace, C, plan, physical, purifications)
    positions = workspace.branch_columns
    residual_type = eltype(plan.residual_right.sectors)
    factor_space = _residual_space(
        residual_type,
        (
            plan.residual_right.sectors[slot] => positions[slot]
            for slot in eachindex(plan.residual_right.sectors)
        ),
    )
    G = TK.TensorMap{eltype(C)}(undef, plan.residual_right.space, factor_space)
    fill!(G, zero(eltype(C)))
    qselected = zero(eltype(workspace.q))

    @inbounds for slot in eachindex(plan.residual_right.sectors)
        positions[slot] = 0
    end
    for purification in purifications
        for route in _channel_routes(plan, physical, purification)
            left_sector = plan.residual_left.sectors[route.left_slot]
            TK.hasblock(C, left_sector) || continue
            rows, route_columns = _apply_route_to_workspace!(
                workspace,
                C,
                plan,
                route,
                physical,
                purification,
            )
            qselected += _active_norm2(
                workspace.route_output,
                rows,
                route_columns,
            )
            first_column = positions[route.right_slot] + 1
            right_sector = plan.residual_right.sectors[route.right_slot]
            _copy_active!(
                TK.block(G, right_sector),
                first_column,
                workspace.route_output,
                rows,
                route_columns,
            )
            positions[route.right_slot] += route_columns
        end
    end
    return G, real(qselected)
end

function _build_selected_factor!(workspace, C, plan, selected::Int)
    G, _ = _build_selected_factor_and_weight!(
        workspace,
        C,
        plan,
        selected,
    )
    return G
end

function _compute_weights_and_factors!(
    workspace::SamplingWorkspace{TracedMPOMode},
    C,
    plan,
)
    # Assemble every physical branch while its route contractions are hot.
    # The returned G_x values are deliberately uncompressed: direct sampling
    # compresses only the selected one, and the prefix cache moves each one out
    # only if its child edge is actually created.
    dp = plan.physical.fulldim
    factors = Vector{typeof(C)}(undef, dp)
    @inbounds for selected in 1:dp
        factor, weight = _build_selected_factor_and_weight!(
            workspace,
            C,
            plan,
            selected,
        )
        factors[selected] = factor
        workspace.q[selected] = weight
    end
    return factors
end

function _compress_factor!(G, plan)
    needs_compression = false
    @inbounds for (slot, sector) in pairs(plan.residual_right.sectors)
        width = TK.hasblock(G, sector) ? size(TK.block(G, sector), 2) : 0
        if width > plan.residual_right.dimensions[slot]
            needs_compression = true
            break
        end
    end
    if needs_compression
        L, _ = TK.rightorth!(G; alg=TK.LQpos())
        return L
    end
    return G
end

function _advance_factor!(
    workspace::SamplingWorkspace{MPSMode},
    C,
    plan,
    selected::Int,
    qselected,
)
    Cnext = _build_selected_factor!(workspace, C, plan, selected)
    rmul!(Cnext, inv(sqrt(qselected)))
    return Cnext
end

function _advance_built_traced_factor!(G, plan, qselected)
    Cnext = _compress_factor!(G, plan)
    rmul!(Cnext, inv(sqrt(qselected)))
    return Cnext
end

function _advance_factor!(
    workspace::SamplingWorkspace{JointMPOMode},
    C,
    plan,
    selected::Int,
    qselected,
)
    Cnext = _build_selected_factor!(workspace, C, plan, selected)
    rmul!(Cnext, inv(sqrt(qselected)))
    return Cnext
end

@inline function _configuration_length(
    sampler::BornSampler{M},
) where {M<:PhysicalSamplingMode}
    return length(sampler.plans)
end
@inline _configuration_length(sampler::BornSampler{JointMPOMode}) =
    2 * length(sampler.plans)

@inline function _store_outcome!(
    ::SamplingWorkspace{M},
    configuration::AbstractVector,
    site::Int,
    chain_length::Int,
    plan,
    selected::Int,
) where {M<:PhysicalSamplingMode}
    configuration[site] = selected
    return nothing
end

@inline function _store_outcome!(
    ::SamplingWorkspace{JointMPOMode},
    configuration::AbstractVector,
    site::Int,
    chain_length::Int,
    plan,
    selected::Int,
)
    x, y = _joint_coordinates(plan, selected)
    configuration[site] = x
    configuration[chain_length + site] = y
    return nothing
end

@inline function _store_outcome!(
    ::SamplingWorkspace{M},
    configuration::AbstractMatrix,
    site::Int,
    chain_length::Int,
    shot::Int,
    plan,
    selected::Int,
) where {M<:PhysicalSamplingMode}
    configuration[site, shot] = selected
    return nothing
end

@inline function _store_outcome!(
    ::SamplingWorkspace{JointMPOMode},
    configuration::AbstractMatrix,
    site::Int,
    chain_length::Int,
    shot::Int,
    plan,
    selected::Int,
)
    x, y = _joint_coordinates(plan, selected)
    configuration[site, shot] = x
    configuration[chain_length + site, shot] = y
    return nothing
end

"""
    bornsample!(rng, sampler::BornSampler, config::AbstractVector{Int}) -> Float64

Draw one physical configuration into the caller-owned `config` vector and
return its log probability. For an `MPS` or an `MPO` in the default
traced-purification mode,
`config[i]` is the one-based flat index in site `i`'s TensorKit physical-space
canonical basis. An `MPO` compiled with `purified=false` instead
expects length `2L`, stores all physical indices in `config[1:L]`, all sampled
purification indices in `config[(L + 1):(2L)]`, and returns the joint log
probability.

The method mutates the caller's `config` and the sampler's reusable numerical
workspace. It does not alter the canonicalized state tensors.
"""
function bornsample!(
    rng::Random.AbstractRNG,
    sampler::BornSampler,
    config::AbstractVector{Int},
)
    Base.require_one_based_indexing(config)
    expected_length = _configuration_length(sampler)
    length(config) == expected_length || throw(DimensionMismatch(
        "config has length $(length(config)); expected $expected_length",
    ))

    workspace = first(sampler.workspaces)
    factor = sampler.initial_factor
    log_probability = zero(eltype(workspace.q))
    chain_length = length(sampler.plans)
    @inbounds for site in eachindex(sampler.plans)
        if site == chain_length
            log_probability += _sample_terminal_site!(
                rng,
                workspace,
                factor,
                sampler.plans[site],
                config,
                site,
                chain_length,
            )
            break
        end
        increment, factor = _sample_site!(
            rng,
            workspace,
            factor,
            sampler.plans[site],
            config,
            site,
            chain_length,
        )
        log_probability += increment
    end
    return log_probability
end

function _sample_terminal_site!(
    rng,
    workspace,
    factor,
    plan,
    config,
    site::Int,
    chain_length::Int,
)
    _compute_weights!(workspace, factor, plan)
    outcome_count = _outcome_count(workspace, plan)
    z = _total_weight(workspace.q, outcome_count, site)
    selected = _draw_outcome(
        rng,
        workspace.q,
        z,
        outcome_count,
    )
    qselected = workspace.q[selected]
    _store_outcome!(
        workspace,
        config,
        site,
        chain_length,
        plan,
        selected,
    )
    return log(qselected) - log(z)
end

function _sample_site!(
    rng,
    workspace,
    factor,
    plan,
    config,
    site::Int,
    chain_length::Int,
)
    _compute_weights!(workspace, factor, plan)
    outcome_count = _outcome_count(workspace, plan)
    z = _total_weight(workspace.q, outcome_count, site)
    selected = _draw_outcome(
        rng,
        workspace.q,
        z,
        outcome_count,
    )
    qselected = workspace.q[selected]
    next_factor = _advance_factor!(workspace, factor, plan, selected, qselected)
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

function _sample_site!(
    rng,
    workspace::SamplingWorkspace{TracedMPOMode},
    factor,
    plan,
    config,
    site::Int,
    chain_length::Int,
)
    factors = _compute_weights_and_factors!(workspace, factor, plan)
    outcome_count = _outcome_count(workspace, plan)
    z = _total_weight(workspace.q, outcome_count, site)
    selected = _draw_outcome(
        rng,
        workspace.q,
        z,
        outcome_count,
    )
    qselected = workspace.q[selected]
    next_factor = _advance_built_traced_factor!(
        factors[selected],
        plan,
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

"""
    bornsample!(rng, sampler::BornSampler)

Draw one sample and return its physical flat-index configuration and log
probability. Recover the ordinary probability with
`exp(shot.log_probability)` when it is needed. The canonical-basis metadata is
already compiled in `sampler.plans` and is not duplicated in the result.
"""
function bornsample!(rng::Random.AbstractRNG, sampler::BornSampler)
    configuration = Vector{Int}(undef, _configuration_length(sampler))
    log_probability = bornsample!(rng, sampler, configuration)
    return (
        configuration=configuration,
        log_probability=log_probability,
    )
end

function _copy_active_weights(workspace, count::Int)
    weights = Vector{eltype(workspace.q)}(undef, count)
    @inbounds for index in 1:count
        weights[index] = workspace.q[index]
    end
    return weights
end

@inline _prefix_environment_type(
    ::BornSampler{TracedMPOMode},
    ::Type{F},
) where {F} = TracedBranchBundle{F}

@inline _prefix_environment_type(
    ::BornSampler,
    ::Type{F},
) where {F} = F

function _initialize_prefix_cache!(
    sampler::BornSampler,
    workspace::SamplingWorkspace,
    cache::PrefixCache{T,R,F,F},
) where {T,R,F}
    factor = sampler.initial_factor
    plan = first(sampler.plans)
    _compute_weights!(workspace, factor, plan)
    outcome_count = _outcome_count(workspace, plan)
    _total_weight(workspace.q, outcome_count, 1)
    q = _copy_active_weights(workspace, outcome_count)
    root_id = _allocate_node_id!(cache)
    extendable = length(sampler.plans) > 1
    root = PrefixNode(
        id=root_id,
        log_probability=zero(eltype(q)),
        q=q,
        factor_space=extendable ? TK.space(factor) : nothing,
    )
    _set_node!(cache, root)
    extendable && _insert_resident!(cache, root, factor)
    return root
end

function _initialize_prefix_cache!(
    sampler::BornSampler{TracedMPOMode},
    workspace::SamplingWorkspace{TracedMPOMode},
    cache::PrefixCache{T,R,F,TracedBranchBundle{F}},
) where {T,R,F}
    factor = sampler.initial_factor
    plan = first(sampler.plans)
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
        branch_factor_spaces=extendable ?
            map(TK.space, factors::Vector{F}) : nothing,
    )
    _set_node!(cache, root)
    extendable && _insert_resident!(
        cache,
        root,
        TracedBranchBundle(factors::Vector{F}),
    )
    return root
end

function _build_prefix_child!(
    sampler::BornSampler,
    workspace::SamplingWorkspace,
    current_cache::PrefixCache{T,R,F,F},
    next_cache::PrefixCache{T,R,F,F},
    parent::PrefixNode,
    selected::Int,
    site::Int,
) where {T,R,F}
    plan = sampler.plans[site]
    qselected = parent.q[selected]
    z = _total_weight(parent.q, length(parent.q), site)
    child_log_probability =
        parent.log_probability + (log(qselected) - log(z))
    child_id = _allocate_node_id!(next_cache)

    parent_factor = _prefix_factor(current_cache, parent.id)
    factor = _advance_factor!(
        workspace,
        parent_factor,
        plan,
        selected,
        qselected,
    )
    next_plan = sampler.plans[site + 1]
    _compute_weights!(workspace, factor, next_plan)
    outcome_count = _outcome_count(workspace, next_plan)
    _total_weight(workspace.q, outcome_count, site + 1)
    q = _copy_active_weights(workspace, outcome_count)
    extendable = site + 1 < length(sampler.plans)
    child = PrefixNode(
        id=child_id,
        log_probability=child_log_probability,
        q=q,
        factor_space=extendable ? TK.space(factor) : nothing,
    )
    _set_node!(next_cache, child)
    extendable && _admit_owned_prefix_factor!(next_cache, child, factor)
    return child
end

function _build_prefix_child!(
    sampler::BornSampler{TracedMPOMode},
    workspace::SamplingWorkspace{TracedMPOMode},
    current_cache::PrefixCache{T,R,F,TracedBranchBundle{F}},
    next_cache::PrefixCache{T,R,F,TracedBranchBundle{F}},
    parent::PrefixNode,
    selected::Int,
    site::Int,
) where {T,R,F}
    plan = sampler.plans[site]
    qselected = parent.q[selected]
    z = _total_weight(parent.q, length(parent.q), site)
    child_log_probability =
        parent.log_probability + (log(qselected) - log(z))
    child_id = _allocate_node_id!(next_cache)

    G = _take_traced_branch!(current_cache, parent, selected)
    factor = _advance_built_traced_factor!(G, plan, qselected)
    next_plan = sampler.plans[site + 1]
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
        branch_factor_spaces=extendable ?
            map(TK.space, factors::Vector{F}) : nothing,
    )
    _set_node!(next_cache, child)
    extendable && _admit_owned_prefix_factor!(
        next_cache,
        child,
        TracedBranchBundle(factors::Vector{F}),
    )
    return child
end

function _get_or_build_prefix_child!(
    sampler::BornSampler,
    workspace,
    current_cache::PrefixCache,
    next_cache::PrefixCache,
    parent::PrefixNode,
    selected::Int,
    site::Int,
)
    slot = parent.children[selected]
    child_id = _child_id(slot)
    iszero(child_id) || return _published_prefix_node(next_cache, child_id)

    lock(slot.lock)
    try
        child_id = _child_id(slot)
        if iszero(child_id)
            child = _build_prefix_child!(
                sampler,
                workspace,
                current_cache,
                next_cache,
                parent,
                selected,
                site,
            )
            # The node and its complete resident/disk environment become
            # visible before the release publication of this child id.
            _publish_child_id!(slot, child.id)
            return child
        end
        return _published_prefix_node(next_cache, child_id)
    finally
        unlock(slot.lock)
    end
end

function _draw_cached_outcome!(
    rng,
    sampler::BornSampler,
    workspace,
    current_cache::PrefixCache,
    current_node_ids::Vector{Int},
    configuration::Matrix{Int},
    shot::Int,
    site::Int,
)
    node = _published_prefix_node(current_cache, current_node_ids[shot])
    chain_length = length(sampler.plans)
    z = _total_weight(node.q, length(node.q), site)
    selected = _draw_outcome(rng, node.q, z, length(node.q))
    _store_outcome!(
        workspace,
        configuration,
        site,
        chain_length,
        shot,
        sampler.plans[site],
        selected,
    )
    return node, selected, z
end

function _advance_cached_layer_shot!(
    rng,
    sampler::BornSampler,
    workspace,
    current_cache::PrefixCache,
    next_cache::PrefixCache,
    current_node_ids::Vector{Int},
    next_node_ids::Vector{Int},
    configuration::Matrix{Int},
    shot::Int,
    site::Int,
)
    node, selected, _ = _draw_cached_outcome!(
        rng,
        sampler,
        workspace,
        current_cache,
        current_node_ids,
        configuration,
        shot,
        site,
    )
    child = _get_or_build_prefix_child!(
        sampler,
        workspace,
        current_cache,
        next_cache,
        node,
        selected,
        site,
    )
    @inbounds next_node_ids[shot] = child.id
    return nothing
end

function _finish_cached_layer_shot!(
    rng,
    sampler::BornSampler,
    workspace,
    current_cache::PrefixCache,
    current_node_ids::Vector{Int},
    configuration::Matrix{Int},
    log_probability,
    shot::Int,
    site::Int,
)
    node, selected, z = _draw_cached_outcome!(
        rng,
        sampler,
        workspace,
        current_cache,
        current_node_ids,
        configuration,
        shot,
        site,
    )
    @inbounds log_probability[shot] =
        node.log_probability + (log(node.q[selected]) - log(z))
    return nothing
end

function _advance_cached_layer!(
    shot_rngs,
    sampler::BornSampler,
    current_cache::PrefixCache,
    next_cache::PrefixCache,
    current_node_ids::Vector{Int},
    next_node_ids::Vector{Int},
    configuration::Matrix{Int},
    site::Int,
    nshots::Int,
    worker_count::Int,
)
    next_shot = Threads.Atomic{Int}(1)
    Threads.@sync for worker_id in 1:worker_count
        Threads.@spawn begin
            workspace = sampler.workspaces[worker_id]
            while true
                shot = Threads.atomic_add!(next_shot, 1)
                shot > nshots && break
                _advance_cached_layer_shot!(
                    shot_rngs[shot],
                    sampler,
                    workspace,
                    current_cache,
                    next_cache,
                    current_node_ids,
                    next_node_ids,
                    configuration,
                    shot,
                    site,
                )
            end
        end
    end
    return nothing
end

function _finish_cached_layer!(
    shot_rngs,
    sampler::BornSampler,
    current_cache::PrefixCache,
    current_node_ids::Vector{Int},
    configuration::Matrix{Int},
    log_probability,
    site::Int,
    nshots::Int,
    worker_count::Int,
)
    next_shot = Threads.Atomic{Int}(1)
    Threads.@sync for worker_id in 1:worker_count
        Threads.@spawn begin
            workspace = sampler.workspaces[worker_id]
            while true
                shot = Threads.atomic_add!(next_shot, 1)
                shot > nshots && break
                _finish_cached_layer_shot!(
                    shot_rngs[shot],
                    sampler,
                    workspace,
                    current_cache,
                    current_node_ids,
                    configuration,
                    log_probability,
                    shot,
                    site,
                )
            end
        end
    end
    return nothing
end

"""
    bornsample!(rng, sampler::BornSampler, nshots::Int;
                ntasks=Threads.nthreads(), disk=false, maxsize=ntasks)

Draw a batch of samples using layer-synchronous sampled-prefix frontiers. Within
each site, worker tasks dynamically claim shots and independently advance them.
A layer barrier then releases the preceding frontier before the next site
starts. Every worker owns an independent contraction workspace, while every
shot retains its own RNG across layers. `ntasks` is not capped by the number of
Julia threads; Julia's scheduler multiplexes the requested tasks. The caller
must not invoke another sampling method on the same sampler concurrently.

For an `MPS` or an `MPO` in the default mode, the returned configuration matrix
has shape `(length(state), nshots)`, with one physical configuration per
column. For an `MPO` compiled with `purified=false`, it has shape
`(2 * length(state), nshots)`: the first `L` rows are physical indices and the
final `L` rows are purification indices. When `disk=true`, metadata stays in
memory while larger node environments in each frontier are managed by a strict
probability top-`maxsize` resident cache and raw temporary files. An MPS or
joint-MPO node stores one normalized collapsed factor. A traced-MPO node instead
stores the complete bank of uncompressed next-physical-outcome factors; an
outcome is moved out of that bank and compressed only when its child edge is
first used.
"""
function bornsample!(
    rng::Random.AbstractRNG,
    sampler::BornSampler,
    nshots::Int;
    ntasks::Int=Threads.nthreads(),
    disk::Bool=false,
    maxsize::Int=ntasks,
)
    nshots >= 0 || throw(ArgumentError("nshots must be nonnegative"))
    ntasks > 0 || throw(ArgumentError("ntasks must be positive"))
    disk && maxsize < 1 && throw(ArgumentError(
        "maxsize must be at least one when disk=true",
    ))

    chain_length = length(sampler.plans)
    Rprob = eltype(first(sampler.workspaces).q)
    Factor = typeof(sampler.initial_factor)
    configuration = Matrix{Int}(
        undef,
        _configuration_length(sampler),
        nshots,
    )
    log_probability = Vector{Rprob}(undef, nshots)
    if iszero(nshots)
        return (
            configuration=configuration,
            log_probability=log_probability,
        )
    end

    worker_count = min(nshots, ntasks)
    _ensure_workspaces!(sampler, worker_count)

    # Only this task touches the caller's RNG. Each shot retains a seeded RNG
    # across layers, making the result independent of worker scheduling and the
    # requested task count.
    shot_rngs = Vector{Random.Xoshiro}(undef, nshots)
    @inbounds for shot in 1:nshots
        shot_rngs[shot] = Random.Xoshiro(rand(rng, UInt64))
    end

    Environment = _prefix_environment_type(sampler, Factor)
    current_cache = _new_prefix_cache(
        Environment,
        Factor,
        Rprob,
        1;
        disk=disk && chain_length > 1,
        maxsize=maxsize,
    )
    next_cache = nothing
    try
        root = _initialize_prefix_cache!(
            sampler,
            first(sampler.workspaces),
            current_cache,
        )
        current_node_ids = fill(root.id, nshots)
        next_node_ids = Vector{Int}(undef, nshots)

        for site in 1:(chain_length - 1)
            next_cache = _new_prefix_cache(
                Environment,
                Factor,
                Rprob,
                nshots;
                disk=disk && site + 1 < chain_length,
                maxsize=maxsize,
            )
            _advance_cached_layer!(
                shot_rngs,
                sampler,
                current_cache,
                next_cache,
                current_node_ids,
                next_node_ids,
                configuration,
                site,
                nshots,
                worker_count,
            )

            _cleanup_prefix_cache!(current_cache)
            current_cache = next_cache
            next_cache = nothing
            current_node_ids, next_node_ids = next_node_ids, current_node_ids
        end

        _finish_cached_layer!(
            shot_rngs,
            sampler,
            current_cache,
            current_node_ids,
            configuration,
            log_probability,
            chain_length,
            nshots,
            worker_count,
        )
    finally
        next_cache === nothing || _cleanup_prefix_cache!(next_cache)
        _cleanup_prefix_cache!(current_cache)
    end

    return (
        configuration=configuration,
        log_probability=log_probability,
    )
end

"""
    bornsample!(rng, state::Union{FiniteMPS.MPS,FiniteMPS.MPO};
                left_boundary=nothing, purified=true)

Compile `state` and draw one sample. This convenience method modifies the
input state's canonical gauge. Reuse a `BornSampler` for repeated shots.
"""
function bornsample!(
    rng::Random.AbstractRNG,
    state::Union{FiniteMPS.MPS,FiniteMPS.MPO};
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
    state::Union{FiniteMPS.MPS,FiniteMPS.MPO},
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
