# Symmetry-aware local channel contractions.

"""Contraction tag for TensorKit sector types with unique fusion."""
struct UniqueStyle end

"""Contraction tag for the fusion-tree kernels needed by non-unique fusion."""
struct FusionTreeStyle end

"""
    SpaceInfo(space)

Compile the sector dimensions of one elementary TensorKit space. Flat-basis
indices follow TensorKit's array order, with the irrep coordinate varying faster
than the degeneracy coordinate inside every sector.
"""
struct SpaceInfo{V,I}
    space::V
    sectors::Vector{I}
    multiplicities::Vector{Int}
    irrepdims::Vector{Int}
    fulldim::Int
end

function SpaceInfo(space)
    sector_list = collect(TK.sectors(space))
    multiplicities = Vector{Int}(undef, length(sector_list))
    irrepdims = Vector{Int}(undef, length(sector_list))

    for (slot, sector) in pairs(sector_list)
        multiplicities[slot] = _integer_dimension(TK.dim(space, sector))
        irrepdims[slot] = _integer_dimension(TK.dim(sector))
    end

    return SpaceInfo(
        space,
        sector_list,
        multiplicities,
        irrepdims,
        _integer_dimension(TK.dim(space)),
    )
end

@inline _sector_slot(space::SpaceInfo, sector) =
    findfirst(isequal(sector), space.sectors)::Int

function _integer_dimension(dimension)
    try
        return Int(dimension)
    catch
        throw(ArgumentError("only spaces with integer dimensions are supported; got $dimension"))
    end
end

# ---------------------------------------------------------------------------
# Residual symmetry
# ---------------------------------------------------------------------------

abstract type AbstractResidualProjector end

struct TrivialResidualProjector <: AbstractResidualProjector end
struct AtomicResidualProjector <: AbstractResidualProjector end

struct ProductResidualProjector{K} <: AbstractResidualProjector
    kept_components::K
end

"""
    ResidualSymmetry

The symmetry left unbroken after fixing local physical/purification basis
vectors. The residual sector type is the type parameter `I`; product-sector
component positions live only in the projector that consumes them.
"""
struct ResidualSymmetry{I,P<:AbstractResidualProjector}
    projector::P
end

ResidualSymmetry(::Type{I}, projector::P) where {I,P<:AbstractResidualProjector} =
    ResidualSymmetry{I,P}(projector)

@inline _project_sector(::TrivialResidualProjector, sector) = TK.Trivial()
@inline _project_sector(::AtomicResidualProjector, sector) = sector

@inline function _project_sector(projector::ProductResidualProjector, sector)
    kept = projector.kept_components
    components = ntuple(i -> sector[kept[i]], length(kept))
    return length(components) == 1 ? only(components) : TK.ProductSector(components)
end

@inline function _project_sector(residual::ResidualSymmetry{I}, sector) where {I}
    return convert(I, _project_sector(residual.projector, sector))
end

function _actual_sectors(::Type{I}, spaces) where {I}
    sectors = I[]
    for space in spaces
        append!(sectors, TK.sectors(space))
    end
    isempty(sectors) && throw(ArgumentError(
        "cannot infer a residual symmetry from empty spaces",
    ))
    return sectors
end

function _component_is_residual(
    ::Type{I},
    sectors,
    component::Int,
) where {I}
    TK.FusionStyle(I) isa TK.UniqueFusion || return false
    return all(sectors) do sector
        _integer_dimension(TK.dim(sector[component])) == 1
    end
end

function _atomic_is_residual(::Type{I}, sectors) where {I}
    TK.FusionStyle(I) isa TK.UniqueFusion || return false
    return all(sector -> _integer_dimension(TK.dim(sector)) == 1, sectors)
end

function _infer_residual_from_sectors(::Type{I}, sectors) where {I}
    if _atomic_is_residual(I, sectors)
        return ResidualSymmetry(I, AtomicResidualProjector())
    end
    return ResidualSymmetry(TK.Trivial, TrivialResidualProjector())
end

function _infer_residual_from_sectors(
    ::Type{I},
    sectors,
) where {Components,I<:TK.ProductSector{Components}}
    component_types = fieldtypes(Components)
    kept = Tuple(
        component for component in eachindex(component_types)
        if _component_is_residual(component_types[component], sectors, component)
    )
    if isempty(kept)
        return ResidualSymmetry(TK.Trivial, TrivialResidualProjector())
    end

    projector = ProductResidualProjector(kept)
    residual_type = typeof(_project_sector(projector, first(sectors)))
    return ResidualSymmetry(residual_type, projector)
end

"""
    _infer_residual_symmetry(I, spaces)

Infer the unbroken symmetry structurally from the original sector type and all
spaces participating in sampler compilation. Atomic sectors are retained only
when they use `UniqueFusion` and every actual irrep is one-dimensional. Product
sectors apply that same test independently to every component.
"""
function _infer_residual_symmetry(::Type{I}, spaces) where {I}
    return _infer_residual_from_sectors(I, _actual_sectors(I, spaces))
end

"""Construct a canonical residual elementary space from actual-sector dimensions."""
function _residual_space(::Type{I}, dims) where {I}
    dimensions = Dict{I,Int}()
    for (sector, dimension) in dims
        value = Int(dimension)
        iszero(value) || (dimensions[convert(I, sector)] = value)
    end
    return TK.Vect[I](dimensions)
end

"""Embedding of one original sector's complete basis into one residual block."""
struct SectorEmbedding
    residual_slot::Int
    rows::UnitRange{Int}
end

"""
    ResidualSpaceInfo(full, residual)

Compile the projection of an original `SpaceInfo` into a residual TensorKit
space. `embeddings[cslot].rows` is local to the residual block, not a range in
the full dense basis.
"""
struct ResidualSpaceInfo{V,I}
    space::V
    sectors::Vector{I}
    dimensions::Vector{Int}
    embeddings::Vector{SectorEmbedding}
end

function ResidualSpaceInfo(full::SpaceInfo, residual::ResidualSymmetry{I}) where {I}
    sectors = I[]
    dimensions = Int[]
    sector_to_slot = Dict{I,Int}()
    embeddings = Vector{SectorEmbedding}(undef, length(full.sectors))

    for original_slot in eachindex(full.sectors)
        residual_sector = _project_sector(residual, full.sectors[original_slot])
        residual_slot = get(sector_to_slot, residual_sector, 0)
        if iszero(residual_slot)
            push!(sectors, residual_sector)
            push!(dimensions, 0)
            residual_slot = length(sectors)
            sector_to_slot[residual_sector] = residual_slot
        end

        width = full.multiplicities[original_slot] * full.irrepdims[original_slot]
        first_row = dimensions[residual_slot] + 1
        last_row = dimensions[residual_slot] + width
        embeddings[original_slot] = SectorEmbedding(
            residual_slot,
            first_row:last_row,
        )
        dimensions[residual_slot] = last_row
    end

    dims = Dict(sectors[slot] => dimensions[slot] for slot in eachindex(sectors))
    space = _residual_space(I, dims)
    return ResidualSpaceInfo(
        space,
        sectors,
        dimensions,
        embeddings,
    )
end

"""Metadata for one vector in a TensorKit elementary-space flat basis."""
struct BasisInfo
    sector_slot::Int
    degeneracy::Int
    irrep::Int
end

function _basis_info(space::SpaceInfo)
    basis = Vector{BasisInfo}(undef, space.fulldim)
    for slot in eachindex(space.sectors)
        first_index = first(TK.axes(space.space, space.sectors[slot]))
        multiplicity = space.multiplicities[slot]
        irrepdim = space.irrepdims[slot]
        for degeneracy in 1:multiplicity, irrep in 1:irrepdim
            flat = first_index - 1 + irrep + (degeneracy - 1) * irrepdim
            basis[flat] = BasisInfo(slot, degeneracy, irrep)
        end
    end
    return basis
end

# Rank-three tensors use a synthetic, one-dimensional purification basis. It
# deliberately has no TensorKit space attached to it.
const _TRIVIAL_PURIFICATION_BASIS = (BasisInfo(1, 1, 1),)

"""
One allowed sector transition and its cached reduced-data and symmetry views.

`B` is always the view returned by `A[fout, fin]`. `kernel` is `nothing` for
`UniqueStyle` and the cached full fusion-tree array for `FusionTreeStyle`.
"""
struct Transition{R,S,B,K}
    left_slot::Int
    right_slot::Int
    B::B
    kernel::K
end

"""
One compiled full-to-residual channel route.

All transitions in a route share fixed physical/purification basis sectors and
the same residual input/output sectors. Their amplitudes must be accumulated
coherently before a branch norm is taken.
"""
struct ChannelRoute
    left_slot::Int
    right_slot::Int
    transition_indices::Vector{Int}
end

function Transition{R,S}(
    left_slot::Int,
    right_slot::Int,
    B,
    kernel,
) where {R,S}
    return Transition{R,S,typeof(B),typeof(kernel)}(
        left_slot,
        right_slot,
        B,
        kernel,
    )
end

abstract type AbstractSitePlan end

"""A fully compiled local contraction plan for an `MPSTensor{R}`."""
struct SitePlan{R,S,L,P,K,VR,RL,RR,KB,TR,Routes} <: AbstractSitePlan
    left::L
    physical::P
    purification::K
    right::VR
    residual_left::RL
    residual_right::RR
    physical_basis::Vector{BasisInfo}
    purification_basis::KB
    transitions::Vector{TR}
    routes::Routes
end

function SitePlan{R,S}(
    left,
    physical,
    purification,
    right,
    residual_left,
    residual_right,
    physical_basis::Vector{BasisInfo},
    purification_basis,
    transitions::Vector{TR},
    routes,
) where {R,S,TR}
    return SitePlan{
        R,S,typeof(left),typeof(physical),typeof(purification),typeof(right),
        typeof(residual_left),typeof(residual_right),typeof(purification_basis),
        TR,typeof(routes),
    }(
        left,
        physical,
        purification,
        right,
        residual_left,
        residual_right,
        physical_basis,
        purification_basis,
        transitions,
        routes,
    )
end

# FiniteMPS' current standard local-leg convention.
leftspace(tensor::FiniteMPS.MPSTensor{3}) = TK.codomain(tensor.A, 1)
physspace(tensor::FiniteMPS.MPSTensor{3}) = TK.codomain(tensor.A, 2)
rightspace(tensor::FiniteMPS.MPSTensor{3}) = TK.domain(tensor.A, 1)

leftspace(tensor::FiniteMPS.MPSTensor{4}) = TK.codomain(tensor.A, 1)
physspace(tensor::FiniteMPS.MPSTensor{4}) = TK.codomain(tensor.A, 2)
purspace(tensor::FiniteMPS.MPSTensor{4}) = TK.domain(tensor.A, 1)
rightspace(tensor::FiniteMPS.MPSTensor{4}) = TK.domain(tensor.A, 2)

function _sampling_spaces(state)
    spaces = Any[]
    for site in eachindex(state)
        _append_sampling_spaces!(spaces, state[site])
    end
    return spaces
end

function _append_sampling_spaces!(spaces, tensor::FiniteMPS.MPSTensor{3})
    append!(spaces, (leftspace(tensor), physspace(tensor), rightspace(tensor)))
    return nothing
end

function _append_sampling_spaces!(spaces, tensor::FiniteMPS.MPSTensor{4})
    append!(spaces, (
        leftspace(tensor),
        physspace(tensor),
        purspace(tensor),
        rightspace(tensor),
    ))
    return nothing
end

"""
    _style_type(tensor)

Choose the contraction path once while compiling a sampler. Concrete sector
types are intentionally not inspected here; TensorKit owns product-symmetry
dispatch through `FusionStyle`.
"""
function _style_type(tensor::FiniteMPS.MPSTensor)
    style = TK.FusionStyle(TK.sectortype(tensor.A))
    if style isa TK.GenericFusion
        throw(ArgumentError("GenericFusion sector types are not supported"))
    elseif style isa TK.UniqueFusion
        return UniqueStyle
    else
        return FusionTreeStyle
    end
end

function _compile_site(
    tensor::FiniteMPS.MPSTensor{3},
    ::Type{S},
    kernel_cache,
    residual::ResidualSymmetry,
) where {S}
    left = SpaceInfo(leftspace(tensor))
    physical = SpaceInfo(physspace(tensor))
    right = SpaceInfo(rightspace(tensor))
    residual_left = ResidualSpaceInfo(left, residual)
    residual_right = ResidualSpaceInfo(right, residual)
    physical_basis = _basis_info(physical)
    purification_basis = _TRIVIAL_PURIFICATION_BASIS
    transitions, transition_groups = _compile_transitions(
        Val(3), S, tensor.A, left, physical, nothing, right, kernel_cache,
    )
    routes = _build_channel_routes(
        transitions,
        transition_groups,
        residual_left,
        residual_right,
    )
    return SitePlan{3,S}(
        left,
        physical,
        nothing,
        right,
        residual_left,
        residual_right,
        physical_basis,
        purification_basis,
        transitions,
        routes,
    )
end

function _compile_site(
    tensor::FiniteMPS.MPSTensor{4},
    ::Type{S},
    kernel_cache,
    residual::ResidualSymmetry,
) where {S}
    left = SpaceInfo(leftspace(tensor))
    physical = SpaceInfo(physspace(tensor))
    purification = SpaceInfo(purspace(tensor))
    right = SpaceInfo(rightspace(tensor))
    residual_left = ResidualSpaceInfo(left, residual)
    residual_right = ResidualSpaceInfo(right, residual)
    physical_basis = _basis_info(physical)
    purification_basis = _basis_info(purification)
    transitions, transition_groups = _compile_transitions(
        Val(4), S, tensor.A, left, physical, purification, right, kernel_cache,
    )
    routes = _build_channel_routes(
        transitions,
        transition_groups,
        residual_left,
        residual_right,
    )
    return SitePlan{4,S}(
        left,
        physical,
        purification,
        right,
        residual_left,
        residual_right,
        physical_basis,
        purification_basis,
        transitions,
        routes,
    )
end

function _compile_transitions(
    rank::Val{R},
    ::Type{S},
    A,
    left,
    physical,
    purification,
    right,
    kernel_cache,
) where {R,S}
    tree_pairs = TK.fusiontrees(A)
    isempty(tree_pairs) && throw(ArgumentError("local tensor has no allowed fusion-tree blocks"))

    first_transition = _compile_transition(
        rank,
        S,
        A,
        first(tree_pairs),
        left,
        physical,
        purification,
        right,
        kernel_cache,
    )
    transitions = Vector{typeof(first_transition)}(undef, length(tree_pairs))
    transitions[1] = first_transition
    transition_groups = _empty_transition_groups(rank, physical, purification)
    _push_transition_group!(
        rank,
        transition_groups,
        1,
        first(tree_pairs),
        physical,
        purification,
    )
    for index in 2:length(tree_pairs)
        transition = _compile_transition(
            rank,
            S,
            A,
            tree_pairs[index],
            left,
            physical,
            purification,
            right,
            kernel_cache,
        )
        typeof(transition) === eltype(transitions) || throw(ArgumentError(
            "fusion-tree blocks do not have a uniform reduced/kernel scalar type",
        ))
        transitions[index] = transition
        _push_transition_group!(
            rank,
            transition_groups,
            index,
            tree_pairs[index],
            physical,
            purification,
        )
    end
    return transitions, transition_groups
end

_empty_transition_groups(::Val{3}, physical, ::Nothing) =
    [Int[] for _ in eachindex(physical.sectors)]

_empty_transition_groups(::Val{4}, physical, purification) =
    [Int[] for _ in eachindex(physical.sectors), _ in eachindex(purification.sectors)]

function _push_transition_group!(
    ::Val{3},
    groups,
    transition_index,
    pair,
    physical,
    ::Nothing,
)
    physical_slot = _sector_slot(physical, pair[1].uncoupled[2])
    push!(groups[physical_slot], transition_index)
    return nothing
end

function _push_transition_group!(
    ::Val{4},
    groups,
    transition_index,
    pair,
    physical,
    purification,
)
    physical_slot = _sector_slot(physical, pair[1].uncoupled[2])
    purification_slot = _sector_slot(purification, pair[2].uncoupled[1])
    push!(groups[physical_slot, purification_slot], transition_index)
    return nothing
end

function _compile_transition(
    ::Val{3},
    ::Type{S},
    A,
    pair,
    left,
    physical,
    ::Nothing,
    right,
    kernel_cache,
) where {S}
    fout, fin = pair
    left_slot = _sector_slot(left, fout.uncoupled[1])
    physical_slot = _sector_slot(physical, fout.uncoupled[2])
    right_slot = _sector_slot(right, fin.uncoupled[1])
    B = A[fout, fin]
    kernel = _compile_kernel(S, pair, B, kernel_cache)

    expected_reduced = (
        left.multiplicities[left_slot],
        physical.multiplicities[physical_slot],
        right.multiplicities[right_slot],
    )
    size(B) == expected_reduced || throw(DimensionMismatch(
        "rank-3 reduced block has size $(size(B)); expected $expected_reduced",
    ))
    _check_kernel_size(
        S,
        kernel,
        (
            left.irrepdims[left_slot],
            physical.irrepdims[physical_slot],
            right.irrepdims[right_slot],
        ),
    )
    return Transition{3,S}(
        left_slot,
        right_slot,
        B,
        kernel,
    )
end

function _compile_transition(
    ::Val{4},
    ::Type{S},
    A,
    pair,
    left,
    physical,
    purification,
    right,
    kernel_cache,
) where {S}
    fout, fin = pair
    left_slot = _sector_slot(left, fout.uncoupled[1])
    physical_slot = _sector_slot(physical, fout.uncoupled[2])
    purification_slot = _sector_slot(purification, fin.uncoupled[1])
    right_slot = _sector_slot(right, fin.uncoupled[2])
    B = A[fout, fin]
    kernel = _compile_kernel(S, pair, B, kernel_cache)

    expected_reduced = (
        left.multiplicities[left_slot],
        physical.multiplicities[physical_slot],
        purification.multiplicities[purification_slot],
        right.multiplicities[right_slot],
    )
    size(B) == expected_reduced || throw(DimensionMismatch(
        "rank-4 reduced block has size $(size(B)); expected $expected_reduced",
    ))
    _check_kernel_size(
        S,
        kernel,
        (
            left.irrepdims[left_slot],
            physical.irrepdims[physical_slot],
            purification.irrepdims[purification_slot],
            right.irrepdims[right_slot],
        ),
    )
    return Transition{4,S}(
        left_slot,
        right_slot,
        B,
        kernel,
    )
end

# In the supported ordinary finite-dimensional UniqueFusion representations,
# every irrep carrier is one-dimensional and the canonical fusion scalar is one.
# TensorKit 0.14 does not define `convert(Array, pair)` for its NoSym `Trivial`
# fusion trees, so this path must not request an explicit fusion tensor.
_kernel_cache(::Type{UniqueStyle}) = nothing
_kernel_cache(::Type{FusionTreeStyle}) = Dict{Any,Any}()

_compile_kernel(::Type{UniqueStyle}, pair, B, ::Nothing) = nothing

function _compile_kernel(::Type{FusionTreeStyle}, pair, B, kernel_cache::Dict)
    return get!(kernel_cache, pair) do
        convert(Array, pair)
    end
end

function _check_kernel_size(::Type{UniqueStyle}, kernel, expected::Tuple)
    all(==(1), expected) || throw(ArgumentError(
        "UniqueStyle requires one-dimensional irreps; got dimensions $expected",
    ))
    return nothing
end

function _check_kernel_size(::Type{FusionTreeStyle}, kernel, expected::Tuple)
    size(kernel) == expected || throw(DimensionMismatch(
        "fusion-tree kernel has size $(size(kernel)); expected $expected",
    ))
    return nothing
end

@inline _group_indices(groups::AbstractVector, physical_slot, purification_slot) =
    groups[physical_slot]
@inline _group_indices(groups::AbstractMatrix, physical_slot, purification_slot) =
    groups[physical_slot, purification_slot]

function _route_group(
    transitions,
    transition_indices,
    residual_left::ResidualSpaceInfo{VL,I},
    residual_right::ResidualSpaceInfo{VR,I},
) where {VL,VR,I}
    routes = ChannelRoute[]
    key_to_route = Dict{Tuple{Int,Int},Int}()

    for transition_index in transition_indices
        transition = transitions[transition_index]
        left_embedding = residual_left.embeddings[transition.left_slot]
        right_embedding = residual_right.embeddings[transition.right_slot]
        key = (left_embedding.residual_slot, right_embedding.residual_slot)
        route_slot = get(key_to_route, key, 0)
        if iszero(route_slot)
            push!(routes, ChannelRoute(
                key[1],
                key[2],
                Int[transition_index],
            ))
            key_to_route[key] = length(routes)
        else
            push!(routes[route_slot].transition_indices, transition_index)
        end
    end
    return routes
end

function _build_channel_routes(
    transitions,
    transition_groups,
    residual_left::ResidualSpaceInfo{VL,I},
    residual_right::ResidualSpaceInfo{VR,I},
) where {VL,VR,I}
    routes = Matrix{Vector{ChannelRoute}}(
        undef,
        size(transition_groups, 1),
        size(transition_groups, 2),
    )
    for physical_slot in axes(routes, 1), purification_slot in axes(routes, 2)
        transition_indices = _group_indices(
            transition_groups,
            physical_slot,
            purification_slot,
        )
        routes[physical_slot, purification_slot] = _route_group(
            transitions,
            transition_indices,
            residual_left,
            residual_right,
        )
    end
    return routes
end

@inline _channel_routes(plan::SitePlan, physical::BasisInfo, purification::BasisInfo) =
    plan.routes[physical.sector_slot, purification.sector_slot]

purification_dimension(::SitePlan{3}) = 1
purification_dimension(plan::SitePlan{4}) = plan.purification.fulldim

kernel_scalartype(plan::SitePlan{R,UniqueStyle}) where {R} =
    eltype(first(plan.transitions).B)
kernel_scalartype(plan::SitePlan{R,FusionTreeStyle}) where {R} =
    eltype(first(plan.transitions).kernel)

scratch_length(::SitePlan{R,UniqueStyle}) where {R} = 0
function scratch_length(plan::SitePlan{R,FusionTreeStyle}) where {R}
    return maximum(plan.transitions; init=0) do transition
        dleft = plan.left.irrepdims[transition.left_slot]
        nright = plan.right.multiplicities[transition.right_slot]
        dleft * nright
    end
end

@inline function _reduced_slice(transition::Transition{3}, degeneracy, purification_degeneracy)
    return view(transition.B, :, degeneracy, :)
end

@inline function _reduced_slice(transition::Transition{4}, degeneracy, purification_degeneracy)
    return view(transition.B, :, degeneracy, purification_degeneracy, :)
end

@inline function _fusion_slice(transition::Transition{3}, irrep, purification_irrep)
    return view(transition.kernel, :, irrep, :)
end

@inline function _fusion_slice(transition::Transition{4}, irrep, purification_irrep)
    return view(transition.kernel, :, irrep, purification_irrep, :)
end

"""
    _apply_route!(Yroute, Cblock, plan, route, physical_basis,
                  purification_basis, scratch)

Compute one coherent residual route. `Cblock` and `Yroute` are block views of
rank-two residual-symmetry TensorMaps for exactly one `(x, y, QL, QR)` route.
The caller clears `Yroute`; this helper adds every original fusion-tree
transition coherently before the caller takes a norm. Different routes must
remain separate.
"""
function _apply_route!(
    Yroute::AbstractMatrix,
    Cblock::AbstractMatrix,
    plan::SitePlan{R,UniqueStyle},
    route::ChannelRoute,
    physical_basis::BasisInfo,
    purification_basis::BasisInfo,
    scratch,
) where {R}
    beta = physical_basis.degeneracy
    eta = purification_basis.degeneracy
    beta_accumulate = one(eltype(Yroute))

    @inbounds for transition_index in route.transition_indices
        transition = plan.transitions[transition_index]
        Bslice = _reduced_slice(transition, beta, eta)
        left_rows = plan.residual_left.embeddings[transition.left_slot].rows
        right_rows = plan.residual_right.embeddings[transition.right_slot].rows
        Cchunk = view(Cblock, left_rows, :)
        Ychunk = view(Yroute, right_rows, :)
        mul!(
            Ychunk,
            transpose(Bslice),
            Cchunk,
            one(eltype(Yroute)),
            beta_accumulate,
        )
    end
    return nothing
end

function _apply_route!(
    Yroute::AbstractMatrix,
    Cblock::AbstractMatrix,
    plan::SitePlan{R,FusionTreeStyle},
    route::ChannelRoute,
    physical_basis::BasisInfo,
    purification_basis::BasisInfo,
    scratch,
) where {R}
    beta = physical_basis.degeneracy
    eta = purification_basis.degeneracy
    mu = physical_basis.irrep
    nu = purification_basis.irrep
    alpha = one(eltype(Yroute))
    factor_columns = size(Cblock, 2)

    @inbounds for transition_index in route.transition_indices
        transition = plan.transitions[transition_index]
        Bslice = _reduced_slice(transition, beta, eta)
        Fslice = _fusion_slice(transition, mu, nu)
        left_slot = transition.left_slot
        right_slot = transition.right_slot
        dleft = plan.left.irrepdims[left_slot]
        nleft = plan.left.multiplicities[left_slot]
        dright = plan.right.irrepdims[right_slot]
        nright = plan.right.multiplicities[right_slot]
        left_rows = plan.residual_left.embeddings[left_slot].rows
        right_rows = plan.residual_right.embeddings[right_slot].rows
        X = reshape(view(scratch, 1:(dleft * nright)), dleft, nright)

        for factor_column in 1:factor_columns
            Ch = reshape(view(Cblock, left_rows, factor_column), dleft, nleft)
            Yh = reshape(view(Yroute, right_rows, factor_column), dright, nright)
            mul!(X, Ch, Bslice)
            mul!(Yh, transpose(Fslice), X, alpha, alpha)
        end
    end
    return nothing
end
