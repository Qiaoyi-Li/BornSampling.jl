# Rank-five local tensors occur when an operator-valued base MPS already has a
# local purification leg and its tangent tensor also carries one persistent
# symmetry leg.  The two legs stay separate throughout tangent sampling:
#
#     left, physical ; local_purification, symmetry, right

leftspace(tensor::FiniteMPS.MPSTensor{5}) = TK.codomain(tensor.A, 1)
physspace(tensor::FiniteMPS.MPSTensor{5}) = TK.codomain(tensor.A, 2)
purspaces(tensor::FiniteMPS.MPSTensor{5}) =
    (TK.domain(tensor.A, 1), TK.domain(tensor.A, 2))
rightspace(tensor::FiniteMPS.MPSTensor{5}) = TK.domain(tensor.A, 3)

function _append_sampling_spaces!(spaces, tensor::FiniteMPS.MPSTensor{5})
    purification, symmetry = purspaces(tensor)
    append!(spaces, (
        leftspace(tensor),
        physspace(tensor),
        purification,
        symmetry,
        rightspace(tensor),
    ))
    return nothing
end

function _compile_site(
    tensor::FiniteMPS.MPSTensor{5},
    ::Type{S},
    kernel_cache,
    residual::ResidualSymmetry,
) where {S}
    left = SpaceInfo(leftspace(tensor))
    physical = SpaceInfo(physspace(tensor))
    purification_spaces = purspaces(tensor)
    purification = (SpaceInfo(first(purification_spaces)),
                      SpaceInfo(last(purification_spaces)))
    right = SpaceInfo(rightspace(tensor))
    residual_left = ResidualSpaceInfo(left, residual)
    residual_right = ResidualSpaceInfo(right, residual)
    physical_basis = _basis_info(physical)
    purification_basis = (_basis_info(first(purification)),
                            _basis_info(last(purification)))
    transitions, transition_groups = _compile_transitions(
        Val(5), S, tensor.A, left, physical, purification, right, kernel_cache,
    )
    routes = _build_channel_routes(
        transitions,
        transition_groups,
        residual_left,
        residual_right,
    )
    return SitePlan{5,S}(
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

_empty_transition_groups(::Val{5}, physical, purification) =
    [Int[] for _ in eachindex(physical.sectors),
               _ in eachindex(first(purification).sectors),
               _ in eachindex(last(purification).sectors)]

function _push_transition_group!(
    ::Val{5},
    groups,
    transition_index,
    pair,
    physical,
    purification,
)
    physical_slot = _sector_slot(physical, pair[1].uncoupled[2])
    first_slot = _sector_slot(first(purification), pair[2].uncoupled[1])
    second_slot = _sector_slot(last(purification), pair[2].uncoupled[2])
    push!(groups[physical_slot, first_slot, second_slot], transition_index)
    return nothing
end

function _compile_transition(
    ::Val{5},
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
    first_purification, second_purification = purification
    left_slot = _sector_slot(left, fout.uncoupled[1])
    physical_slot = _sector_slot(physical, fout.uncoupled[2])
    first_slot = _sector_slot(first_purification, fin.uncoupled[1])
    second_slot = _sector_slot(second_purification, fin.uncoupled[2])
    right_slot = _sector_slot(right, fin.uncoupled[3])
    B = A[fout, fin]
    kernel = _compile_kernel(S, pair, B, kernel_cache)

    expected_reduced = (
        left.multiplicities[left_slot],
        physical.multiplicities[physical_slot],
        first_purification.multiplicities[first_slot],
        second_purification.multiplicities[second_slot],
        right.multiplicities[right_slot],
    )
    size(B) == expected_reduced || throw(DimensionMismatch(
        "rank-5 reduced block has size $(size(B)); expected $expected_reduced",
    ))
    _check_kernel_size(
        S,
        kernel,
        (
            left.irrepdims[left_slot],
            physical.irrepdims[physical_slot],
            first_purification.irrepdims[first_slot],
            second_purification.irrepdims[second_slot],
            right.irrepdims[right_slot],
        ),
    )
    return Transition{5,S}(left_slot, right_slot, B, kernel)
end

function _build_channel_routes(
    transitions,
    transition_groups::AbstractArray{<:Any,3},
    residual_left::ResidualSpaceInfo{VL,I},
    residual_right::ResidualSpaceInfo{VR,I},
) where {VL,VR,I}
    routes = Array{Vector{ChannelRoute},3}(undef, size(transition_groups))
    for physical_slot in axes(routes, 1),
        first_slot in axes(routes, 2),
        second_slot in axes(routes, 3)
        routes[physical_slot, first_slot, second_slot] = _route_group(
            transitions,
            transition_groups[physical_slot, first_slot, second_slot],
            residual_left,
            residual_right,
        )
    end
    return routes
end

@inline function _channel_routes(
    plan::SitePlan{5},
    physical::BasisInfo,
    purification::NTuple{2,BasisInfo},
)
    return plan.routes[
        physical.sector_slot,
        first(purification).sector_slot,
        last(purification).sector_slot,
    ]
end

purification_dimension(plan::SitePlan{5}) =
    first(plan.purification).fulldim * last(plan.purification).fulldim

@inline function _reduced_slice(
    transition::Transition{5},
    degeneracy,
    purification_degeneracy::NTuple{2,Int},
)
    return view(
        transition.B,
        :,
        degeneracy,
        first(purification_degeneracy),
        last(purification_degeneracy),
        :,
    )
end

@inline function _fusion_slice(
    transition::Transition{5},
    irrep,
    purification_irrep::NTuple{2,Int},
)
    return view(
        transition.kernel,
        :,
        irrep,
        first(purification_irrep),
        last(purification_irrep),
        :,
    )
end

function _apply_route!(
    Yroute::AbstractMatrix,
    Cblock::AbstractMatrix,
    plan::SitePlan{5,UniqueStyle},
    route::ChannelRoute,
    physical_basis::BasisInfo,
    purification_basis::NTuple{2,BasisInfo},
    scratch,
)
    beta = physical_basis.degeneracy
    eta = (first(purification_basis).degeneracy,
           last(purification_basis).degeneracy)
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
    plan::SitePlan{5,FusionTreeStyle},
    route::ChannelRoute,
    physical_basis::BasisInfo,
    purification_basis::NTuple{2,BasisInfo},
    scratch,
)
    beta = physical_basis.degeneracy
    eta = (first(purification_basis).degeneracy,
           last(purification_basis).degeneracy)
    mu = physical_basis.irrep
    nu = (first(purification_basis).irrep,
          last(purification_basis).irrep)
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
