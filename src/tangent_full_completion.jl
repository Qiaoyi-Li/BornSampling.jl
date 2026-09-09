"""
Suffix Gram tensors before any local basis is selected.

Every field is a TensorMap over the original symmetry. In array notation the
fields use ket indices first: `inserted[b; a]`, `cross[b; q, a]`, and, for joint
sampling, `uninserted[b, qbra; qket, a]`. Traced sampling contracts the two q
legs of `uninserted`; without a global q leg all three fields are rank two.
The residual sampling matrices use the transposed virtual-index order.
"""
struct SymmetricTangentCompletion{I,K,N}
    inserted::I
    cross::K
    uninserted::N
end

function _right_symmetric_tangent_completion(
    ::Type{T},
    state::FiniteMPSTangents.TangentMPS,
    ::Type{M},
) where {T,M<:TangentSamplingMode}
    Al = last(state.base.Al)
    B = last(state.B)
    virtual = rightspace(Al)
    TK.dim(virtual) == 1 || throw(ArgumentError(
        "the final right virtual space must be one-dimensional",
    ))
    inserted = TK.id(T, virtual)
    if _tensor_rank(B) == _tensor_rank(Al)
        cross = zeros(T, virtual, virtual)
        uninserted = zeros(T, virtual, virtual)
    else
        symmetry = _tensor_rank(B) == 4 ? purspace(B) : last(purspaces(B))
        cross = zeros(T, virtual, TK.:⊗(symmetry, virtual))
        uninserted = M === TracedTangentMode ?
            zeros(T, virtual, virtual) :
            zeros(T, TK.:⊗(virtual, symmetry), TK.:⊗(symmetry, virtual))
    end
    return SymmetricTangentCompletion(inserted, cross, uninserted)
end

function _retreat_symmetric_tangent_completion(
    metric::SymmetricTangentCompletion,
    Al::FiniteMPS.MPSTensor{3},
    Ar::FiniteMPS.MPSTensor{3},
    B::FiniteMPS.MPSTensor{3},
    ::Type{M},
) where {M<:TangentSamplingMode}
    L, R, insertion = Al.A, Ar.A, B.A
    I, K, N = metric.inserted, metric.cross, metric.uninserted
    TK.@tensor next_i[b; a] := R[b, p, d] * I[d, c] * conj(R[a, p, c])
    TK.@tensor next_k[b; a] := insertion[b, p, d] * I[d, c] * conj(R[a, p, c])
    TK.@tensor next_k[b; a] += L[b, p, d] * K[d, c] * conj(R[a, p, c])
    TK.@tensor next_n[b; a] := L[b, p, d] * N[d, c] * conj(L[a, p, c])
    TK.@tensor next_n[b; a] += insertion[b, p, d] * I[d, c] * conj(insertion[a, p, c])
    TK.@tensor cross_n[b; a] := L[b, p, d] * K[d, c] * conj(insertion[a, p, c])
    next_n += cross_n + adjoint(cross_n)
    return SymmetricTangentCompletion(next_i, next_k, next_n)
end

function _retreat_symmetric_tangent_completion(
    metric::SymmetricTangentCompletion,
    Al::FiniteMPS.MPSTensor{4},
    Ar::FiniteMPS.MPSTensor{4},
    B::FiniteMPS.MPSTensor{4},
    ::Type{M},
) where {M<:TangentSamplingMode}
    L, R, insertion = Al.A, Ar.A, B.A
    I, K, N = metric.inserted, metric.cross, metric.uninserted
    TK.@tensor next_i[b; a] := R[b, p, y, d] * I[d, c] * conj(R[a, p, y, c])
    TK.@tensor next_k[b; a] := insertion[b, p, y, d] * I[d, c] * conj(R[a, p, y, c])
    TK.@tensor next_k[b; a] += L[b, p, y, d] * K[d, c] * conj(R[a, p, y, c])
    TK.@tensor next_n[b; a] := L[b, p, y, d] * N[d, c] * conj(L[a, p, y, c])
    TK.@tensor next_n[b; a] += insertion[b, p, y, d] * I[d, c] * conj(insertion[a, p, y, c])
    TK.@tensor cross_n[b; a] := L[b, p, y, d] * K[d, c] * conj(insertion[a, p, y, c])
    next_n += cross_n + adjoint(cross_n)
    return SymmetricTangentCompletion(next_i, next_k, next_n)
end

function _retreat_symmetric_tangent_completion(
    metric::SymmetricTangentCompletion,
    Al::FiniteMPS.MPSTensor{3},
    Ar::FiniteMPS.MPSTensor{3},
    B::FiniteMPS.MPSTensor{4},
    ::Type{M},
) where {M<:TangentSamplingMode}
    L, R, insertion = Al.A, Ar.A, B.A
    I, K, N = metric.inserted, metric.cross, metric.uninserted
    TK.@tensor next_i[b; a] := R[b, p, d] * I[d, c] * conj(R[a, p, c])
    TK.@tensor next_k[b; q a] := insertion[b, p, q, d] * I[d, c] * conj(R[a, p, c])
    TK.@tensor next_k[b; q a] += L[b, p, d] * K[d, q, c] * conj(R[a, p, c])
    if M === TracedTangentMode
        TK.@tensor next_n[b; a] := L[b, p, d] * N[d, c] * conj(L[a, p, c])
        TK.@tensor next_n[b; a] += insertion[b, p, q, d] * I[d, c] * conj(insertion[a, p, q, c])
        TK.@tensor cross_n[b; a] := L[b, p, d] * K[d, q, c] * conj(insertion[a, p, q, c])
        next_n += cross_n + adjoint(cross_n)
    else
        TK.@tensor next_n[b r; q a] := L[b, p, d] * N[d, r, q, c] * conj(L[a, p, c])
        TK.@tensor next_n[b r; q a] += insertion[b, p, q, d] * I[d, c] * conj(insertion[a, p, r, c])
        TK.@tensor next_n[b r; q a] += L[b, p, d] * K[d, q, c] * conj(insertion[a, p, r, c])
        TK.@tensor next_n[b r; q a] += insertion[b, p, q, d] * conj(K[c, r, d]) * conj(L[a, p, c])
    end
    return SymmetricTangentCompletion(next_i, next_k, next_n)
end

function _retreat_symmetric_tangent_completion(
    metric::SymmetricTangentCompletion,
    Al::FiniteMPS.MPSTensor{4},
    Ar::FiniteMPS.MPSTensor{4},
    B::FiniteMPS.MPSTensor{5},
    ::Type{M},
) where {M<:TangentSamplingMode}
    L, R, insertion = Al.A, Ar.A, B.A
    I, K, N = metric.inserted, metric.cross, metric.uninserted
    TK.@tensor next_i[b; a] := R[b, p, y, d] * I[d, c] * conj(R[a, p, y, c])
    TK.@tensor next_k[b; q a] := insertion[b, p, y, q, d] * I[d, c] * conj(R[a, p, y, c])
    TK.@tensor next_k[b; q a] += L[b, p, y, d] * K[d, q, c] * conj(R[a, p, y, c])
    if M === TracedTangentMode
        TK.@tensor next_n[b; a] := L[b, p, y, d] * N[d, c] * conj(L[a, p, y, c])
        TK.@tensor next_n[b; a] += insertion[b, p, y, q, d] * I[d, c] * conj(insertion[a, p, y, q, c])
        TK.@tensor cross_n[b; a] := L[b, p, y, d] * K[d, q, c] * conj(insertion[a, p, y, q, c])
        next_n += cross_n + adjoint(cross_n)
    else
        TK.@tensor next_n[b r; q a] := L[b, p, y, d] * N[d, r, q, c] * conj(L[a, p, y, c])
        TK.@tensor next_n[b r; q a] += insertion[b, p, y, q, d] * I[d, c] * conj(insertion[a, p, y, r, c])
        TK.@tensor next_n[b r; q a] += L[b, p, y, d] * K[d, q, c] * conj(insertion[a, p, y, r, c])
        TK.@tensor next_n[b r; q a] += insertion[b, p, y, q, d] * conj(K[c, r, d]) * conj(L[a, p, y, c])
    end
    return SymmetricTangentCompletion(next_i, next_k, next_n)
end

# Expand one fusion-tree block at a time directly into residual-sector pairs.
# In particular the joint N tensor is never expanded into a D² Q² dense array:
# only its equal-q components are accumulated for subsequent sampling.
@inline _completion_kernel(::Type{UniqueStyle}, pair) = nothing
@inline _completion_kernel(::Type{FusionTreeStyle}, pair) = convert(Array, pair)
@inline _completion_kernel_value(::Nothing, indices...) = 1
@inline _completion_kernel_value(kernel, indices...) = @inbounds kernel[indices...]

@inline _completion_joint_kernel(::Type{UniqueStyle}, pair) = nothing

function _completion_joint_kernel(::Type{FusionTreeStyle}, pair)
    # TensorKit expands a pair by contracting these trees' coupled indices.
    # Fix the same q coordinate in both trees before that contraction, so even
    # the irrep-only temporary has one q dimension instead of two.
    outgoing = convert(Array, first(pair))
    incoming = convert(Array, last(pair))
    T = promote_type(eltype(outgoing), eltype(incoming))
    kernel = Array{T}(undef, size(outgoing, 1), size(outgoing, 2), size(incoming, 2))
    for q in axes(kernel, 2)
        mul!(
            view(kernel, :, q, :),
            view(outgoing, :, q, :),
            adjoint(view(incoming, q, :, :)),
        )
    end
    return kernel
end

function _residual_completion_scalar(tensor, full, residual, ::Type{S}) where {S}
    T = TK.scalartype(tensor)
    result = TangentBlockMatrix{T}()
    for pair in TK.fusiontrees(tensor)
        reduced = tensor[pair...]
        all(iszero, reduced) && continue
        ket_slot = _sector_slot(full, pair[1].uncoupled[1])
        bra_slot = _sector_slot(full, pair[2].uncoupled[1])
        ket, bra = residual.embeddings[ket_slot], residual.embeddings[bra_slot]
        destination = _tangent_destination_block!(
            result, bra.residual_slot, ket.residual_slot,
            residual.dimensions[bra.residual_slot], residual.dimensions[ket.residual_slot],
        )
        kernel = _completion_kernel(S, pair)
        dk, db = full.irrepdims[ket_slot], full.irrepdims[bra_slot]
        for mk in axes(reduced, 1), mb in axes(reduced, 2), ik in 1:dk, ib in 1:db
            row = first(bra.rows) - 1 + (mb - 1) * db + ib
            column = first(ket.rows) - 1 + (mk - 1) * dk + ik
            @inbounds destination[row, column] += reduced[mk, mb] *
                _completion_kernel_value(kernel, ik, ib)
        end
    end
    return result
end

function _residual_completion_cross(tensor, full, residual, symmetry_basis, ::Type{S}) where {S}
    T = TK.scalartype(tensor)
    result = [TangentBlockMatrix{T}() for _ in symmetry_basis]
    symmetry = SpaceInfo(TK.domain(tensor, 1))
    for pair in TK.fusiontrees(tensor)
        reduced = tensor[pair...]
        all(iszero, reduced) && continue
        ket_slot = _sector_slot(full, pair[1].uncoupled[1])
        bra_slot = _sector_slot(full, pair[2].uncoupled[2])
        q_slot = _sector_slot(symmetry, pair[2].uncoupled[1])
        ket, bra = residual.embeddings[ket_slot], residual.embeddings[bra_slot]
        kernel = _completion_kernel(S, pair)
        dk, db = full.irrepdims[ket_slot], full.irrepdims[bra_slot]
        for (q, basis) in pairs(symmetry_basis)
            basis.sector_slot == q_slot || continue
            destination = _tangent_destination_block!(
                result[q], bra.residual_slot, ket.residual_slot,
                residual.dimensions[bra.residual_slot], residual.dimensions[ket.residual_slot],
            )
            for mk in axes(reduced, 1), mb in axes(reduced, 3), ik in 1:dk, ib in 1:db
                row = first(bra.rows) - 1 + (mb - 1) * db + ib
                column = first(ket.rows) - 1 + (mk - 1) * dk + ik
                @inbounds destination[row, column] += reduced[mk, basis.degeneracy, mb] *
                    _completion_kernel_value(kernel, ik, basis.irrep, ib)
            end
        end
    end
    return result
end

function _residual_completion_joint(tensor, full, residual, symmetry_basis, ::Type{S}) where {S}
    T = TK.scalartype(tensor)
    result = [TangentBlockMatrix{T}() for _ in symmetry_basis]
    symmetry = SpaceInfo(TK.domain(tensor, 1))
    for pair in TK.fusiontrees(tensor)
        # A fixed q in bra and ket belongs to the same original sector.
        pair[1].uncoupled[2] == pair[2].uncoupled[1] || continue
        reduced = tensor[pair...]
        all(iszero, reduced) && continue
        ket_slot = _sector_slot(full, pair[1].uncoupled[1])
        bra_slot = _sector_slot(full, pair[2].uncoupled[2])
        q_slot = _sector_slot(symmetry, pair[2].uncoupled[1])
        ket, bra = residual.embeddings[ket_slot], residual.embeddings[bra_slot]
        kernel = _completion_joint_kernel(S, pair)
        dk, db = full.irrepdims[ket_slot], full.irrepdims[bra_slot]
        for (q, basis) in pairs(symmetry_basis)
            basis.sector_slot == q_slot || continue
            destination = _tangent_destination_block!(
                result[q], bra.residual_slot, ket.residual_slot,
                residual.dimensions[bra.residual_slot], residual.dimensions[ket.residual_slot],
            )
            for mk in axes(reduced, 1), mb in axes(reduced, 4), ik in 1:dk, ib in 1:db
                row = first(bra.rows) - 1 + (mb - 1) * db + ib
                column = first(ket.rows) - 1 + (mk - 1) * dk + ik
                @inbounds destination[row, column] +=
                    reduced[mk, basis.degeneracy, basis.degeneracy, mb] *
                    _completion_kernel_value(kernel, ik, basis.irrep, ib)
            end
        end
    end
    return result
end

"""Convert an already-computed original-symmetry suffix to sampling blocks."""
function _residual_tangent_completion(
    metric::SymmetricTangentCompletion,
    step::TangentLocalPlan,
    ::Type{M};
    side::Symbol=:right,
) where {M<:TangentSamplingMode}
    full = side === :right ? step.left.right : step.left.left
    residual = side === :right ? step.left.residual_right : step.left.residual_left
    S = TK.FusionStyle(TK.sectortype(metric.inserted)) isa TK.UniqueFusion ?
        UniqueStyle : FusionTreeStyle
    inserted = _residual_completion_scalar(metric.inserted, full, residual, S)
    has_symmetry = TK.numind(metric.cross) == 3
    cross = has_symmetry ?
        _residual_completion_cross(metric.cross, full, residual, step.symmetry_basis, S) :
        [_residual_completion_scalar(metric.cross, full, residual, S)]
    uninserted = has_symmetry && M === JointTangentMode ?
        _residual_completion_joint(metric.uninserted, full, residual, step.symmetry_basis, S) :
        [_residual_completion_scalar(metric.uninserted, full, residual, S)]
    return TangentCompletionMetric(
        _tangent_residual_ranges(residual.dimensions), inserted, cross, uninserted,
    )
end
