# One-frontier prefix metadata and numerical-environment storage.
#
# A batch advances every shot one site at a time. During a layer transition the
# current cache supplies parent environments and a separate next cache receives
# child nodes. After the layer barrier the current cache is discarded. MPS and
# joint-MPO nodes own one normalized collapsed factor, while traced-MPO nodes
# own the complete bank of uncompressed next-physical-branch factors.

"""A concurrently published child edge in the sampled-prefix tree."""
struct ChildSlot
    id::Threads.Atomic{Int}
    lock::ReentrantLock
end

ChildSlot() = ChildSlot(Threads.Atomic{Int}(0), ReentrantLock())

@inline _child_id(slot::ChildSlot) = slot.id[]

function _publish_child_id!(slot::ChildSlot, id::Int)
    slot.id[] = id
    return nothing
end

"""
Metadata for one sampled prefix.

`q` contains the next-site branch weights and is always resident; its sum is
computed at the point of use instead of being stored as duplicate metadata.
The numerical environment itself is deliberately absent. For an extendable
MPS or joint-MPO prefix, `factor_space` reconstructs its normalized collapsed
factor. For an extendable traced-MPO prefix, `branch_factor_spaces` reconstruct
the complete uncompressed `G_x` bank, whose entries are moved out once their
edges are created. Other sampling modes may use a self-describing environment
codec and set `extendable=true` while leaving both space fields as `nothing`.
A prefix immediately before the final site has `extendable=false`, needs no
environment form or child edges, and has an empty `children` vector. After
publication through its parent's `ChildSlot`, every node field is read-only.
"""
struct PrefixNode{R}
    id::Int
    log_probability::R
    q::Vector{R}
    children::Vector{ChildSlot}
    factor_space::Union{Nothing,TK.TensorMapSpace}
    branch_factor_spaces::Union{Nothing,Vector{TK.TensorMapSpace}}
end

function PrefixNode(;
    id::Int,
    log_probability::R,
    q::AbstractVector{R},
    factor_space::Union{Nothing,TK.TensorMapSpace},
    branch_factor_spaces::Union{Nothing,AbstractVector}=nothing,
    extendable::Union{Nothing,Bool}=nothing,
) where {R}
    spaces = branch_factor_spaces === nothing ? nothing :
             TK.TensorMapSpace[space for space in branch_factor_spaces]
    spaces === nothing || length(spaces) == length(q) ||
        throw(DimensionMismatch(
            "branch factor space count $(length(spaces)) does not match " *
            "$(length(q)) branch weights",
        ))
    has_environment = factor_space !== nothing || spaces !== nothing
    should_extend = something(extendable, has_environment)
    !should_extend && has_environment && throw(ArgumentError(
        "a terminal prefix cannot own a continuation environment",
    ))
    children = !should_extend ? ChildSlot[] :
               [ChildSlot() for _ in eachindex(q)]
    return PrefixNode{R}(
        id,
        log_probability,
        Vector{R}(q),
        children,
        factor_space,
        spaces,
    )
end

"""
The complete uncompressed physical-branch bank owned by one traced-MPO node.

Individual entries are moved out exactly once when their child edge is first
created. The per-edge lock serializes callers selecting the same entry, while
different entries can be consumed concurrently.
"""
mutable struct TracedBranchBundle{F}
    factors::Vector{Union{Nothing,F}}
end

function TracedBranchBundle(factors::AbstractVector{F}) where {F}
    owned = Vector{Union{Nothing,F}}(undef, length(factors))
    copyto!(owned, factors)
    return TracedBranchBundle{F}(owned)
end

"""
Storage for one sampled-prefix frontier.

The metadata vector is bounded by the shot count. A cache is written while its
frontier is constructed, then its environments are read or consumed while that
frontier advances; the layer barrier separates those phases. `admission_lock`
serializes construction-time dictionary writes and top-K replacement. Disk I/O
for an evicted victim happens while that lock is held, before the resident
dictionary is updated.
"""
mutable struct PrefixCache{T,R,F,E}
    nodes::Vector{Union{Nothing,PrefixNode{R}}}
    resident::Dict{Int,E}
    maxsize::Int
    directory::Union{Nothing,String}

    next_node_id::Threads.Atomic{Int}
    admission_lock::ReentrantLock

    min_resident_id::Int
end

function _new_prefix_cache(
    ::Type{F},
    ::Type{R},
    capacity::Int;
    disk::Bool,
    maxsize::Int=1024,
) where {T,R,F<:TK.TensorMap{T}}
    return _new_prefix_cache(
        F,
        F,
        R,
        capacity;
        disk=disk,
        maxsize=maxsize,
    )
end

function _new_prefix_cache(
    ::Type{E},
    ::Type{F},
    ::Type{R},
    capacity::Int;
    disk::Bool,
    maxsize::Int=1024,
) where {T,R,F<:TK.TensorMap{T},E}
    nodes = Vector{Union{Nothing,PrefixNode{R}}}(undef, capacity)
    fill!(nodes, nothing)
    resident_limit = disk ? maxsize : capacity
    directory = disk ? mktempdir(; prefix="Bornsampling-prefix-") : nothing

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

function _allocate_node_id!(cache::PrefixCache)
    return Threads.atomic_add!(cache.next_node_id, 1) + 1
end

function _set_node!(cache::PrefixCache{T,R}, node::PrefixNode{R}) where {T,R}
    @inbounds cache.nodes[node.id] = node
    return nothing
end

@inline function _published_prefix_node(
    cache::PrefixCache{T,R},
    id::Int,
) where {T,R}
    # A child id is published only after this slot has been written completely.
    # `ChildSlot.id` uses release/acquire atomics, so traversing an existing edge
    # can read immutable metadata without funneling every shot through one lock.
    # The root is registered before worker tasks are spawned. Internal callers
    # that inspect an unpublished node do so in the registering task itself.
    @inbounds return cache.nodes[id]::PrefixNode{R}
end

@inline function _factor_path(cache::PrefixCache, id::Int)
    directory = cache.directory::String
    return joinpath(directory, "factor_$id.bin")
end

@inline function _branch_factor_path(
    cache::PrefixCache,
    node_id::Int,
    selected::Int,
)
    directory = cache.directory::String
    return joinpath(directory, "branches_$(node_id)_$(selected).bin")
end

function _write_raw_factor_atomic!(path::String, factor)
    temporary = path * ".tmp"
    try
        open(temporary, "w") do io
            write(io, factor.data)
            flush(io)
        end
        mv(temporary, path; force=false)
    catch
        ispath(temporary) && rm(temporary; force=true)
        rethrow()
    end
    return nothing
end

function _write_factor_atomic!(
    cache::PrefixCache{T,R,F,F},
    node::PrefixNode,
    factor::F,
) where {T,R,F}
    path = _factor_path(cache, node.id)
    _write_raw_factor_atomic!(path, factor)
    return nothing
end

function _write_bundle_atomic!(
    cache::PrefixCache{T,R,F,TracedBranchBundle{F}},
    node::PrefixNode,
    bundle::TracedBranchBundle{F},
) where {T,R,F}
    @inbounds for selected in eachindex(bundle.factors)
        factor = bundle.factors[selected]
        factor === nothing && continue
        path = _branch_factor_path(cache, node.id, selected)
        _write_raw_factor_atomic!(path, factor)
    end
    return nothing
end

_write_environment_atomic!(
    cache::PrefixCache{T,R,F,F},
    node::PrefixNode,
    factor::F,
) where {T,R,F} = _write_factor_atomic!(cache, node, factor)

_write_environment_atomic!(
    cache::PrefixCache{T,R,F,TracedBranchBundle{F}},
    node::PrefixNode,
    bundle::TracedBranchBundle{F},
) where {T,R,F} = _write_bundle_atomic!(cache, node, bundle)

function _read_branch_factor(
    cache::PrefixCache{T,R,F,TracedBranchBundle{F}},
    node::PrefixNode,
    selected::Int,
) where {T,R,F}
    spaces = node.branch_factor_spaces::Vector{TK.TensorMapSpace}
    factor = TK.TensorMap{T}(undef, spaces[selected])
    path = _branch_factor_path(cache, node.id, selected)
    open(path, "r") do io
        read!(io, factor.data)
    end
    rm(path; force=true)
    return factor::F
end

function _read_factor(
    cache::PrefixCache{T,R,F,F},
    node::PrefixNode,
) where {T,R,F}
    path = _factor_path(cache, node.id)
    factor_space = node.factor_space::TK.TensorMapSpace
    factor = TK.TensorMap{T}(undef, factor_space)
    open(path, "r") do io
        read!(io, factor.data)
    end
    return factor::F
end

"""
Return a node's immutable factor.

The resident dictionary and node metadata are read only after the construction
barrier. If the lookup misses, the complete atomically-renamed disk file is
already visible.
"""
function _prefix_factor(cache::PrefixCache{T,R,F,F}, id::Int) where {T,R,F}
    node = _published_prefix_node(cache, id)
    factor = get(cache.resident, node.id, nothing)
    factor === nothing || return factor
    return _read_factor(cache, node)
end

"""
Move one uncompressed branch factor out of a traced node's environment bank.

The edge's `ChildSlot.lock` serializes callers for the same `selected` value.
Different entries belong to distinct child edges and can be moved concurrently.
"""
function _take_traced_branch!(
    cache::PrefixCache{T,R,F,TracedBranchBundle{F}},
    node::PrefixNode,
    selected::Int,
) where {T,R,F}
    bundle = get(cache.resident, node.id, nothing)

    if bundle !== nothing
        owned = bundle.factors[selected]
        owned === nothing && error(
            "traced branch $selected of prefix $(node.id) was already consumed",
        )
        bundle.factors[selected] = nothing
        return owned::F
    end

    return _read_branch_factor(cache, node, selected)
end

function _recompute_resident_minimum!(cache::PrefixCache{T,R}) where {T,R}
    minimum_id = 0
    minimum_log_probability = convert(R, Inf)
    for id in keys(cache.resident)
        node = @inbounds cache.nodes[id]::PrefixNode{R}
        if node.log_probability < minimum_log_probability
            minimum_id = id
            minimum_log_probability = node.log_probability
        end
    end
    cache.min_resident_id = minimum_id
    return nothing
end

function _insert_resident!(
    cache::PrefixCache{T,R,F,E},
    node::PrefixNode,
    environment::E,
) where {T,R,F,E}
    cache.resident[node.id] = environment
    cache.directory === nothing || _recompute_resident_minimum!(cache)
    return nothing
end

"""
Store the numerical environment for a newly registered child node. Ownership
is transferred to the cache, so the caller must not mutate `owned` afterwards.
For MPS and joint MPO nodes this is one normalized factor; for traced MPO nodes
it is the complete uncompressed physical-branch bank.

The caller publishes the child id only after this function returns. Because a
destination cache contains one depth only, strict admission computes the exact
top-`maxsize` set within that frontier, independent of the parent frontier.
"""
function _admit_owned_prefix_factor!(
    cache::PrefixCache{T,R,F,E},
    node::PrefixNode{R},
    owned::E,
) where {T,R,F,E}
    if cache.directory === nothing
        lock(cache.admission_lock)
        try
            _insert_resident!(cache, node, owned)
            return nothing
        finally
            unlock(cache.admission_lock)
        end
    end

    write_new_to_disk = false
    lock(cache.admission_lock)
    try
        if length(cache.resident) < cache.maxsize
            _insert_resident!(cache, node, owned)
            return nothing
        else
            minimum = _published_prefix_node(
                cache,
                cache.min_resident_id,
            ).log_probability
            if !(node.log_probability > minimum)
                # Admission is deliberately strict. Exact ties stay cold.
                write_new_to_disk = true
            end
        end

        if !write_new_to_disk
            victim_id = cache.min_resident_id
            victim = _published_prefix_node(cache, victim_id)
            victim_environment = cache.resident[victim_id]

            # Publish the complete cold representation before replacing the
            # resident dictionary entry.
            _write_environment_atomic!(
                cache,
                victim,
                victim_environment,
            )

            cache.resident[node.id] = owned
            delete!(cache.resident, victim_id)
            _recompute_resident_minimum!(cache)
            return nothing
        end
    finally
        unlock(cache.admission_lock)
    end

    _write_environment_atomic!(cache, node, owned)
    return nothing
end

"""Release one frontier's metadata, environments, and temporary directory."""
function _cleanup_prefix_cache!(cache::PrefixCache)
    empty!(cache.resident)
    fill!(cache.nodes, nothing)
    directory = cache.directory
    directory === nothing || (ispath(directory) && rm(directory; recursive=true, force=true))
    return nothing
end
