# Batch-local prefix-tree metadata and numerical-environment storage.
#
# Every `PrefixNode` stays in memory for the lifetime of one batch. Its larger
# numerical environment is managed separately by `PrefixCache`: MPS and joint
# MPO nodes own one normalized collapsed factor, while traced MPO nodes own the
# complete bank of uncompressed next-physical-branch factors. High-probability
# node environments remain resident; cold environments are written as raw
# TensorMap storage vectors when disk storage is enabled. Nodes immediately
# before the final site retain only the already-computed branch weights needed
# to finish a shot.

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
edges are created. A prefix immediately before the final site needs neither
environment form nor child edges, so both space fields are `nothing` and
`children` is empty. After publication through its parent's `ChildSlot`, the
node fields except the atomic `logical_resident` flag are read-only.
"""
struct PrefixNode{R}
    id::Int
    log_probability::R
    q::Vector{R}
    children::Vector{ChildSlot}
    logical_resident::Threads.Atomic{Bool}
    factor_space::Union{Nothing,TK.TensorMapSpace}
    branch_factor_spaces::Union{Nothing,Vector{TK.TensorMapSpace}}
end

function PrefixNode(;
    id::Int,
    log_probability::R,
    q::AbstractVector{R},
    factor_space::Union{Nothing,TK.TensorMapSpace},
    branch_factor_spaces::Union{Nothing,AbstractVector}=nothing,
) where {R}
    spaces = branch_factor_spaces === nothing ? nothing :
             TK.TensorMapSpace[space for space in branch_factor_spaces]
    spaces === nothing || length(spaces) == length(q) ||
        throw(DimensionMismatch(
            "branch factor space count $(length(spaces)) does not match " *
            "$(length(q)) branch weights",
        ))
    extendable = factor_space !== nothing || spaces !== nothing
    children = !extendable ? ChildSlot[] :
               [ChildSlot() for _ in eachindex(q)]
    return PrefixNode{R}(
        id,
        log_probability,
        Vector{R}(q),
        children,
        Threads.Atomic{Bool}(false),
        factor_space,
        spaces,
    )
end

"""
The complete uncompressed physical-branch bank owned by one traced-MPO node.

Individual entries are moved out exactly once, when their child edge is first
created. The lock also prevents an eviction writer from serializing a factor
while that factor is being removed from the resident bank.
"""
mutable struct TracedBranchBundle{F}
    factors::Vector{Union{Nothing,F}}
    lock::ReentrantLock
end

function TracedBranchBundle(factors::AbstractVector{F}) where {F}
    owned = Vector{Union{Nothing,F}}(undef, length(factors))
    copyto!(owned, factors)
    return TracedBranchBundle{F}(owned, ReentrantLock())
end

"""
Storage owned by one batch invocation of `bornsample!`.

The metadata vector has the exact structural upper bound
`1 + nshots * max(L - 1, 0)`.
`resident_lock` protects only short dictionary accesses. All changes to the
resident top-K set are serialized by `admission_lock`; disk I/O for an evicted
victim happens while that writer lock is held but while the victim is still
reachable from `resident`. A traced bank has its own lock so eviction cannot
serialize a `G_x` while an edge is moving that same entry out.
"""
mutable struct PrefixCache{T,R,F<:TK.TensorMap{T},E}
    nodes::Vector{Union{Nothing,PrefixNode{R}}}
    resident::Dict{Int,E}
    maxsize::Int
    directory::Union{Nothing,String}

    next_node_id::Threads.Atomic{Int}
    resident_lock::ReentrantLock
    admission_lock::ReentrantLock

    min_resident_id::Int
end

function _new_prefix_cache(
    ::Type{F},
    ::Type{R},
    nshots::Int,
    L::Int;
    disk::Bool,
    maxsize::Int=1024,
) where {T,R,F<:TK.TensorMap{T}}
    return _new_prefix_cache(
        F,
        F,
        R,
        nshots,
        L;
        disk=disk,
        maxsize=maxsize,
    )
end

function _new_prefix_cache(
    ::Type{E},
    ::Type{F},
    ::Type{R},
    nshots::Int,
    L::Int;
    disk::Bool,
    maxsize::Int=1024,
) where {T,R,F<:TK.TensorMap{T},E}
    # Complete configurations are outputs, not reusable prefixes, so the tree
    # contains only depths 0:(L - 1).
    prefix_depths = max(L - 1, 0)
    capacity = Base.checked_add(1, Base.checked_mul(nshots, prefix_depths))
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
    lock(bundle.lock)
    try
        @inbounds for selected in eachindex(bundle.factors)
            factor = bundle.factors[selected]
            factor === nothing && continue
            path = _branch_factor_path(cache, node.id, selected)
            _write_raw_factor_atomic!(path, factor)
        end
    finally
        unlock(bundle.lock)
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

A resident lookup holds `resident_lock` only for the dictionary access.  A
reader that obtains a resident TensorMap keeps a strong reference even if that
node is evicted immediately afterwards.  If the lookup misses, eviction order
guarantees that the complete, atomically renamed disk file is already visible.
"""
function _prefix_factor(cache::PrefixCache{T,R,F,F}, id::Int) where {T,R,F}
    node = _published_prefix_node(cache, id)
    factor = nothing
    lock(cache.resident_lock)
    try
        factor = get(cache.resident, node.id, nothing)
    finally
        unlock(cache.resident_lock)
    end
    factor === nothing || return factor
    return _read_factor(cache, node)
end

"""
Move one uncompressed branch factor out of a traced node's environment bank.

The edge's `ChildSlot.lock` serializes callers for the same `selected` value;
the bundle lock additionally coordinates distinct edges with top-K eviction.
"""
function _take_traced_branch!(
    cache::PrefixCache{T,R,F,TracedBranchBundle{F}},
    node::PrefixNode,
    selected::Int,
) where {T,R,F}
    bundle = nothing
    lock(cache.resident_lock)
    try
        bundle = get(cache.resident, node.id, nothing)
    finally
        unlock(cache.resident_lock)
    end

    if bundle !== nothing
        owned = nothing
        lock(bundle.lock)
        try
            owned = bundle.factors[selected]
            owned === nothing && error(
                "traced branch $selected of prefix $(node.id) was already consumed",
            )
            bundle.factors[selected] = nothing

            # An eviction may have serialized this branch after the resident
            # lookup but before the bundle lock was acquired. The in-memory
            # copy wins; remove that now-redundant cold copy if it exists.
            if cache.directory !== nothing
                path = _branch_factor_path(cache, node.id, selected)
                ispath(path) && rm(path; force=true)
            end
        finally
            unlock(bundle.lock)
        end
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
    node.logical_resident[] = true
    cache.directory === nothing || _recompute_resident_minimum!(cache)
    return nothing
end

"""
Store the numerical environment for a newly registered child node. Ownership
is transferred to the cache, so the caller must not mutate `owned` afterwards.
For MPS and joint MPO nodes this is one normalized factor; for traced MPO nodes
it is the complete uncompressed physical-branch bank.

The caller must publish the child id only after this function returns. A child
of a logically cold parent bypasses top-K admission entirely: the resident
threshold can only increase once the cache is full, while a child cannot be
more probable than its parent.
"""
function _admit_owned_prefix_factor!(
    cache::PrefixCache{T,R,F,E},
    node::PrefixNode{R},
    parent_id::Int,
    owned::E,
) where {T,R,F,E}
    parent = _published_prefix_node(cache, parent_id)

    if cache.directory === nothing
        lock(cache.resident_lock)
        try
            _insert_resident!(cache, node, owned)
            return nothing
        finally
            unlock(cache.resident_lock)
        end
    end

    # Once false for a published parent, this flag can never become true again.
    # This fast path therefore needs neither the writer lock nor a top-K scan.
    if !parent.logical_resident[]
        _write_environment_atomic!(cache, node, owned)
        return nothing
    end

    write_new_to_disk = false
    lock(cache.admission_lock)
    try
        # The parent may have been evicted while this task waited for admission.
        if !parent.logical_resident[]
            write_new_to_disk = true
        else
            lock(cache.resident_lock)
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
            finally
                unlock(cache.resident_lock)
            end

            if !write_new_to_disk
                victim_id = cache.min_resident_id
                victim = _published_prefix_node(cache, victim_id)
                victim_environment = nothing
                lock(cache.resident_lock)
                try
                    victim_environment = cache.resident[victim_id]
                finally
                    unlock(cache.resident_lock)
                end

                # Keep the victim resident and readable until its complete
                # factor or branch-bank disk representation has been published.
                _write_environment_atomic!(
                    cache,
                    victim,
                    victim_environment,
                )

                lock(cache.resident_lock)
                try
                    cache.resident[node.id] = owned
                    delete!(cache.resident, victim_id)
                    victim.logical_resident[] = false
                    node.logical_resident[] = true
                    _recompute_resident_minimum!(cache)
                finally
                    unlock(cache.resident_lock)
                end
                return nothing
            end
        end
    finally
        unlock(cache.admission_lock)
    end

    _write_environment_atomic!(cache, node, owned)
    return nothing
end

"""Release all resident environments and remove the batch's temporary directory."""
function _cleanup_prefix_cache!(cache::PrefixCache)
    empty!(cache.resident)
    directory = cache.directory
    directory === nothing || (ispath(directory) && rm(directory; recursive=true, force=true))
    return nothing
end
