# Batch-owned, consume-once storage for tangent completion environments.
#
# The outer sampling task is the sole reader and writer. A separate lock or
# cache policy would therefore add machinery without protecting a real
# concurrent access path.

"""Storage for one batch's pending site completions."""
mutable struct TangentCompletionStore{C}
    values::Vector{Union{Nothing,C}}
    directory::Union{Nothing,String}
end

function TangentCompletionStore{C}(
    length::Integer;
    disk::Bool,
) where {C}
    values = Vector{Union{Nothing,C}}(undef, Int(length))
    fill!(values, nothing)
    directory = disk ?
                mktempdir(; prefix="Bornsampling-tangent-completion-") :
                nothing
    return TangentCompletionStore{C}(values, directory)
end

@inline function _completion_path(
    store::TangentCompletionStore,
    site::Int,
)
    return joinpath(store.directory::String, "completion_$site.bin")
end

"""Transfer ownership of `completion` into the store for `site`."""
function _put_completion!(
    store::TangentCompletionStore{C},
    site::Integer,
    completion::C,
) where {C}
    index = Int(site)
    if store.directory === nothing
        @inbounds store.values[index] = completion
    else
        open(_completion_path(store, index), "w") do io
            serialize(io, completion)
        end
    end
    return nothing
end

"""Move the completion for `site` out of the store and release its backing."""
function _take_completion!(
    store::TangentCompletionStore{C},
    site::Integer,
) where {C}
    index = Int(site)
    if store.directory === nothing
        completion = @inbounds store.values[index]
        @inbounds store.values[index] = nothing
        return completion::C
    end

    path = _completion_path(store, index)
    completion = open(deserialize, path)::C
    rm(path)
    return completion
end

"""Release every pending completion and the store's private directory."""
function _cleanup_completion_store!(store::TangentCompletionStore)
    fill!(store.values, nothing)
    directory = store.directory
    if directory !== nothing
        ispath(directory) && rm(directory; recursive=true)
        store.directory = nothing
    end
    return nothing
end
