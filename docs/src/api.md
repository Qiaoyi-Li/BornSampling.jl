# [Public API](@id public_api)

```@meta
CurrentModule = Bornsampling
```

Bornsampling exports [`BornSampler`](@ref) and [`bornsample!`](@ref).

## Constructing a sampler

```@docs
BornSampler
```

For an `MPS`, construction selects pure-state sampling and the configuration
length is `L = length(state)`.

For an `MPO`, `purified=true` selects the exact physical marginal and also
returns `L` indices. `purified=false` selects the joint distribution and
returns `2L` indices ordered as

```text
x₁, x₂, …, xL, y₁, y₂, …, yL.
```

A rank-3 site inside an MPO contributes the synthetic purification index
`yᵢ = 1`. This gives mixed rank-3/rank-4 MPOs one consistent configuration
layout.

`left_boundary` accepts a pure boundary vector for a supported nontrivial left
boundary. Sampler construction normalizes that vector, canonicalizes the state
at site 1, compiles the local contractions, and allocates reusable worker
workspaces.

## Drawing samples

```@docs
bornsample!
```

The allocation-conscious single-shot form writes into a caller-owned vector:

```julia
config = Vector{Int}(undef, length(state))
logp = Bornsampling.bornsample!(rng, sampler, config)
```

The allocating convenience form returns a named tuple:

```julia
shot = Bornsampling.bornsample!(rng, sampler)
# (configuration = ..., log_probability = ...)
```

For batched sampling:

```julia
batch = Bornsampling.bornsample!(
    rng,
    sampler,
    nshots;
    ntasks=Threads.nthreads(),
    disk=false,
    maxsize=ntasks,
)
```

`batch.configuration` stores one shot per column and
`batch.log_probability[n]` is the log probability of column `n`. `ntasks`
chooses the number of Julia worker tasks. With `disk=true`, `maxsize` chooses
how many prefix environments remain resident in memory.

An ordinary probability is obtained when needed with
`exp(shot.log_probability)` or `exp.(batch.log_probability)`.

## Direct state convenience

An `MPS` or `MPO` may be passed directly:

```julia
shot = Bornsampling.bornsample!(rng, deepcopy(state))
batch = Bornsampling.bornsample!(rng, deepcopy(state), nshots; ntasks=4)
```

These forms construct a temporary sampler and therefore canonicalize the
supplied state. Reusing an explicit `BornSampler` retains its compiled plans,
prefix-ready metadata, and numerical workspaces across calls.

## Configuration basis

Every physical index is a one-based flat index in the corresponding
`Bornsampling.TK` physical-space canonical basis. Within a sector, the irrep
index changes fastest and the degeneracy index changes next. Joint MPO
configurations use the same convention for each purification index.

The exclamation mark records the public mutations: the core form writes
`config`, every sampler call reuses mutable numerical workspaces, and direct
state forms canonicalize their input state.
