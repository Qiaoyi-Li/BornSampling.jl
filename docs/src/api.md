# [Public API](@id public_api)

```@meta
CurrentModule = BornSampling
```

BornSampling exports [`BornSampler`](@ref) and [`bornsample!`](@ref).

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

For a `FiniteMPSTangents.TangentMPS`, the sampler represents the coherent sum
of all one-site insertion terms under the tangent-space/Hilbert-space
isomorphism. With `purified=true`, local MPO purification legs and the
persistent global tangent-symmetry leg are traced, and the configuration
contains `L` physical indices. With `purified=false`, all existing local
purification legs and the one global leg are sampled. The layout is

```text
x₁, …, xL [, y₁, …, yL] [, q].
```

The `y` group is present when any base tensor is rank four; rank-three sites in
such a mixed base contribute `yᵢ = 1`. The final `q` entry is present only when
the tangent tensors carry the extra persistent leg. The tangent and its base
point are not mutated or canonicalized. The compiled sampler retains views
into their tensor blocks, so they must not be modified while it is in use.

`left_boundary` accepts a pure boundary vector for a supported nontrivial left
boundary. Sampler construction normalizes that vector, compiles the local
contractions, and allocates reusable worker workspaces. Direct MPS/MPO inputs
are additionally canonicalized at site 1.

## Drawing samples

```@docs
bornsample!
```

The allocation-conscious single-shot form writes into a caller-owned vector:

```julia
config = Vector{Int}(undef, length(state))
logp = BornSampling.bornsample!(rng, sampler, config)
```

The allocating convenience form returns a named tuple:

```julia
shot = BornSampling.bornsample!(rng, sampler)
# (configuration = ..., log_probability = ...)
```

For batched sampling:

```julia
batch = BornSampling.bornsample!(
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
chooses the number of Julia worker tasks. Shots advance one site at a time and
are dynamically assigned within each layer. With `disk=true`, `maxsize`
chooses how many environments remain resident in each prefix frontier; the
adjacent current and next frontiers coexist while a layer is being completed.
For a tangent sampler, `disk=true` also gives its once-per-batch right suffix
sweep a separate consume-once disk store. The first active completion stays in
memory; each later suffix environment is loaded once, shared read-only by all
workers at a layer, and deleted after the layer barrier. This suffix policy is
controlled only by `disk`; `maxsize` still applies solely to prefix frontiers.

An ordinary probability is obtained when needed with
`exp(shot.log_probability)` or `exp.(batch.log_probability)`.

## Direct state convenience

An `MPS`, `MPO`, or `TangentMPS` may be passed directly:

```julia
shot = BornSampling.bornsample!(rng, deepcopy(state))
batch = BornSampling.bornsample!(rng, deepcopy(state), nshots; ntasks=4)
```

These forms construct a temporary sampler. Direct MPS/MPO forms canonicalize
the supplied state, whereas the TangentMPS form leaves the tangent and its base
unchanged. Reusing an explicit `BornSampler` retains its compiled plans,
prefix-ready metadata, and numerical workspaces across calls.

## Configuration basis

Every physical index is a one-based flat index in the corresponding
`BornSampling.TK` physical-space canonical basis. Within a sector, the irrep
index changes fastest and the degeneracy index changes next. Joint MPO
configurations use the same convention for each purification index.

The exclamation mark records the public mutations: the core form writes
`config`, every sampler call reuses mutable numerical workspaces, and direct
MPS/MPO forms canonicalize their input state. Direct TangentMPS forms do not
canonicalize or mutate the tangent or its base.
