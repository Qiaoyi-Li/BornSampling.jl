# [Public API](@id public_api)

```@meta
CurrentModule = BornSampling
```

BornSampling exports [`BornSampler`](@ref) and [`bornsample!`](@ref).

## Constructing a sampler

```@docs
BornSampler
```

For an `MPS`, `purified=true` traces the left boundary and returns
`L = length(state)` physical indices. With `purified=false`, the boundary index
is sampled jointly, returning `[x₁, …, xL, b]` of length `L + 1`.

For an `MPO`, `purified=true` traces the left boundary and local purification
legs, returning `L` physical indices. With `purified=false`, physical,
purification, and boundary indices are sampled jointly, returning `2L + 1`
entries ordered as:

```text
x₁, x₂, …, xL, y₁, y₂, …, yL, b.
```

At rank-three sites lacking a purification leg, `yᵢ` is set to `0`.

For a `FiniteMPSTangents.TangentMPS`, the sampler represents the coherent sum
of one-site insertion terms. With `purified=true`, boundary, purification, and
global tangent legs are traced, returning `L` physical indices. With
`purified=false`, all indices are sampled jointly with layout:

```text
x₁, …, xL [, y₁, …, yL], b [, q].
```

The `y` group is included when the base state contains rank-four tensors (with
`yᵢ = 0` at rank-three sites). The `q` entry is included when the tangent
carries a global symmetry leg. Constructing a tangent sampler retains views into
the underlying tensors without modifying them.

Constructing a `BornSampler` compiles local contractions, allocates reusable
workspaces, and canonicalizes MPS/MPO inputs at site 1.

## Drawing samples

```@docs
bornsample!
```

The in-place single-shot form writes into a caller-provided vector:

```julia
config = Vector{Int}(undef, length(state))
logp = BornSampling.bornsample!(rng, sampler, config)
```

For joint sampling, size `config` according to the joint configuration layout.

The allocating single-shot form returns a named tuple:

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

`batch.configuration` stores one shot per column, and `batch.log_probability[n]`
is the log probability of column `n`. `ntasks` sets the number of concurrent
Julia worker tasks. With `disk=true`, prefix environments are offloaded to
disk, keeping at most `maxsize` environments in memory per layer. For tangent
sampling, `disk=true` also buffers intermediate right-suffix environments to
disk.

Probabilities can be recovered with `exp.(batch.log_probability)`.

## Direct state convenience

An `MPS`, `MPO`, or `TangentMPS` may be passed directly:

```julia
shot = BornSampling.bornsample!(rng, deepcopy(state))
batch = BornSampling.bornsample!(rng, deepcopy(state), nshots; ntasks=4)
```

These convenience methods construct a temporary sampler. Reusing an explicit
`BornSampler` avoids recompilation and workspace reallocation across repeated
calls.

## Configuration basis

Sampled physical indices are 1-based flat indices in the `TensorKit` canonical
basis, with irrep indices changing fastest, followed by degeneracy indices.
Purification, boundary (`b`), and global tangent (`q`) indices follow the same
canonical ordering. A value of `0` indicates an absent purification leg at a
rank-three site.
