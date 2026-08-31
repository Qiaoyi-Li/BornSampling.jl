# Bornsampling

[![Stable documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://Qiaoyi-Li.github.io/Bornsampling.jl/stable/)
[![Development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://Qiaoyi-Li.github.io/Bornsampling.jl/dev/)

This repository is a downstream package of
[`FiniteMPS.jl`](https://github.com/Qiaoyi-Li/FiniteMPS.jl) for
drawing canonical-basis snapshots of MPS and MPO according to their Born probabilities. It outputs the sampled configurations and their log
probabilities for subsequent statistical analysis.

## Installation

From the Julia package prompt:

```julia-repl
pkg> add Bornsampling
```

## Quick start

```julia
import Bornsampling
using Bornsampling.FiniteMPS
using Bornsampling.Random

rng = MersenneTwister(1234)
state_for_sampling = deepcopy(state)
sampler = Bornsampling.BornSampler(state_for_sampling)

config = Vector{Int}(undef, length(state_for_sampling))
logp = Bornsampling.bornsample!(rng, sampler, config)

batch = Bornsampling.bornsample!(
    rng,
    sampler,
    1000;
    ntasks=Threads.nthreads(),
)
```

Constructing a sampler canonicalizes its state in place and retains compiled
views of the local tensors. Reuse that sampler for repeated shots and treat its
state as sampling-owned afterward.

`FiniteMPSTangents.TangentMPS` uses the same `BornSampler` and `bornsample!`
entry points. Its coherent single-insertion sum is sampled as a Hilbert-space
state. With `purified=true`, local MPO purification legs and a persistent
global tangent-symmetry leg are traced. With `purified=false`, the local legs
are sampled and the global leg is sampled once; joint output is ordered
`[x; y; q]`, with absent groups omitted. Constructing this sampler does not
mutate the tangent or its base, but the compiled sampler retains tensor-block
views, so treat both as sampling-owned afterward.

Each nonempty tangent batch builds one right-to-left suffix-environment sweep
and shares it across all shots in that call. With `disk=true`, the first active
completion stays in memory; the remaining suffix environments are written,
loaded once per layer, and deleted as the batch moves left to right. This rule
is independent of `maxsize`, which continues to control only the sampled-prefix
frontier cache.

## Learn more

- [Tutorial](https://Qiaoyi-Li.github.io/Bornsampling.jl/dev/tutorial/)
- [Implementation](https://Qiaoyi-Li.github.io/Bornsampling.jl/dev/implementation/)
- [API](https://Qiaoyi-Li.github.io/Bornsampling.jl/dev/api/)
- [Full Hubbard examples](examples/Hubbard/README.md)
