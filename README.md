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
state, while local MPO purification legs and a persistent tangent symmetry leg
are traced. Constructing this sampler does not mutate the tangent or its base,
but the compiled sampler retains tensor-block views, so treat both as
sampling-owned afterward.

## Learn more

- [Tutorial](https://Qiaoyi-Li.github.io/Bornsampling.jl/dev/tutorial/)
- [Implementation](https://Qiaoyi-Li.github.io/Bornsampling.jl/dev/implementation/)
- [API](https://Qiaoyi-Li.github.io/Bornsampling.jl/dev/api/)
- [Full Hubbard examples](examples/Hubbard/README.md)
