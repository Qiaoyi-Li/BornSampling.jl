# BornSampling

[![Stable documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://Qiaoyi-Li.github.io/BornSampling.jl/stable/)
[![Development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://Qiaoyi-Li.github.io/BornSampling.jl/dev/)
[![CI](https://github.com/Qiaoyi-Li/BornSampling.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Qiaoyi-Li/BornSampling.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/Qiaoyi-Li/BornSampling.jl/graph/badge.svg?branch=main)](https://codecov.io/gh/Qiaoyi-Li/BornSampling.jl)

This repository is a downstream package of
[`FiniteMPS.jl`](https://github.com/Qiaoyi-Li/FiniteMPS.jl) for
drawing canonical-basis snapshots of MPS and MPO according to their Born probabilities. It outputs the sampled configurations and their log
probabilities for subsequent statistical analysis.

## Installation

From the Julia package prompt:

```julia-repl
pkg> add BornSampling
```

## Quick start

```julia
import BornSampling
using BornSampling.FiniteMPS: randMPS, ℂ
using BornSampling.Random: MersenneTwister, seed!

seed!(1234)
state = randMPS(4, ℂ^2, ℂ^1)
rng = MersenneTwister(1234)
sampler = BornSampling.BornSampler(state)

config = Vector{Int}(undef, length(state))
logp = BornSampling.bornsample!(rng, sampler, config)

batch = BornSampling.bornsample!(
    rng,
    sampler,
    1000;
    ntasks=Threads.nthreads(),
)
```

Constructing a sampler canonicalizes its state in place and retains compiled
views of the local tensors. Reuse that sampler for repeated shots and treat its
state as sampling-owned afterward.

## Learn more

- [Tutorial](https://Qiaoyi-Li.github.io/BornSampling.jl/dev/tutorial/)
- [Implementation](https://Qiaoyi-Li.github.io/BornSampling.jl/dev/implementation/)
- [API](https://Qiaoyi-Li.github.io/BornSampling.jl/dev/api/)
- [Full Hubbard examples](examples/Hubbard/README.md)
