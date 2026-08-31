# BornSampling.jl

```@meta
CurrentModule = BornSampling
```

This repository is a downstream package of
[`FiniteMPS.jl`](https://github.com/Qiaoyi-Li/FiniteMPS.jl) for drawing
canonical-basis snapshots of MPS and MPO according to their Born probabilities.
It outputs the sampled configurations and their log probabilities for
subsequent statistical analysis.

## First sample

```@example first_sample
import BornSampling
using BornSampling.FiniteMPS: randMPS, ℂ
using BornSampling.Random: MersenneTwister, seed!

seed!(1234)
state = randMPS(4, ℂ^2, ℂ^1)
sampler = BornSampling.BornSampler(state)

rng = MersenneTwister(1234)
shot = BornSampling.bornsample!(rng, sampler)

(
    configuration = shot.configuration,
    log_probability = shot.log_probability,
)
```

Sampler construction canonicalizes `state` in place. The
compiled sampler then owns that canonicalized tensor data and can be reused for
many shots.

Continue with the [Tutorial](@ref tutorial), [Implementation](@ref
implementation), and [Public API](@ref public_api).

```@contents
Pages = ["tutorial.md", "implementation.md", "api.md"]
Depth = 2
```
