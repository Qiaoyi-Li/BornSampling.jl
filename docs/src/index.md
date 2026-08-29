# Bornsampling.jl

```@meta
CurrentModule = Bornsampling
```

This documentation was generated with AI assistance and is provided for reference only.

This repository is a downstream package of
[`FiniteMPS.jl`](https://github.com/Qiaoyi-Li/FiniteMPS.jl) for drawing
canonical-basis snapshots of MPS and MPO according to their Born probabilities.
It outputs the sampled configurations and their log probabilities for
subsequent statistical analysis.

## First sample

```julia
import Bornsampling
using Bornsampling.FiniteMPS
using Bornsampling.Random

state_for_sampling = deepcopy(state)
sampler = Bornsampling.BornSampler(state_for_sampling)

rng = MersenneTwister(1234)
shot = Bornsampling.bornsample!(rng, sampler)

shot.configuration
shot.log_probability
```

Sampler construction canonicalizes `state_for_sampling` in place. The
compiled sampler then owns that canonicalized tensor data and can be reused for
many shots.

Continue with the [Tutorial](@ref tutorial), [Implementation](@ref
implementation), and [Public API](@ref public_api).

```@contents
Pages = ["tutorial.md", "implementation.md", "api.md"]
Depth = 2
```
