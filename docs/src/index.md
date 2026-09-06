# BornSampling.jl

```@meta
CurrentModule = BornSampling
```

BornSampling draws canonical-basis snapshots of MPS and MPO states according to
their Born probabilities, based on
[`FiniteMPS.jl`](https://github.com/Qiaoyi-Li/FiniteMPS.jl). It returns the
sampled configurations alongside their log probabilities.

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

Constructing a `BornSampler` canonicalizes `state` in place and compiles
contraction plans that can be reused across repeated sampling calls.

See the [Tutorial](@ref tutorial), [Implementation](@ref implementation), and
[Public API](@ref public_api) for details.

```@contents
Pages = ["tutorial.md", "implementation.md", "api.md"]
Depth = 2
```
