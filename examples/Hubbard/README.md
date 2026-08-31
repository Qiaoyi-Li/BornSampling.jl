# Hubbard cylinder examples

This directory contains two executable Born-sampling examples for the Hubbard
model on a YC4×8 square-lattice cylinder. Both use U(1) charge and SU(2) spin
symmetry, compare a direct ``S^zS^z=(\mathbf S\cdot\mathbf S)/3`` contraction
with a batched Born-sample estimate, and plot the correlations averaged by
cylinder distance.

## Set up and run

From this directory, instantiate the example project once:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Run either calculation with four Julia threads:

```sh
julia --threads=4 --project=. hubbard_sampling.jl
julia --threads=4 --project=. hubbard_tantrg_sampling.jl
```

The main model, bond-dimension, sampling, task-count, and random-seed parameters
are defined near the top of each script.

## Ground-state sampling

`hubbard_sampling.jl` prepares an approximate ground-state MPS with RSVD-CBE
one-site DMRG. Its default model has hopping ``t=1``, interaction ``U=12``,
one-eighth hole doping (28 electrons on 32 sites), target bond dimension
``D=1024``, and ``N_s=1000`` Born samples.

The script computes the direct and sampled ``S^zS^z`` matrices, their
distance-averaged curves, and standard errors for the sampled estimates. It
writes

```text
output/hubbard_szsz_vs_distance.png
```

The console reports DMRG, direct-contraction, sampler-compilation, and sampling
times, including sampling time per shot. `main()` returns the energy history,
timings, direct correlation matrix, sampled correlation matrix, and its
elementwise standard-error estimate.

## Thermal sampling at ``\beta=1``

`hubbard_tantrg_sampling.jl` prepares a half-filled thermal MPO factor with
``U=8``, particle-hole-symmetric chemical potential ``\mu=4``, target bond
dimension ``D=1024``, and ``N_s=100`` Born samples. SETTN starts the thermal
factor at ``\beta=2^{-15}``; CBE/RSVD one-site TDVP evolves it along
``2^{-15},\ldots,2^0`` to ``\beta=1``.

The script computes the direct and sampled thermal ``S^zS^z`` matrices,
distance-averaged curves, and sample standard errors. It writes

```text
output/hubbard_tantrg_beta1_szsz_vs_distance.png
```

The console reports tanTRG, direct-contraction, sampler-compilation, and
sampling times, including sampling time per shot. `main()` returns the timing
summary, shot count, direct correlation matrix, sampled correlation matrix,
and its elementwise standard-error estimate.

## Purification sampling mode

The thermal script constructs `BornSampling.BornSampler(factor)` with the
default `purified=true`. This draws physical configurations from the exact
marginal obtained by tracing the MPO purification index.

For joint physical--purification snapshots, construct
`BornSampling.BornSampler(factor; purified=false)`. The resulting configuration
matrix places the physical rows first and the purification rows second:

```text
x₁, …, xL, y₁, …, yL
```
