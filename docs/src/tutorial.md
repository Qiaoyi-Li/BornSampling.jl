# [Tutorial](@id tutorial)

This tutorial samples a half-filled Hubbard chain from a ground-state MPS and
a thermal purification MPO. The MPS produces physical-basis snapshots; the
MPO demonstrates both the exact physical marginal and the joint
physical--purification distribution.

The examples use a chain of length ``L=16``, ``U=8``, ``\mu=U/2``, and bond
dimension ``D=512``. Documentation builds execute every block online, so this
compact system and bond dimension keep the build practical. The calculations
demonstrate the workflow and do not guarantee bond-dimension or sweep
convergence.

```@setup HubbardTutorial
import Bornsampling
using CairoMakie
using Statistics

const FiniteMPS = Bornsampling.FiniteMPS
const TK = Bornsampling.TK
const Random = Bornsampling.Random

include(joinpath(
    dirname(dirname(pathof(Bornsampling))),
    "docs",
    "src",
    "tutorial_helpers.jl",
))
using .BornsamplingTutorialHelpers
```

## [Shared Hubbard setup](@id tutorial_setup)

The Hamiltonian is

```math
H = -\sum_{\langle i,j\rangle,\sigma}
\left(c^\dagger_{i\sigma}c_{j\sigma} + \mathrm{h.c.}\right)
+ U\sum_i n_{i\uparrow}n_{i\downarrow} - \mu\sum_i n_i.
```

The charge U(1) and spin SU(2) representation comes from
`Bornsampling.FiniteMPS`; reduced TensorKit spaces are available through
`Bornsampling.TK`. Model construction, state preparation, direct contraction,
and plotting live in `tutorial_helpers.jl`, leaving the sampling path visible
here.

```@example HubbardTutorial
L = 16
U = 8.0
mu = U / 2
D = 512
Ns_rankone = 1000
Ns_traced = 100
traced_batch_count = 10
ntasks = Threads.nthreads()

hamiltonian = hubbard_hamiltonian(L; U, mu)
sz_lookup = hubbard_sz_lookup()
```

For every curve, pairs at the same integer separation ``r=j-i`` are averaged
inside each shot. Those per-shot radial values determine the sample mean and
its standard error.

## [Ground-state MPS sampling](@id tutorial_mps)

At half filling the total U(1) charge is zero in the particle--hole-centered
convention. A spin-singlet boundary initializes the MPS, and CBE 1-DMRG
prepares the state at bond-dimension ceiling ``D``.

```@example HubbardTutorial
ground_state, ground_result = prepare_ground_state(
    hamiltonian;
    nsites=L,
    D,
    seed=3101,
)
ground_result
```

`Bornsampling.BornSampler` compiles the canonical-basis contraction plans.
The batched call returns one configuration per matrix column together with its
log probability.

```@example HubbardTutorial
ground_direct = direct_szsz_by_distance(ground_state; ntasks)

ground_sampler = Bornsampling.BornSampler(ground_state)
ground_batch = Bornsampling.bornsample!(
    Random.MersenneTwister(3102),
    ground_sampler,
    Ns_rankone;
    ntasks,
)

ground_mean, ground_se = sampled_szsz_by_distance(
    ground_batch.configuration,
    sz_lookup,
)

plot_szsz_comparison(
    ground_direct,
    ((
        mean=ground_mean,
        standard_error=ground_se,
        color=:royalblue,
        label="MPS samples",
    ),);
    direct_label="Ground-state expectation",
    filename="tutorial_hubbard_ground_state_szsz.png",
)
```

![](./figures/tutorial_hubbard_ground_state_szsz.png)

The sample mean estimates the diagonal spin correlation, and the whiskers show
its standard error. The Hamiltonian is retained for thermal preparation.

```@example HubbardTutorial
ground_batch = nothing
ground_sampler = nothing
ground_state = nothing
GC.gc()
```

## [Thermal purification MPO sampling](@id tutorial_mpo)

For a thermal factor ``|X\rangle``, `purified=true` draws the exact physical
marginal

```math
p(x) = \frac{\sum_y |X(x,y)|^2}{\langle X|X\rangle},
```

while `purified=false` draws the joint distribution

```math
p(x,y) = \frac{|X(x,y)|^2}{\langle X|X\rangle}.
```

Selecting the physical part of a joint sample gives the same marginal
``p(x)``. SETTN initializes the thermal factor at high temperature, and CBE
1-TDVP evolves it on an exponential grid to ``\beta=1``.

```@example HubbardTutorial
factor, beta = prepare_thermal_factor(hamiltonian; D)
beta
```

### Exact physical marginal

The direct thermal contraction provides the reference curve. The traced
sampler then produces physical configurations from the exact marginal.
Running the traced shots as several small batches trades lower peak
prefix-bank memory for less parallel work and prefix reuse per call.

```@example HubbardTutorial
thermal_direct = direct_szsz_by_distance(factor; ntasks)

traced_state = deepcopy(factor)
traced_sampler = Bornsampling.BornSampler(traced_state; purified=true)
traced_rng = Random.MersenneTwister(3201)
traced_configuration = Matrix{Int}(undef, L, Ns_traced)
traced_batch_size = Ns_traced ÷ traced_batch_count

for batch in 1:traced_batch_count
    columns = ((batch - 1) * traced_batch_size + 1):(batch * traced_batch_size)
    result = Bornsampling.bornsample!(
        traced_rng,
        traced_sampler,
        traced_batch_size;
        ntasks,
    )
    traced_configuration[:, columns] .= result.configuration
end

traced_mean, traced_se = sampled_szsz_by_distance(
    traced_configuration,
    sz_lookup,
)

traced_configuration = nothing
traced_sampler = nothing
traced_state = nothing
GC.gc()
```

### Joint physical--purification samples

Joint sampling returns a ``2L\times N_s`` configuration matrix ordered as
``[x_1,\ldots,x_L,y_1,\ldots,y_L]``. The physical rows feed the same diagonal
observable estimator.

```@example HubbardTutorial
joint_sampler = Bornsampling.BornSampler(factor; purified=false)
joint_batch = Bornsampling.bornsample!(
    Random.MersenneTwister(3202),
    joint_sampler,
    Ns_rankone;
    ntasks,
)
joint_physical_configuration = @view joint_batch.configuration[1:L, :]
joint_mean, joint_se = sampled_szsz_by_distance(
    joint_physical_configuration,
    sz_lookup,
)

joint_summary = (;
    joint_configuration_size=size(joint_batch.configuration),
    samples=Ns_rankone,
)
joint_summary
```

The direct contraction, traced marginal samples, and physical part of the
joint samples form the three thermal comparison curves.

```@example HubbardTutorial
plot_szsz_comparison(
    thermal_direct,
    (
        (
            mean=traced_mean,
            standard_error=traced_se,
            color=:royalblue,
            label="Trace (×$(Ns_traced))",
        ),
        (
            mean=joint_mean,
            standard_error=joint_se,
            color=:darkorange,
            label="Joint (×$(Ns_rankone))",
        ),
    );
    direct_label="Thermal expectation",
    filename="tutorial_hubbard_thermal_szsz.png",
)
```

![](./figures/tutorial_hubbard_thermal_szsz.png)
