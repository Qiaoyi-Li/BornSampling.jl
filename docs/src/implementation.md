# [Implementation](@id implementation)

BornSampling samples configurations using a single left-to-right factor
propagation. The probability model dictates the sampling mode, while compiled
local plans perform symmetry-aware contractions.

## Probability model

### MPS amplitudes with an open boundary

For an MPS, local tensors follow the canonical index ordering

```math
A_i[\ell_i,x_i,r_i],
```

and the contracted network defines the amplitude ``\psi(b,\boldsymbol{x})``,
where ``b`` labels the left-boundary basis with irrep dimension ``d_b``.

With `purified=true`, the boundary is traced out to sample physical
configurations:

```math
p(\boldsymbol{x})
= \frac{\sum_b|\psi(b,\boldsymbol{x})|^2}
        {\sum_{b,\boldsymbol{x}'}|\psi(b,\boldsymbol{x}')|^2}.
```

With `purified=false`, the boundary index ``b`` is sampled first from the joint
distribution, returning ``[\boldsymbol{x};b]`` of length ``L+1``.

### Purification amplitudes

An MPO is treated as a pure state defined over its open left boundary, physical
spaces, and local purification spaces. A rank-four local tensor has index
ordering

```math
X_i[\ell_i,x_i,y_i,r_i],
```

where ``x_i`` and ``y_i`` denote physical and purification indices,
respectively. At rank-three sites lacking a purification leg, the output sets
``y_i=0``. The complete network defines ``X(b,\boldsymbol{x},\boldsymbol{y})``
and the joint distribution

```math
p(\boldsymbol{x},\boldsymbol{y},b)
= \frac{|X(b,\boldsymbol{x},\boldsymbol{y})|^2}
        {\langle X|X\rangle}.
```

The two sampling modes correspond to:

```math
\begin{aligned}
\text{traced mode:}\quad
p(\boldsymbol{x})
  &= \sum_{\boldsymbol{y},b}p(\boldsymbol{x},\boldsymbol{y},b),\\
\text{joint mode:}\quad
(\boldsymbol{x},\boldsymbol{y},b)
  &\sim p(\boldsymbol{x},\boldsymbol{y},b).
\end{aligned}
```

Setting `purified=true` samples from the traced physical marginal
``p(\boldsymbol{x})``, returning ``L`` physical indices. Setting
`purified=false` draws the boundary ``b`` followed by all physical and
purification indices, returning ``[\boldsymbol{x};\boldsymbol{y};b]`` of length
``2L+1``. Discarding ``\boldsymbol{y}`` and ``b`` from joint samples recovers
the exact physical marginal.

### Tangent-vector states

A `FiniteMPSTangents.TangentMPS` represents a tangent state with open left
boundary ``b``:

```math
|\Phi_{b,\boldsymbol y,q}\rangle
=\left[\sum_{j=1}^L
A^l_1\cdots A^l_{j-1}B_{j,q}A^r_{j+1}\cdots A^r_L\right]_b,
```

where the sum over insertion sites is coherent, and ``q`` is an optional global
symmetry leg.

With `purified=true`, the boundary, purification legs, and global leg are
traced out:

```math
\rho=\sum_{b,\boldsymbol y,q}
|\Phi_{b,\boldsymbol y,q}\rangle\langle\Phi_{b,\boldsymbol y,q}|,
```

returning ``L`` physical indices. With `purified=false`, ``b`` and ``q`` are
sampled prior to site updates, returning
``[\boldsymbol x;\boldsymbol y;b;q]``. If the base state contains rank-four
tensors, each site includes a ``y_i`` slot (recording ``0`` at rank-three
sites); if all base tensors are rank three, the ``\boldsymbol y`` group is
omitted.

### Sequential probabilities

At site ``i``, let ``a_i`` denote the sampled outcome: ``x_i`` in traced or MPS
modes, or ``(x_i, y_i)`` at a rank-four joint-mode site. Contracting the site
tensor against the current environment yields branch weights ``q_i(a_i)``:

```math
z_i = \sum_{a_i}q_i(a_i), \qquad
p(a_i\mid a_1,\ldots,a_{i-1}) = \frac{q_i(a_i)}{z_i}.
```

The physical sites contribute

```math
\sum_{i=1}^{L}\left[\log q_i(a_i)-\log z_i\right]
```

to the total log probability. In joint mode, the conditional log probabilities
of the initial boundary draw ``b`` (and global tangent leg ``q``, if present)
are included, giving the total log probability of the sampled configuration.

### Unbiased estimates from samples

For ``N`` independent physical samples ``\boldsymbol{x}^{(s)}``, the empirical
probability of a configuration ``\boldsymbol{x}_0`` is

```math
\widehat p(\boldsymbol{x}_0)
= \frac{N_{\boldsymbol{x}_0}}{N}
= \frac{1}{N}\sum_{s=1}^{N}
   \mathbf{1}\!\left[\boldsymbol{x}^{(s)}=\boldsymbol{x}_0\right],
```

which is an unbiased estimator:

```math
\mathbb{E}\,\widehat p(\boldsymbol{x}_0)
= \frac{1}{N}\sum_{s=1}^{N}
   \mathbb{P}\!\left(\boldsymbol{x}^{(s)}=\boldsymbol{x}_0\right)
= p(\boldsymbol{x}_0).
```

The same property holds for the physical marginal of joint samples, since

```math
\mathbb{P}(\boldsymbol{x}^{(s)}=\boldsymbol{x}_0)
= \sum_{\boldsymbol{y},b}p(\boldsymbol{x}_0,\boldsymbol{y},b)
= p(\boldsymbol{x}_0).
```

For an observable diagonal in the sampling basis with values
``O(\boldsymbol{x})``, the sample mean

```math
\widehat{\langle O\rangle}
= \frac{1}{N}\sum_{s=1}^{N}O(\boldsymbol{x}^{(s)})
```

satisfies ``\mathbb{E}\,\widehat{\langle O\rangle} = \langle O\rangle``. Its
standard error is estimated by ``s_O/\sqrt{N}``, where ``s_O^2`` is the sample
variance.

## Factorized boundary propagation

After canonicalization at site 1, the sampler maintains the left environment in
factorized form:

```math
\rho_i = C_i C_i^\dagger.
```

For traced boundary sampling, the initial factor is
``C_1=I_{d_b}/\sqrt{d_b}``, with one column per boundary state. Defining the
local slice and propagated factor as

```math
K_i(x,y)[r,\ell]=X_i[\ell,x,y,r],
\qquad
Y_i(x,y)=K_i(x,y)C_i,
```

each physical-site update preserves unit Frobenius norm.

### Traced MPO mode

When computing branch weights, purification outcomes are stacked horizontally
as column blocks:

```math
G_i(x)=
\begin{bmatrix}
Y_i(x,1) & Y_i(x,2) & \cdots & Y_i(x,d_y)
\end{bmatrix}.
```

The node caches the bank ``\mathcal{G}_i=(G_i(1),\ldots,G_i(d_x))``, yielding
branch weights directly:

```math
q_i(x)=\lVert G_i(x)\rVert_F^2
=\sum_y\lVert K_i(x,y)C_i\rVert_F^2.
```

When child branches are traversed, the selected factor
``G_i(x) / \sqrt{q_i(x)}`` is compressed via LQ decomposition ``G_i = L_i Q_i``
whenever the column dimension exceeds the right bond dimension. Because
``Q_i Q_i^\dagger = I``,

```math
G_i G_i^\dagger = L_i L_i^\dagger,
```

replacing ``G_i`` with ``L_i`` preserves the environment exactly while capping
the rank at the right bond dimension. Storing the uncompressed bank
``\mathcal{G}_i`` allows single-pass weight evaluation and reuse across child
nodes.

### MPS and joint modes

A traced MPS propagates its ``d_b`` boundary columns through each selected
channel. When the column count exceeds the bond dimension ``D_i``, LQ
compression reduces the rank without altering ``C_i C_i^\dagger``.

In joint sampling, the boundary state is sampled at site 1, reducing ``C_1`` to
a single column. Subsequent site updates:

```math
C_{i+1}=\frac{K_i(x,y)C_i}{\sqrt{q_i(x,y)}}
```

preserve this rank-one property, scaling as ``\mathcal{O}(D^2)`` per
contraction. For traced MPS with ``r \le d_b`` columns, each update costs
``\mathcal{O}(r D^2)``. In traced MPO mode, purification legs can increase the
rank up to ``D``, yielding ``\mathcal{O}(D^3)`` scaling in the worst case.

For tangent joint sampling, conditioning on the boundary, the global ``q`` (if
present), and local purification indices maintains a rank-one factor structure
with ``\mathcal{O}(D^2)`` propagation cost. In traced tangent mode, tracing
purification legs similarly requires history compression and scales as
``\mathcal{O}(D^3)``.

## Global modes and local tensor ranks

The input type and `purified` flag determine the sampling semantics:

| Input type | Sampled outcomes | Factor update |
|:--|:--|:--|
| `MPS`, `purified=true` | ``x_i`` | Boundary trace with LQ compression |
| `MPS`, `purified=false` | Boundary ``b``, then ``x_i`` | Rank-one propagation |
| `MPO`, `purified=true` | ``x_i`` | Boundary and purification trace with LQ compression |
| `MPO`, `purified=false` | Boundary ``b``, then ``(x_i, y_i)`` | Rank-one propagation |
| `TangentMPS`, `purified=true` | ``x_i`` | Traced boundary in ``(U, V_q)`` history |
| `TangentMPS`, `purified=false` | Boundary ``b``, optional ``q``, then local outcomes | Rank-one history propagation |

An MPO may contain both rank-three and rank-four tensors. At rank-three sites,
contractions treat the purification leg as a singleton and joint sampling sets
``y_i=0``.

## Compiled symmetry contractions

Each site is compiled into a `SitePlan` containing basis metadata, views into
reduced `TensorMap` blocks, allowed sector transitions, and cached fusion-tree
kernels. A physical flat index maps to
``(\text{sector}, \text{degeneracy}, \text{irrep})`` in canonical `TensorKit`
order.

### Residual symmetry

Projecting onto a local basis state can break part of the global symmetry. The
compiler determines the residual symmetry structurally: one-dimensional
`UniqueFusion` components are preserved as residual charges, while non-Abelian
carrier indices are absorbed into the row degeneracy of the residual tensor
blocks.

Original virtual sectors are grouped into residual sector blocks. Transitions
sharing the same residual charges form a `ChannelRoute`, accumulating
amplitudes coherently before taking the norm. This preserves block sparsity
while accounting for interference between sectors sharing a residual charge.

### Contraction kernels

The contraction kernel is determined by
`BornSampling.TK.FusionStyle(sector_type)`:

- `UniqueStyle`: Directly contracts one-dimensional irrep carriers via matrix
  multiplication on reduced blocks.
- `FusionTreeStyle`: Uses cached fusion-tree arrays to contract degeneracy and
  irrep coordinates into preallocated scratch buffers.

## Layer-synchronous prefix reuse

Batched sampling advances all shots layer by layer from left to right. Prefixes
entering a layer form the current frontier, while selected branches are
collected into the next frontier. Each prefix node stores its accumulated log
probability, next-site branch weights, and environment tensor. When multiple
shots follow the same prefix path, they share the node's branch weights and
environment.

Within each layer, an atomic counter assigns work to worker tasks. In MPS and
joint-MPO batches, shots sharing a nonresident parent form one job so its factor
is loaded once for the group; shots with resident parents remain individual
jobs. Each worker operates with its own contraction workspace, and per-shot
random number generators are seeded upfront to ensure reproducibility
regardless of task scheduling.

When all shots complete a layer, the next frontier becomes active and the
preceding frontier is released. Child nodes are published with fine-grained
locking: only the first worker to visit an edge constructs the new node, while
subsequent workers follow a lock-free path.

### Tangent suffix completions

Before sampling a tangent batch, a right-to-left sweep builds suffix
environments in the original symmetry:

```math
I=R^\dagger R,\qquad K_q=R^\dagger T_q,\qquad
N_{q,q'}=T_q^\dagger T_{q'}.
```

Joint sampling retains both ``q`` legs of ``N`` during the sweep; traced sampling
uses ``N_{\mathrm{tr}}=\sum_q N_{q,q}``. The sweep runs once per batch, with cost
determined by the original symmetry's reduced blocks and fusion channels. Each
completed suffix is converted to residual-sector matrices for storage, keeping
only ``N_{q,q}`` in joint mode, while the next transfer uses the original-symmetry
environment. Forward sampling starts after all completions are prepared and
shares each site completion read-only across worker tasks.

With `disk=false`, converted suffix completions are held in memory. With
`disk=true`, they are written directly to temporary files and deleted after
consumption, keeping resident suffix memory constant with respect to chain
length. Neither mode stores a chain of original-symmetry environments.

## Probability-ranked environment storage

Prefix metadata and branch weights remain in memory, while tensor environments
can be offloaded to disk. When disk storage is enabled, each frontier retains at
most `maxsize` environments with the highest prefix probabilities in memory;
remaining environments are serialized to disk.

Because frontiers are ranked independently, each layer retains its top-`maxsize`
environments. As shots transition between layers, environments are admitted to
memory or written to disk through atomic operations. When a frontier is released
at the layer boundary, its temporary files are cleaned up.
