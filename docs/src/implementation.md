# [Implementation](@id implementation)

BornSampling is organized around one left-to-right factor propagation.  The
probability model determines the sampling mode, while compiled local plans
provide the symmetry-aware contractions used by every mode.

## Probability model

### MPS amplitudes with an open boundary

For an MPS, each local tensor has the canonical order

```math
A_i[\ell_i,x_i,r_i],
```

and the contracted network defines ``\psi(b,\boldsymbol{x})``, where ``b``
labels the full left-boundary basis. The boundary contains one original sector
with reduced multiplicity one; its full irrep dimension ``d_b`` may exceed
one for a non-Abelian symmetry.

With `purified=true`, BornSampling traces the boundary and draws

```math
p(\boldsymbol{x})
= \frac{\sum_b|\psi(b,\boldsymbol{x})|^2}
        {\sum_{b,\boldsymbol{x}'}|\psi(b,\boldsymbol{x}')|^2}.
```

With `purified=false`, a synthetic first layer draws ``b`` before the physical
sites from the joint Born distribution. The returned configuration is
``[\boldsymbol{x};b]``, of length ``L+1``, including ``b=1`` when ``d_b=1``.

### Purification amplitudes

An MPO is interpreted as a pure amplitude on its open left boundary, physical
spaces, and local purification spaces. A rank-four local tensor has the order

```math
X_i[\ell_i,x_i,y_i,r_i],
```

and a rank-three site records ``y_i=0`` in joint output to mark its absent
purification leg. Actual purification legs use one-based basis indices,
including ``1`` for a one-dimensional leg. The complete network defines
``X(b,\boldsymbol{x},\boldsymbol{y})`` and hence the joint distribution

```math
p(\boldsymbol{x},\boldsymbol{y},b)
= \frac{|X(b,\boldsymbol{x},\boldsymbol{y})|^2}
        {\langle X|X\rangle}.
```

The two MPO modes follow directly from this distribution:

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

`purified=true` selects traced mode and `purified=false` selects joint mode.
Joint sampling draws ``b`` first and returns
``[\boldsymbol{x};\boldsymbol{y};b]``, of length ``2L+1``. Each site has a
``y_i`` slot, and the boundary entry is always retained.

Discarding ``\boldsymbol{y}`` and ``b`` from joint samples produces samples
distributed according to the exact physical marginal ``p(\boldsymbol{x})``.

### Tangent-vector states

A `FiniteMPSTangents.TangentMPS` represents the Hilbert-space state with an
open left-boundary index ``b``:

```math
|\Phi_{b,\boldsymbol y,q}\rangle
=\left[\sum_{j=1}^L
A^l_1\cdots A^l_{j-1}B_{j,q}A^r_{j+1}\cdots A^r_L\right]_b.
```

The insertion-site sum is coherent: all cross terms at fixed
``(b,\boldsymbol y,q)`` are retained. The extra ``q`` leg is one persistent
global label shared by all insertion positions. With `purified=true`, tangent
sampling traces the left boundary, local purification legs, and global leg:

```math
\rho=\sum_{b,\boldsymbol y,q}
|\Phi_{b,\boldsymbol y,q}\rangle\langle\Phi_{b,\boldsymbol y,q}|.
```

With `purified=false`, synthetic layers draw ``b`` and then ``q`` when present,
followed by site outcomes. Joint output is ``[\boldsymbol x;\boldsymbol y;b;q]``:
``b`` is always included, and ``q`` is included when its leg exists. If any
base tensor has rank four, the tangent is MPO-like and has one ``y_i`` slot
per site, with ``0`` at rank-three sites and one-based indices at rank-four
sites. With all rank-three base tensors, it is MPS-like and omits the
``\boldsymbol y`` group. Traced output contains the ``L`` physical indices.

### Sequential probabilities

At physical site ``i``, let ``a_i`` denote the sampled local outcome: ``x_i``
in MPS and traced modes, or ``(x_i,y_i)`` at a rank-four joint-mode site. A
rank-three joint-mode site samples ``x_i``. The contraction produces
nonnegative branch weights ``q_i(a_i)`` and

```math
z_i = \sum_{a_i}q_i(a_i), \qquad
p(a_i\mid a_1,\ldots,a_{i-1}) = \frac{q_i(a_i)}{z_i}.
```

The physical-site contribution to the returned log probability is

```math
\sum_{i=1}^{L}\left[\log q_i(a_i)-\log z_i\right].
```

Joint sampling also includes the conditional log probabilities of its initial
boundary draw and, for a tangent with a global leg, the subsequent ``q`` draw.
Thus `log_probability` describes the complete returned configuration.

### Unbiased estimates from samples

For ``N`` independent physical samples ``\boldsymbol{x}^{(s)}``, the empirical
probability of a fixed configuration ``\boldsymbol{x}_0`` is

```math
\widehat p(\boldsymbol{x}_0)
= \frac{N_{\boldsymbol{x}_0}}{N}
= \frac{1}{N}\sum_{s=1}^{N}
   \mathbf{1}\!\left[\boldsymbol{x}^{(s)}=\boldsymbol{x}_0\right].
```

Its expectation is

```math
\mathbb{E}\,\widehat p(\boldsymbol{x}_0)
= \frac{1}{N}\sum_{s=1}^{N}
   \mathbb{P}\!\left(\boldsymbol{x}^{(s)}=\boldsymbol{x}_0\right)
= p(\boldsymbol{x}_0).
```

The same derivation applies after retaining only ``\boldsymbol{x}`` from joint
samples, because

```math
\mathbb{P}(\boldsymbol{x}^{(s)}=\boldsymbol{x}_0)
= \sum_{\boldsymbol{y},b}p(\boldsymbol{x}_0,\boldsymbol{y},b)
= p(\boldsymbol{x}_0).
```

For an observable diagonal in the sampled basis, with value
``O(\boldsymbol{x})``, the sample mean

```math
\widehat{\langle O\rangle}
= \frac{1}{N}\sum_{s=1}^{N}O(\boldsymbol{x}^{(s)})
```

is unbiased since

```math
\mathbb{E}\,\widehat{\langle O\rangle}
= \sum_{\boldsymbol{x}}p(\boldsymbol{x})O(\boldsymbol{x})
= \langle O\rangle.
```

With the usual unbiased sample variance ``s_O^2``, ``s_O^2/N`` estimates the
variance of this mean and ``s_O/\sqrt{N}`` supplies its standard error.

## Factorized boundary propagation

After canonicalizing an MPS or MPO at site 1, the sampler needs only the
collapsed left environment. It stores that environment as a factor

```math
\rho_i = C_i C_i^\dagger.
```

Let ``C_i`` denote the factor entering physical site ``i``. Tracing the left
boundary initializes ``C_1=I_{d_b}/\sqrt{d_b}``, with one column per full-basis
boundary value. Each physical-site update normalizes its output to unit
Frobenius norm. For a fixed local physical and purification basis value, define

```math
K_i(x,y)[r,\ell]=X_i[\ell,x,y,r],
\qquad
Y_i(x,y)=K_i(x,y)C_i.
```

### Traced MPO mode

During the weight pass, every physical outcome receives an uncompressed factor
whose purification outcomes become distinct columns:

```math
G_i(x)=
\begin{bmatrix}
Y_i(x,1) & Y_i(x,2) & \cdots & Y_i(x,d_y)
\end{bmatrix}.
```

The node stores the complete environment bank
``\mathcal{G}_i=(G_i(1),\ldots,G_i(d_x))`` and obtains every branch weight from
that same pass:

```math
q_i(x)=\lVert G_i(x)\rVert_F^2
=\sum_y\lVert K_i(x,y)C_i\rVert_F^2,
```

When an outcome ``x`` first creates a child edge, that child takes the already
stored ``G_i(x)`` from the bank and normalizes it by ``\sqrt{q_i(x)}``. A
residual-symmetry block with more columns than right-space rows receives the
exact LQ factorization ``G_i=L_iQ_i`` at this point. Since
``Q_iQ_i^\dagger=I``,

```math
G_iG_i^\dagger=L_iL_i^\dagger,
```

so replacing that block by ``L_i`` preserves the density factor exactly and
bounds its column count by the right bond dimension. Each resident traced node
therefore reserves space for all physical-branch factors; this larger
environment payload enables one-pass weight construction and immediate reuse
by every child that is reached.

### MPS and joint modes

A traced MPS propagates its ``d_b`` boundary columns through each selected
physical channel. Its rank after site ``i`` is bounded by ``\min(d_b,D_i)``,
where ``D_i`` is the full right bond dimension. Exact LQ compression reduces
residual blocks with more columns than rows, preserving ``CC^\dagger``.

Joint MPS and MPO sampling use the first-site contraction to draw the boundary
value and conditionally normalize one ket column. They then select ``(x,y)``
at rank-four sites and ``x`` at rank-three sites, with update

```math
C_{i+1}=\frac{K_i(x,y)C_i}{\sqrt{q_i(x,y)}},
```

The factor remains rank one. These modes therefore inherit the usual
dense-equivalent ``O(D^2)`` sequential contraction. A traced MPS with ``r``
active columns requires ``O(D^2r)`` work per local contraction, with
``r\leq d_b``. Tracing local MPO purification can raise the factor rank to
``D`` and gives ``O(D^3)`` worst-case work. In every mode the environment
remains represented lazily by ``C_iC_i^\dagger``.

For tangent joint sampling, selecting the boundary, the global ``q`` when
present, and one local ``y_i`` at each rank-four site keeps the history factor
rank one. The hot path applies each compiled residual-block route directly to
the ``U`` and ``V_q`` factor columns. Thus the
fixed-``q`` MPS/MPO joint propagation has the same dense-equivalent
``O(D^2)`` character. Traced local purification may require a common exact
history compression and has ``O(D^3)`` worst-case work.

## Global modes and local tensor ranks

The outer `BornSampling.FiniteMPS` concrete type chooses the global semantics:

| input and constructor | sampled outcome | factor update |
|:--|:--|:--|
| `BornSampling.FiniteMPS.MPS`, `purified=true` | ``x_i`` | boundary trace and exact factor compression |
| `BornSampling.FiniteMPS.MPS`, `purified=false` | one boundary ``b``, then ``x_i`` | rank-one propagation |
| `BornSampling.FiniteMPS.MPO`, `purified=true` | ``x_i`` | boundary and local purification trace, with exact compression |
| `BornSampling.FiniteMPS.MPO`, `purified=false` | one boundary ``b``, then local outcomes | rank-one propagation |
| `FiniteMPSTangents.TangentMPS`, `purified=true` | ``x_i`` | boundary trace in common ``(U,V_q)`` history |
| `FiniteMPSTangents.TangentMPS`, `purified=false` | one boundary ``b``, optional global ``q``, then local outcomes | rank-one history propagation |

Construction checks once that every tensor in an MPS is rank three.  An MPO
may mix rank-three and rank-four sites. At a rank-three MPO site, contraction
uses a singleton purification basis and joint output records ``y_i=0``. A
traced MPO carries the factor rank from its boundary and earlier rank-four
sites. This division keeps the probability model global while local static
rank controls only leg access and the contraction kernel.

## Compiled symmetry contractions

Each site is compiled into a `SitePlan`.  The plan records full-basis metadata,
views of the reduced TensorMap blocks, allowed sector transitions, and cached
fusion-tree kernels.  A physical flat index resolves to
``(sector, degeneracy, irrep)`` in TensorKit's canonical array order, so fixing
a sampled basis vector fixes both its degeneracy and irrep coordinates.

### Residual symmetry

Fixing local basis vectors can leave only part of the original symmetry
unbroken.  Compilation infers this residual symmetry structurally.  A
one-dimensional `UniqueFusion` component is retained as a residual charge; in
a product symmetry this decision is made component by component.  Remaining
components, including non-Abelian carrier coordinates, are folded into the
row degeneracy of the residual TensorMap.

Every original virtual sector is embedded into a residual sector block.  Local
transitions with the same residual input and output charges form one
`ChannelRoute`, and their amplitudes are accumulated coherently before taking
a norm.  This representation preserves charge block sparsity while retaining
the interference between original sectors that share a residual charge.

### Unique and fusion-tree paths

The contraction style is selected once through
`BornSampling.TK.FusionStyle(sector_type)`:

- `UniqueStyle` contracts one-dimensional irrep carriers directly from the
  reduced block view with matrix multiplication.
- `FusionTreeStyle` caches `convert(Array, (fout, fin))` during compilation.
  At sampling time it slices the reduced block and the cached fusion kernel,
  then contracts their degeneracy and irrep coordinates with preallocated
  scratch space.

Both paths propagate the ket with ordinary `transpose`; conjugation enters
through Frobenius norms and ``CC^\dagger``.  Rank-three and rank-four methods
supply the corresponding reduced and fusion-kernel slices, while the site
loop and probability calculation remain shared.

## Layer-synchronous prefix reuse

A batched call advances every shot through the synthetic and physical layers.
The current frontier stores the sampled prefixes entering that layer; a
separate next frontier receives the children selected there. Each node carries its prefix log
probability, next-site branch weights, child slots, and TensorMap space metadata
for its numerical environment. MPS nodes own one factor, with several columns
when their boundary is traced. Joint-MPO nodes own one rank-one factor.
Traced-MPO nodes own the complete uncompressed ``G_x`` bank produced by their
weight pass. Shots that reach the same prefix reuse its branch
weights and environment payload.

Within a layer, a shared atomic counter dynamically assigns shots to workers.
Every worker has an independent numerical workspace, and every shot retains its
own seeded RNG across layers. The seed sequence is drawn from the caller's RNG
before work starts, so scheduling does not change the sampled result. `ntasks`
controls the number of Julia tasks up to the number of shots; Julia schedules
those tasks over the available threads.

After all shots finish a layer, the barrier makes the next frontier complete.
The preceding frontier's metadata, environments, and temporary files are then
released before the next site starts. A frontier contains at most ``N`` nodes;
during a transition only the adjacent current and next frontiers coexist.

Parallel publication uses narrow locks:

- every child edge has an atomic node id and a lock used only while its first
  worker constructs that child;
- an existing edge uses its published id through the lock-free read path;
- immutable node metadata is fully initialized before the id is published;
- each worker writes its own output column and uses its own contraction
  workspace.

This synchronization belongs to one batched `bornsample!` invocation. The
sampler's public concurrency contract leaves coordination between simultaneous
external invocations to the caller.

### Tangent suffix completions

Before a nonempty tangent batch enters the shared left-to-right scheduler, it
performs one right-to-left sweep of q-resolved quadratic environments:

```math
I=R^\dagger R,\qquad K_q=R^\dagger T_q,\qquad
N_q=T_q^\dagger T_q.
```

These matrices complete the sampled prefix weight without expanding the
``L\times L`` insertion-site cross terms. Each metric stores only its nonzero
residual-sector block pairs, including the possibly charge-shifting pairs of
``K_q``; each transfer visits only compatible channel/metric blocks. The setup
therefore has blockwise cubic cost (worst-case dense equivalent
``O(LQD^3)``) once per batch, not once per shot. Every site completion is then
consumed exactly once by the outer layer scheduler and shared read-only by all
workers. The same-site ``B^\dagger I B`` term enters once; the two ``K_q``
orientations account for distinct-site cross terms without double counting.

The synthetic boundary and global-``q`` roots share the full-chain completion.
It weights the boundary draw and the conditional ``q`` draw, then physical
layers consume completions indexed by local site.

With `disk=false`, pending completions live in batch memory. With `disk=true`,
the first active completion stays in memory and the right sweep serializes the
rest. The scheduler loads one completion before its layer barrier, and that
file is immediately deleted. Active right-environment memory is therefore
constant in chain length; all remaining files and references are removed by
the batch `finally` cleanup. This deterministic completion store is separate
from the probability-ranked prefix cache and has no `maxsize` policy.

## Probability-ranked environment storage

Prefix metadata and branch weights stay in memory for their frontier. Larger
numerical environments are managed separately. With disk storage enabled,
every frontier maintains its own probability-ranked set of at most `maxsize`
resident environments: one factor for an MPS or joint-MPO node, or
one complete branch bank for a traced-MPO node. Other environments in that
frontier are stored as raw TensorMap records.

A new environment enters its frontier's resident set only when its prefix
probability is strictly greater than the current minimum; exact ties stay on
disk. Because the next frontier is ranked independently, its final resident set
is the exact top-`maxsize` subset at that depth. While a layer transition is in
progress, the adjacent frontiers can together hold up to ``2\,\texttt{maxsize}``
resident environments. After the barrier releases the preceding frontier, the
active set is again bounded by `maxsize`.

The source frontier's metadata and resident dictionary remain fixed while its
environments are read or consumed and the destination frontier is constructed.
An admission lock serializes destination dictionary updates and top-set
replacement. Each raw TensorMap record is published through a temporary file
followed by an atomic rename, and immutable space metadata reconstructs a
factor or a traced branch-bank member. Removing the source frontier at
the layer boundary also removes its temporary directory.
