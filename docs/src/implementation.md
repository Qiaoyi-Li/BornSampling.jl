# [Implementation](@id implementation)

Bornsampling is organized around one left-to-right factor propagation.  The
probability model determines the sampling mode, while compiled local plans
provide the symmetry-aware contractions used by every mode.

## Probability model

### Pure MPS amplitudes

For an MPS, each local tensor has the canonical order

```math
A_i[\ell_i,x_i,r_i],
```

and the contracted network defines an amplitude ``\psi(\boldsymbol{x})``.
Bornsampling draws the physical configuration from

```math
p(\boldsymbol{x})
= \frac{|\psi(\boldsymbol{x})|^2}{\langle\psi|\psi\rangle}.
```

### Purification amplitudes

An MPO is interpreted as a pure amplitude on a physical space and a
purification space.  A rank-four local tensor has the order

```math
X_i[\ell_i,x_i,y_i,r_i],
```

and a rank-three local tensor inside the same MPO supplies the synthetic value
``y_i=1``.  The complete network defines ``X(\boldsymbol{x},\boldsymbol{y})``
and hence the joint distribution

```math
p(\boldsymbol{x},\boldsymbol{y})
= \frac{|X(\boldsymbol{x},\boldsymbol{y})|^2}
        {\langle X|X\rangle}.
```

The two MPO modes follow directly from this distribution:

```math
\begin{aligned}
\text{traced mode:}\quad
p(\boldsymbol{x})
  &= \sum_{\boldsymbol{y}}p(\boldsymbol{x},\boldsymbol{y}),\\
\text{joint mode:}\quad
(\boldsymbol{x},\boldsymbol{y})
  &\sim p(\boldsymbol{x},\boldsymbol{y}).
\end{aligned}
```

Thus discarding ``\boldsymbol{y}`` from joint samples also produces samples
distributed according to the exact physical marginal ``p(\boldsymbol{x})``.

### Sequential probabilities

At site ``i``, let ``a_i`` denote the sampled local outcome: ``x_i`` in MPS
and traced-MPO modes, or ``(x_i,y_i)`` in joint-MPO mode.  The contraction
produces nonnegative branch weights ``q_i(a_i)`` and

```math
z_i = \sum_{a_i}q_i(a_i), \qquad
p(a_i\mid a_1,\ldots,a_{i-1}) = \frac{q_i(a_i)}{z_i}.
```

The returned log probability is accumulated directly from conditional log
weights,

```math
\log p(a_1,\ldots,a_L)
= \sum_{i=1}^{L}\left[\log q_i(a_i)-\log z_i\right].
```

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
= \sum_{\boldsymbol{y}}p(\boldsymbol{x}_0,\boldsymbol{y})
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

After canonicalizing the state at site 1, the sampler needs only the collapsed
left environment.  It stores that environment as a factor

```math
\rho_i = C_i C_i^\dagger,
\qquad
\operatorname{tr}(C_i C_i^\dagger)=\lVert C_i\rVert_F^2=1.
```

The initial pure boundary gives a factor with one auxiliary column.  For a
fixed local physical and purification basis value, define

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

### Pure and joint modes

An MPS has ``d_y=1`` and begins with one factor column, so the update remains a
single ket column throughout the chain.  Joint MPO sampling also selects one
specific pair ``(x,y)`` at each site; its update is

```math
C_{i+1}=\frac{K_i(x,y)C_i}{\sqrt{q_i(x,y)}},
```

and likewise remains a pure, rank-one factor.  These modes therefore inherit
the usual dense-equivalent ``O(D^2)`` sequential contraction, while tracing a
purification can raise the factor rank to ``D`` and gives ``O(D^3)`` worst-case
work.  In every mode the environment remains represented lazily by
``C_iC_i^\dagger``.

## Global modes and local tensor ranks

The outer `Bornsampling.FiniteMPS` concrete type chooses the global semantics:

| input and constructor | sampled outcome | factor update |
|:--|:--|:--|
| `Bornsampling.FiniteMPS.MPS` | ``x_i`` | pure rank-one propagation |
| `Bornsampling.FiniteMPS.MPO`, `purified=true` | ``x_i`` | purification trace and exact factor compression |
| `Bornsampling.FiniteMPS.MPO`, `purified=false` | ``(x_i,y_i)`` | joint rank-one propagation |

Construction checks once that every tensor in an MPS is rank three.  An MPO
may mix rank-three and rank-four sites.  At a rank-three MPO site, the local
purification basis is the singleton ``y_i=1``; a traced MPO still carries any
factor rank accumulated at earlier rank-four sites.  This division keeps the
probability model global while local static rank controls only leg access and
the contraction kernel.

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
`Bornsampling.TK.FusionStyle(sector_type)`:

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

## Batched prefix-tree reuse

A batched call builds a tree of sampled prefixes. A node stores its prefix log
probability, next-site branch weights, child slots, and TensorMap space
metadata for its numerical environment. MPS and joint-MPO nodes own one
normalized rank-one factor. Traced-MPO nodes own the complete uncompressed
``G_x`` bank produced by their weight pass. When two shots reach the same
prefix, they reuse the branch weights and the corresponding environment
payload. The requested shot count bounds the tree by
``1+N\max(L-1,0)`` nodes.

Workers receive independent numerical workspaces and per-shot RNGs.  Shot
seeds are drawn from the caller's RNG before tasks start, making results stable
under task scheduling.  `ntasks` controls the number of Julia tasks up to the
number of shots; Julia schedules those tasks over the available threads.

Parallel publication uses narrow locks:

- every child edge has an atomic node id and a lock used only while its first
  worker constructs that child;
- an existing edge uses its published id through the lock-free read path;
- immutable node metadata is fully initialized before the id is published;
- each worker writes its own output column and uses its own contraction
  workspace.

This synchronization belongs to one batched `bornsample!` invocation.  The
sampler's public concurrency contract leaves coordination between simultaneous
external invocations to the caller.

## Probability-ranked environment storage

Prefix metadata and branch weights remain resident for the whole batch. The
larger numerical environments are managed separately. With disk storage
enabled, `maxsize` node environments form a probability-ranked resident set:
one normalized factor for an MPS or joint-MPO node, or one complete branch bank
for a traced-MPO node. The batch's temporary directory holds the other node
environments as raw TensorMap records.

A new node environment enters the resident set only when its prefix
probability is strictly greater than the current minimum. Once the resident
set is full, that minimum is monotone nondecreasing. Moreover,

```math
p(\text{child prefix})
=p(\text{parent prefix})p(a_i\mid\text{parent})
\le p(\text{parent prefix}).
```

Therefore descendants of a cold parent go directly to disk, and a loaded cold
environment retains its cold classification. This keeps admission work
proportional to newly competitive prefixes.

The cache publishes each raw TensorMap record through a temporary file followed
by an atomic rename. Immutable space metadata in the node reconstructs a
rank-one factor or every available member of a traced branch bank. Dictionary
access uses a short resident lock, while a separate admission lock serializes
top-set replacement. An environment selected for eviction remains readable in
memory through publication of its complete disk representation; the resident
dictionary then advances to the new top set. The batch owns the temporary
directory for its full lifetime.
