# Theory: the CV ZX calculus

`cvzx` implements the continuous-variable ZX calculus proposed in [1]. This page summarizes
the parts of that calculus the library actually implements, and gives an explicit
translation table between the paper's notation and `cvzx`'s Python API. It assumes you have
skimmed the paper (or at least its Table I, Table II, and Section IV.A) — this is a map
between the two, not a restatement of the physics.

## Diagrams and generators

A CV ZX diagram is built from two "spiders" (Table II of the paper):

- the **q-spider**, labelled with a real polynomial phase function $f(x)$, representing
  $\int_{\mathbb{R}} \mathrm{d}s\, e^{if(s)} \lvert s\ldots s\rangle_{q^m}\,{}_{q^n}\langle s\ldots s\rvert$;
- the **p-spider**, its Fourier dual, representing the same construction in the $p$ basis
  with a $-i$ phase convention.

together with four fixed generators: `Swap`, `Fourier` ($\hat F$), `FourierInv` ($\hat
F^\dagger$), and `Fourier2` ($\hat F^2$).

| Paper | `cvzx.base_gates` |
|---|---|
| $q$-spider $f(x)$, arity $m \leftarrow n$ | `QSpider(num_inputs, num_outputs, phase)` |
| $p$-spider $f(x)$, arity $m \leftarrow n$ | `PSpider(num_inputs, num_outputs, phase)` |
| Real polynomial phase function $f(x)$ | `ZxPoly` (a `sympy.Poly` subclass; `ZxPoly({1: 2, 2: 3})` is $2x + 3x^2$) |
| Swap | `Swap()` |
| $\hat F$ / $\hat F^\dagger$ / $\hat F^2$ | `Fourier()` / `FourierInv()` / `Fourier2()` |
| Parallelization $D_1 \otimes D_2$ | `TensorDiagram([d1, d2])` |
| Composition $D_2 \circ D_1$ | `CompositionDiagram([d1, d2])` (composed in **left-to-right, causal** order — see note below) |
| Diagram contraction (Eq. 51) | `ContractedDiagram(first, second, I1, I2, J1, J2)` |
| Blank spider (zero phase function) | `QSpider(n, m, ZxPoly({}))` — this is the identity wire when $n = m = 1$ |

The paper draws diagrams right-to-left to match bra-ket notation (Def. 7). `cvzx`'s
`CompositionDiagram([d1, d2, ...])` instead lists elements in **left-to-right causal
order** — `d1` first, `d2` second, and so on — matching how you'd read a circuit
diagram or write down a sequence of gates. The underlying operator composition is
identical; only the list order convention differs.

The Diagram contraction (Eq. 51) reads as follows $ \int ds_{\bar{i}} ds_{\bar{j}} {}_{q_{\bar{i}}}\langle s_{\bar{i}} | [D_1] | s_{\bar{j}} \rangle_{q_{\bar{j}}} \otimes {}_{q_{\bar{j}}}  \langle s_{\bar{j}} | [D_2] | s_{\bar{i}} \rangle_{q_{\bar{i}}}$

Here, $\bar{i} = i_1, ..., i_n$ and $\bar{j} = j_1, ..., j_m$ are variables representing multiple copies of the symbol being subscripted, one for each item in the respective list. Therefore in `cvzx.base_gates`, it is represented using `ContractedDiagram(first, second, I1, I2, J1, J2)` where `first` and `second` represents $D1$ and $D2$. Notice that we have 4 sets of indices ($I1, I2, J1, J2$). This allows to specify which inputs of $D1$ are contracted with outputs of $D2$ and the other way around. This diagram contraction can also be performed using the `partial_trace` method of a `TensorDiagram` by specifying which diagrams of the tensor product and which indices will be traced out. For visualization purposes we restricted this method for successive diagrams in a tensor product.

`cvzx.gates.CompactDiagram` (and every gate class built on it: `DisplacementGate`,
`PhaseRotationGate`, ...) is not a paper concept — it is an engineering layer that stores a
gate as a single opaque leaf plus a lazily-attached `decomposition` into the generators
above, so that a circuit can be built and manipulated at the gate level and only expanded
into spiders when a rewrite rule actually needs to see inside it (`CompactDiagram.expand()`,
`cvzx.gates.expand_all`). This can help to wrap a sub-circuit into a single object.

## Gate decompositions (Table I / Section III C)

Every gate in the paper's Table I has a corresponding `cvzx.gates` class, whose
`.expand()` builds exactly the spider decomposition given in Section III C:

| Paper gate | Decomposition (paper Eq.) | `cvzx.gates` class |
|---|---|---|
| $\hat D(\alpha)$ displacement | Eq. (57): $q$-spider $\otimes$ $p$-spider | `DisplacementGate(alpha)` |
| $\hat R(\theta)$ phase rotation | Eq. (58): three quadratic spiders (or `Fourier2` at $\theta = (2n{+}1)\pi$) | `PhaseRotationGate(theta)` |
| $\hat S(r)$ / $\widehat{Sq}(\tau)$ squeezing | Eq. (59): four quadratic spiders, $\tau = e^{-r}$ | `SqueezingGate(tau)` |
| $\widehat{CS}_{1,2}(g)$ controlled-sum | Eq. (61)/(62): $q$-/$p$-spider pair, squeezed if $g \neq 1$ | `ControlledSumGate(gain, control, target)` |
| $\widehat{CZ}(g)$ controlled-Z | Eq. (63)/(64): CSUM conjugated by `Fourier` | `ControlledZGate(gain)` |
| $\widehat{BS}(\theta)$ beamsplitter | Eq. (65)/(66): five squeezing/CSUM factors | `BeamsplitterGate(theta)` |
| $\widehat{CPG}(\gamma)$ cubic phase | Eq. (68): a single cubic-phase $q$-spider | `CubicPhaseGate(gamma)` |

`cvzx.gates` also defines a few gates that are convenient building blocks in the codebase
but are not named directly in the paper's Table I — `ShearXInvariantGate`/`ShearPInvariantGate`
(the individual quadratic shears the phase-rotation/squeezing decompositions are built
from), `Squeezing45Gate` (squeezing at a $45°$ angle, Eq. (67)'s balanced-beamsplitter
building block), `ArbitraryGate` ($R(\alpha)S(\lambda)R(\beta)$, the general one-mode
Gaussian form of Theorem 2 / Table III), `TwoModeShearGate`, and `MeasurementGate` (a
homodyne effect — see its docstring for the derivation). Their naming mirrors MQC3's
`graph.ops`/`circuit.ops` module, since a `cvzx` diagram is meant to
compile onto that machinery.

## Rewrite rules (Section IV A)

`cvzx.nx_rewrite_rules` implements a `RewriteRule` per paper rule (or per closely-related
group of rules), operating on the `networkx` graph form rather than the `Diagram` tree
directly — see {doc}`../dev_guide/rewrite_engine` for why. `cvzx.optimize.optimize` runs
them to a fixed point (see {doc}`../user_guide/optimization`).

| Paper rule | `cvzx.nx_rewrite_rules` class | Notes |
|---|---|---|
| Identity ($id$), Eq. (69) | `IdentityRule` | Removes a bare $QSpider(1,1,0)$/$PSpider(1,1,0)$ splice. |
| Fusion ($f$), Eq. (70)–(71) | `FusionRule` | Merges two directly-adjacent same-color spiders into one, summing phases. |
| Associativity theorems, Appendix B | `ChainReductionRule` | Collapses a same-*gate-type* chain (`R`, `BS`, `Sq`, `D`, `F`/`Finv`/`F2`, `CSUM`, `CZ`) into one gate using the paper's Theorem 3/4-style composition laws, without expanding to spiders first. |
| Fourier ($F$), Eq. (74)–(76), and the rotation self-loop identities, Eq. (164)–(165) | `FourierNormalizationRule` | Folds an `F`/`Finv`/`F2` adjacent to a rotation or squeezing gate into the other gate's parameter. |
| Displacement ($d$), squeezing ($s$) | `TerminalAbsorptionRule` | Folds a rotation (always) or, under `assume_infinite_squeezing=True`, a squeezing gate or an opposite-color raw spider ("cross-color discard") into an adjacent terminal state/effect. |
| Copy ($c$) Eq. (77)–(78) and Quadratic copy ($qc$) Eq. (107)–(108) | `CopyRule` | Copies a $(0,1)$/$(1,0)$ terminal through an adjacent wide spider into $n$ copies. Requires the copied spider's phase to be in $\mathbb{R}_1[X]$ (`CopyRule.is_in_R1`), per the paper's constraint. |

`assume_infinite_squeezing` (a keyword on `TerminalAbsorptionRule` and `optimize()`) tracks
exactly the paper's caveat in Section VI.A: squeezing absorption and the copy rule are only
exact when the terminal states involved are treated as idealized, infinitely-squeezed
eigenstates ($\lvert 0\rangle_q$, $\lvert 0\rangle_p$), not physical finite-squeezing
approximations of them. With it left `False` (the default), `optimize()` only applies
rewrites that are exact for any physical state.

## What isn't implemented

- **Completeness beyond 1-mode Gaussian diagrams.** The paper only proves completeness for
  chains of 1-mode quadratic spiders (Theorem 2); `cvzx`'s rewrite rules are a *sound* but
  not necessarily *complete* set for anything beyond that (Section IV.C of the paper).
- **Bialgebra as a multi-mode structural rewrite.** `cvzx` never restructures a
  `TensorDiagram`/`CompositionDiagram` via the bialgebra law directly; it only relies on
  bialgebra-derived identities (fusion associativity, chain-reduction algebra) that were
  already proven sound in the paper.
- **Quadratic ($q$), inversion ($inv$), antipode ($a$), rotation self-loop.** In this first
version of the compiler, we focused on rules which were valuable in optimizing diagrams.
- **GKP diagrams, non-unitary/complex-phase spiders, mixed states** (Sections VI.B) are out
  of scope for this codebase.

## Bibliography

[1] Hironari Nagayoshi, Warit Asavanant, Ryuhoh Ide, Kosuke Fukui, Atsushi Sakaguchi, Jun-ichi Yoshikawa, Nicolas C. Menicucci, and  Akira Furusawa, "ZX Graphical Calculus for Continuous-Variable Quantum Processes" Physical Review Research, vol. 7, no. 3, Aug. 2025, p. 033141. APS Physics, [doi:10.1103/PhysRevResearch.7.033141](https://doi.org/10.1103/PhysRevResearch.7.033141).