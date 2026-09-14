"""Optimization-quality metrics for CV ZX diagrams.

Measures how much `cvzx.passes.optimize.optimize()` shrinks a diagram, in
terms that matter for circuit complexity rather than raw graph bookkeeping:
generator counts (spiders, surviving compact gates), the non-Gaussian
("non-Clifford"-analogue) resource count, and stage depth. See the dev
guide ("Benchmarking optimization quality") for the rationale behind each
metric and worked examples.

All functions here are read-only analysis, backend-agnostic (`networkx` or
`rustworkx`, see `cvzx.config.Backend`), and deliberately avoid
`cvzx.backends.{nx,rx}.graph.get_proper_nodes()`: that helper's `kind`
filter differs between the two backend modules (nx: `kind == "proper"`
only; rx: `kind in {"proper", "compact"}`), so using it here would silently
change what "generator count" means depending on which backend happens to
be installed. `iter_leaf_attrs()` is the one place that inconsistency is
worked around; every counting function below is built on it.

`count_spiders`/`count_generators` additionally exclude two shapes of pure
bookkeeping leaf (see `is_bookkeeping_leaf`): identity/wiring spiders and
`VoidDiagram` placeholders. This matters more than it sounds like it should
-- confirmed empirically while building this module, not just from reading
docstrings: `optimize()` always hands back `OptimizeResult.diagram` as the
diagram converted from the graph *right before* its own end-of-pipeline
cleanup pass strips exactly these two things (see `optimize()`'s and
`VoidDiagram`'s docstrings). Counting them as "real" leaves means a
diagram's own `normalize_diagram`-inserted row filler and CopyRule's
same-arity dead placeholders get compared against a hand-built "before"
diagram that never had any -- inflating "after" counts enough to make a
diagram that `optimize()` genuinely simplified (fewer gates, ancillas
eliminated) look like it grew.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cvzx.backend import get_backend_modules
from cvzx.backends.nx.graph import to_graph as _nx_to_graph
from cvzx.ir.base import CompositionDiagram, Diagram
from cvzx.passes.normalize import normalize_diagram
from cvzx.utils.helpers import is_wiring_node_from_attrs

if TYPE_CHECKING:
    from collections.abc import Iterator

    from cvzx.backends.nx.graph import CVZXGraph as NxCVZXGraph
    from cvzx.backends.rx.graph import CVZXGraph as RxCVZXGraph
    from cvzx.config import Backend

# Node `kind`s that represent an atomic generator (as opposed to a
# TensorDiagram/CompositionDiagram/ContractedDiagram bookkeeping wrapper,
# `kind == "container"`).
_LEAF_KINDS = frozenset({"proper", "compact"})

# `type`s that are pure rewrite-rule/normalization bookkeeping rather than
# real circuit content -- see `is_bookkeeping_leaf`.
_BOOKKEEPING_TYPES = frozenset({"VoidDiagram"})

# The phase-polynomial degree `optimize()`'s non-Clifford metric defaults
# to treating as "non-Clifford" -- see the dev guide for why this matches
# `CubicPhaseGate`'s own "non_gaussian" tagging rather than the stricter
# degree > 1 boundary `TerminalAbsorptionRule`/`CopyRule` use internally
# for their own, unrelated, exact-identity applicability check.
_DEFAULT_DEGREE_THRESHOLD = 3

# Un-expanded `CompactDiagram` gates never carry a `ZxPoly` phase directly
# (their `phase` graph attribute holds the raw gate parameter instead --
# e.g. `gamma` for `CubicPhaseGate`, `gain` for `ControlledSumGate` -- see
# `_add_proper_node` in `cvzx.backends.nx.graph`), so their phase degree
# can't be read off a `ZxPoly.degree()` call the way a `QSpider`/`PSpider`
# leaf's can. Nor can it be read off the gate's own `spider_type` field
# (would need the original `Diagram` object -- but `to_graph()` deletes
# every node's `"diagram"` attribute before returning, specifically so the
# graph doesn't keep the input diagram's objects alive; confirmed by
# reading `to_graph`'s final loop, not assumed). Instead, map each gate's
# `type` name directly to the phase degree its (fixed-shape) `.expand()`
# decomposition is known to produce. `CubicPhaseGate` is the only gate
# class in `cvzx.ir.gates` tagging itself `spider_type = "non_gaussian"`;
# every other gate (including `ArbitraryGate`, which decomposes into
# rotation/squeezing only) expands to degree <= 2, so has no entry here.
_NON_GAUSSIAN_COMPACT_DEGREE: dict[str, int] = {"CubicPhaseGate": 3}


def iter_leaf_attrs(cvzx_graph: NxCVZXGraph | RxCVZXGraph) -> Iterator[dict[str, Any]]:
    """Yield the attribute dict of every generator (leaf) node in `cvzx_graph`.

    Backend-agnostic: works whether `cvzx_graph.graph` is a `networkx.DiGraph`
    or a `rustworkx.PyDiGraph`, and always uses the `kind in {"proper",
    "compact"}` definition of "leaf" (see the module docstring for why this
    doesn't just delegate to `get_proper_nodes()`).

    Parameters
    ----------
    cvzx_graph : NxCVZXGraph | RxCVZXGraph
        The graph to walk.

    Returns
    -------
    Iterator[dict[str, Any]]
        One attribute dict per leaf node, in no particular order.
    """
    graph = cvzx_graph.graph
    if hasattr(graph, "node_indices"):  # rustworkx PyDiGraph
        all_attrs = (graph[idx] for idx in graph.node_indices())
    else:  # networkx DiGraph
        all_attrs = (attrs for _, attrs in graph.nodes(data=True))  # type: ignore[call-arg]
    return (attrs for attrs in all_attrs if attrs.get("kind") in _LEAF_KINDS)


def is_bookkeeping_leaf(attrs: dict[str, Any]) -> bool:
    """True if `attrs` is rewrite-rule/normalization bookkeeping, not real content.

    Two shapes of bookkeeping leaf get excluded from the "real complexity"
    metrics below: an identity/wiring spider (`is_wiring_node_from_attrs`
    -- filler `normalize_diagram` inserts on every otherwise-untouched row
    of every stage) and a `VoidDiagram` (a same-arity placeholder a rewrite
    rule leaves behind when it eliminates a state/effect elsewhere -- see
    `VoidDiagram`'s own docstring).

    Parameters
    ----------
    attrs : dict[str, Any]
        Node attributes from the graph.

    Returns
    -------
    bool
    """
    return is_wiring_node_from_attrs(attrs) or attrs.get("type") in _BOOKKEEPING_TYPES


def count_spiders(cvzx_graph: NxCVZXGraph | RxCVZXGraph) -> int:
    """Count `QSpider`/`PSpider` leaves in `cvzx_graph`, excluding identity wires.

    Parameters
    ----------
    cvzx_graph : NxCVZXGraph | RxCVZXGraph
        The graph to count leaves in.

    Returns
    -------
    int
    """
    return sum(
        1
        for attrs in iter_leaf_attrs(cvzx_graph)
        if attrs.get("kind") == "proper"
        and attrs.get("type") in {"QSpider", "PSpider"}
        and not is_wiring_node_from_attrs(attrs)
    )


def count_gates(cvzx_graph: NxCVZXGraph | RxCVZXGraph) -> int:
    """Count surviving un-expanded `CompactDiagram` gate leaves in `cvzx_graph`.

    Parameters
    ----------
    cvzx_graph : NxCVZXGraph | RxCVZXGraph
        The graph to count leaves in.

    Returns
    -------
    int
    """
    return sum(1 for attrs in iter_leaf_attrs(cvzx_graph) if attrs.get("kind") == "compact")


def count_generators(cvzx_graph: NxCVZXGraph | RxCVZXGraph) -> int:
    """Count every atomic generator leaf in `cvzx_graph`, excluding bookkeeping.

    Spiders, surviving compact gates, and other proper leaves (`Swap`,
    `Fourier`/`FourierInv`/`Fourier2`) combined -- the headline "how many
    atomic operations remain" count, independent of whether a gate has
    been expanded into spiders yet. Excludes `is_bookkeeping_leaf` leaves
    (identity wires, `VoidDiagram`), which represent eliminated content,
    not surviving content.

    Parameters
    ----------
    cvzx_graph : NxCVZXGraph | RxCVZXGraph
        The graph to count leaves in.

    Returns
    -------
    int
    """
    return sum(1 for attrs in iter_leaf_attrs(cvzx_graph) if not is_bookkeeping_leaf(attrs))


def count_nodes(cvzx_graph: NxCVZXGraph | RxCVZXGraph) -> int:
    """Total node count of `cvzx_graph` (leaves and container wrappers alike).

    A weaker complexity signal than `count_generators`: it also counts
    `TensorDiagram`/`CompositionDiagram`/`ContractedDiagram` bookkeeping
    nodes, so it moves with container-flattening cleanup as well as with
    genuine circuit simplification. Included mainly for parity with
    `CVZXGraph.__repr__`.

    Parameters
    ----------
    cvzx_graph : NxCVZXGraph | RxCVZXGraph
        The graph to count nodes in.

    Returns
    -------
    int
    """
    return len(cvzx_graph)


def count_edges(cvzx_graph: NxCVZXGraph | RxCVZXGraph) -> int:
    """Total edge count of `cvzx_graph`. See `count_nodes` for the same caveat.

    Parameters
    ----------
    cvzx_graph : NxCVZXGraph | RxCVZXGraph
        The graph to count edges in.

    Returns
    -------
    int
    """
    graph = cvzx_graph.graph
    if hasattr(graph, "edge_list"):  # rustworkx PyDiGraph
        return len(graph.edge_list())
    return graph.number_of_edges()  # type: ignore[union-attr, no-any-return]


def count_non_clifford_phases(
    cvzx_graph: NxCVZXGraph | RxCVZXGraph, *, degree_threshold: int = _DEFAULT_DEGREE_THRESHOLD
) -> int:
    """Count leaves whose phase-polynomial degree is >= `degree_threshold`.

    The CV-ZX analogue of qubit ZX calculus's T-count: a `QSpider`/`PSpider`
    leaf's degree is read directly off its `ZxPoly` phase attribute (a real
    `ZxPoly`, untouched by `to_graph()`'s attribute cleanup); an un-expanded
    compact gate's degree is looked up by its `type` name in
    `_NON_GAUSSIAN_COMPACT_DEGREE` (see module docstring for why this reads
    `type` rather than the gate's own `phase` attribute, which for a
    compact gate holds its raw parameter, e.g. `gamma`, not a `ZxPoly`).
    Leaves with no phase at all, or a gate `type` absent from that table
    (`Swap`, `Fourier*`, every Gaussian gate, measurement effects), never
    count, regardless of threshold.

    Parameters
    ----------
    cvzx_graph : NxCVZXGraph | RxCVZXGraph
        The graph to count leaves in.
    degree_threshold : int, optional
        The minimum phase degree to count as non-Clifford. Defaults to 3,
        matching `CubicPhaseGate`'s own "non_gaussian" tagging -- see the
        dev guide for the alternative (`degree_threshold=2`, matching
        `TerminalAbsorptionRule`/`CopyRule`'s stricter internal boundary).

    Returns
    -------
    int
    """
    count = 0
    for attrs in iter_leaf_attrs(cvzx_graph):
        if attrs.get("kind") == "proper":
            phase = attrs.get("phase")
            degree = phase.degree() if phase is not None else None
        else:
            node_type = attrs.get("type")
            degree = _NON_GAUSSIAN_COMPACT_DEGREE.get(node_type) if isinstance(node_type, str) else None
        if degree is not None and bool(degree >= degree_threshold):
            count += 1
    return count


def diagram_depth(diagram: Diagram) -> int | None:
    """Stage count of `diagram`, via `normalize_diagram`'s own stage output.

    Reuses `normalize_diagram()` (`cvzx.passes.normalize`) exactly as its
    public contract already exposes it, rather than reaching into its
    private `stage_records` state: if it returns a `CompositionDiagram`,
    each element is by construction exactly one stage, so the depth is
    `len(result.diagrams)`.

    Parameters
    ----------
    diagram : Diagram
        The diagram to measure. Should be in compact form (not yet passed
        through `expand_two_mode_gates`), matching `normalize_diagram`'s
        own expectation.

    Returns
    -------
    int | None
        The stage count, `0` for a leafless diagram, or `None` if
        `diagram` contains a `ContractedDiagram` -- `normalize_diagram`
        conservatively leaves those unchanged (see its docstring), so no
        stage count is defined for them.

    Notes
    -----
    A diagram built entirely from bare identity wires but *already*
    expressed as a `CompositionDiagram` (rather than, say, a single
    identity leaf) is a degenerate case this can misreport as depth
    `len(diagram.diagrams)` instead of `0`: `normalize_diagram` elides such
    wiring internally and returns the (still-composed) input unchanged,
    which this function can't distinguish from a genuine single-stage
    result without reaching into that private elision logic. Not a
    concern for real circuits, which always have substantive leaves.
    """
    cvzx_graph = _nx_to_graph(diagram)
    graph = cvzx_graph.graph
    if any(attrs.get("container_type") == "contracted" for _, attrs in graph.nodes(data=True)):
        return None
    if not any(attrs.get("kind") in _LEAF_KINDS for _, attrs in graph.nodes(data=True)):
        return 0

    normalized = normalize_diagram(diagram)
    if isinstance(normalized, CompositionDiagram):
        return len(normalized.diagrams)
    return 1


@dataclass(frozen=True)
class DiagramMetrics:
    """A single snapshot of `diagram`'s optimization-quality metrics.

    Attributes
    ----------
    spiders : int
        See `count_spiders`.
    gates : int
        See `count_gates`.
    generators : int
        See `count_generators`.
    nodes : int
        See `count_nodes`.
    edges : int
        See `count_edges`.
    non_clifford_phases : int
        See `count_non_clifford_phases`.
    depth : int | None
        See `diagram_depth`.
    """

    spiders: int
    gates: int
    generators: int
    nodes: int
    edges: int
    non_clifford_phases: int
    depth: int | None


def compute_metrics(
    diagram: Diagram, *, backend: Backend | str | None = None, degree_threshold: int = _DEFAULT_DEGREE_THRESHOLD
) -> DiagramMetrics:
    """Compute every optimization-quality metric for `diagram` in one snapshot.

    Builds the `CVZXGraph` the same way `cvzx.passes.optimize.optimize()`
    does (via `cvzx.backend.get_backend_modules`), so the metrics reflect
    exactly the representation the optimizer itself works on.

    Parameters
    ----------
    diagram : Diagram
        The diagram to measure.
    backend : Backend | str | None, optional
        Which `CVZXGraph` backend to build. `None` (the default) uses
        `cvzx.config.DEFAULT_BACKEND`.
    degree_threshold : int, optional
        Forwarded to `count_non_clifford_phases`.

    Returns
    -------
    DiagramMetrics
    """
    _, graph_mod, _ = get_backend_modules(backend)
    cvzx_graph = graph_mod.to_graph(diagram)
    return DiagramMetrics(
        spiders=count_spiders(cvzx_graph),
        gates=count_gates(cvzx_graph),
        generators=count_generators(cvzx_graph),
        nodes=count_nodes(cvzx_graph),
        edges=count_edges(cvzx_graph),
        non_clifford_phases=count_non_clifford_phases(cvzx_graph, degree_threshold=degree_threshold),
        depth=diagram_depth(diagram),
    )


@dataclass(frozen=True)
class MetricsComparison:
    """A before/after `DiagramMetrics` pair, with ratio/reduction helpers.

    Attributes
    ----------
    before : DiagramMetrics
        Metrics computed on the diagram before optimization.
    after : DiagramMetrics
        Metrics computed on the diagram after optimization. Always
        computed from `OptimizeResult.diagram`, never re-derived from
        `OptimizeResult.graph` -- see `optimize()`'s own docstring for why
        the cleaned graph is not guaranteed losslessly representable as a
        `Diagram` tree.
    """

    before: DiagramMetrics
    after: DiagramMetrics

    def ratio(self, field: str) -> float | None:
        """`after.<field> / before.<field>`, or `None` if undefined.

        Undefined when either side is `None` (e.g. `depth` on a diagram
        containing a `ContractedDiagram`) or `before.<field>` is `0`
        (would divide by zero).

        Parameters
        ----------
        field : str
            The `DiagramMetrics` field name to compare.

        Returns
        -------
        float | None
        """
        before_value = getattr(self.before, field)
        after_value = getattr(self.after, field)
        if before_value is None or after_value is None or before_value == 0:
            return None
        return float(after_value) / float(before_value)

    def reduction(self, field: str) -> float | None:
        """`1 - ratio(field)`, i.e. the fraction `field` shrank by.

        Parameters
        ----------
        field : str
            The `DiagramMetrics` field name to compare.

        Returns
        -------
        float | None
        """
        ratio = self.ratio(field)
        return None if ratio is None else 1 - ratio


def compare_metrics(
    before: Diagram,
    after: Diagram,
    *,
    backend: Backend | str | None = None,
    degree_threshold: int = _DEFAULT_DEGREE_THRESHOLD,
) -> MetricsComparison:
    """Compute and pair up `DiagramMetrics` for `before` and `after`.

    Parameters
    ----------
    before : Diagram
        The diagram before optimization.
    after : Diagram
        The diagram after optimization (see `MetricsComparison.after` for
        why this should be `OptimizeResult.diagram`, not `.graph`).
    backend : Backend | str | None, optional
        Forwarded to `compute_metrics` for both snapshots.
    degree_threshold : int, optional
        Forwarded to `compute_metrics` for both snapshots.

    Returns
    -------
    MetricsComparison
    """
    return MetricsComparison(
        before=compute_metrics(before, backend=backend, degree_threshold=degree_threshold),
        after=compute_metrics(after, backend=backend, degree_threshold=degree_threshold),
    )
