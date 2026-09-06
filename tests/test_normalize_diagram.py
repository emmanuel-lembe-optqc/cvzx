"""Unit tests for `cvzx.normalize_diagram.normalize_diagram`.

`normalize_diagram` has no numeric/semantic evaluator to check its output
against (this library is a pure diagram-rewriting engine, not a state
simulator -- see the module docstring), so "correctness" here is checked
structurally instead, along three invariants that together pin down what
the module docstring actually promises:

    1. Every non-identity leaf present in the input still appears, with
       the same type and parameters, in the output (`_leaf_signatures`) --
       normalizing must not lose, duplicate, or mutate a real gate.
    2. `num_inputs`/`num_outputs` are preserved exactly.
    3. The output's shape genuinely alternates type-2 (one wide leaf plus
       identity filler) and type-1 (all-narrow) stages, in the same
       causal order the input's wide leaves actually depend on each
       other in -- not just "some structure that happens to type-check".

The two bail-out cases (`ContractedDiagram` present, or no leaves at all)
are checked directly via object identity (`is`), since the module
docstring promises the input is returned completely unchanged there.
"""

import unittest

from sympy import pi

from cvzx.base_gates import (
    CompositionDiagram,
    Diagram,
    Fourier,
    PSpider,
    QSpider,
    TensorDiagram,
    VoidDiagram,
    ZxPoly,
)
from cvzx.gates import ControlledSumGate, PhaseRotationGate, SqueezingGate
from cvzx.normalize_diagram import normalize_diagram
from cvzx.nx_graph import to_graph
from cvzx.visualize_base_gates import visualize_before_after

_ZERO = ZxPoly({})


def _identity():  # ruff: ignore[missing-return-type-private-function]
    """A fresh 1-mode identity spider, matching `normalize_diagram`'s own filler."""
    return QSpider(1, 1, _ZERO)


def _is_identity_attrs(attrs: dict) -> bool:
    """True for a graph node representing a plain identity wire.

    Both the input circuits below and `normalize_diagram`'s own filler
    wires use exactly this shape (`QSpider(1, 1, ZxPoly({}))`), so this
    is used to exclude filler wires when comparing "real" gate content
    between an input diagram and its normalized form -- the number of
    identity wires is deliberately *not* part of that contract (see
    module docstring: stages are free to pad with as many filler wires
    as the row bookkeeping needs).
    """
    return (
        attrs.get("type") == "QSpider"
        and attrs.get("num_inputs") == 1
        and attrs.get("num_outputs") == 1
        and attrs.get("phase") == _ZERO
    )


def _leaf_signatures(diagram):  # ruff: ignore[missing-type-function-argument, missing-return-type-private-function]
    """Multiset of every non-identity leaf's (type, shape, params) in `diagram`.

    Built straight from `to_graph()` node attributes rather than from
    the `Diagram` objects themselves, since `normalize_diagram` always
    reconstructs fresh leaf instances (see `_leaf_diagram`) -- comparing
    by `==`/`is` would never match even when the content is identical.

    Returns
    -------
    list[tuple]
        Sorted so two structurally-equal diagrams compare equal
        regardless of leaf ordering.
    """
    graph = to_graph(diagram)
    sigs = []
    for _, attrs in graph.nodes(data=True):
        if attrs.get("kind") not in {"proper", "compact"}:
            continue
        if _is_identity_attrs(attrs):
            continue
        sigs.append((
            attrs.get("type"),
            attrs.get("num_inputs"),
            attrs.get("num_outputs"),
            attrs.get("control"),
            attrs.get("target"),
            repr(attrs.get("phase")),
        ))
    return sorted(sigs, key=repr)


def _wide_leaves_in_stage(stage):  # ruff: ignore[missing-type-function-argument, missing-return-type-private-function]
    """The (0 or 1) "wide" (>1-mode) leaves directly inside one top-level stage."""
    elements = stage.diagrams if isinstance(stage, TensorDiagram) else [stage]
    return [d for d in elements if max(d.num_inputs, d.num_outputs) > 1]


def _stage_kind(stage):  # ruff: ignore[missing-type-function-argument, missing-return-type-private-function]
    """Classify one stage of a normalized `CompositionDiagram` as "wide" or "narrow".

    A well-formed stage from `normalize_diagram` never contains more
    than one wide leaf (that is exactly the type-2/type-1 alternation
    the module exists to establish), so this doubles as a structural
    sanity check: it raises if that invariant is ever violated.

    Raises
    ------
    AssertionError
        If `stage` contains more than one wide leaf.
    """
    wide_leaves = _wide_leaves_in_stage(stage)
    if len(wide_leaves) > 1:
        msg = f"stage unexpectedly contains {len(wide_leaves)} wide leaves: {stage!r}"
        raise AssertionError(msg)
    return "wide" if wide_leaves else "narrow"


def _stage_is_all_identity(stage):  # ruff: ignore[missing-type-function-argument, missing-return-type-private-function]
    """True if every element of one top-level stage is a bare identity wire.

    This is exactly the "empty tensor" the design rule forbids: a
    type-1 stage whose rows are nothing but wiring, carrying no real
    content at all.
    """
    elements = stage.diagrams if isinstance(stage, TensorDiagram) else [stage]
    return all(_is_identity_element(element) for element in elements)


def _is_identity_element(element):  # ruff: ignore[missing-type-function-argument, missing-return-type-private-function]
    """True for a bare `Diagram` leaf that is a plain identity wire.

    Unlike `_is_identity_attrs` (which reads graph-node attributes),
    this reads a `Diagram` object directly -- every element
    `normalize_diagram` ever places in a stage (real gate or filler
    alike) is always a bare leaf, never a further-nested container, so
    this never needs to go through `to_graph`.
    """
    return (
        type(element).__name__ == "QSpider"
        and element.num_inputs == 1
        and element.num_outputs == 1
        and element.phase == _ZERO
    )


def _leaf_signature(element):  # ruff: ignore[missing-type-function-argument, missing-return-type-private-function]
    """(type, shape, params) signature of one bare leaf `Diagram`, read directly."""
    return (
        type(element).__name__,
        element.num_inputs,
        element.num_outputs,
        getattr(element, "control", None),
        getattr(element, "target", None),
        repr(getattr(element, "phase", None)),
    )


def _real_wiring(diagram):  # ruff: ignore[missing-type-function-argument, missing-return-type-private-function]
    """Trace real-leaf-to-real-leaf (and external in/out) wiring, chasing through identity leaves.

    This is the check `_leaf_signatures` deliberately does *not* do: it
    confirms every real gate is present, but says nothing about what
    it's actually wired to. This walks the `Diagram` tree directly and
    recursively -- `TensorDiagram` rows recurse into each element,
    `CompositionDiagram` stages chase forward through `.connectivity`
    from one stage to the next -- so it works equally well on a fresh,
    still-nested input diagram (e.g. a `CompositionDiagram` sitting
    inside one row of a `TensorDiagram`) and on `normalize_diagram`'s
    flat, alternating-stage output, which is exactly what's needed to
    compare the two.

    This deliberately never goes through `to_graph`'s "composition"
    edges: those materialize a *different*, independently-buggy
    convention (`_add_composition_node`'s docstring claims "output port
    of left -> input port of right", but `CompositionDiagram.__post_init__`
    validates the opposite -- keys are input ports of the *later*
    diagram, values are output ports of the *earlier* one, confirmed by
    `__post_init__`'s own range checks and by round-tripping a
    hand-built non-self-inverse permutation through `to_graph`/
    `to_diagram`). Going through the `Diagram`-level `.connectivity`
    contract directly sidesteps that unrelated bug entirely and checks
    exactly what `normalize_diagram` is actually responsible for.

    Each same-signature group of leaves (e.g. two structurally-identical
    `ControlledSumGate`s) is disambiguated by its occurrence order among
    leaves sharing that signature, in the traversal's own natural
    left-to-right/depth-first order -- stable and consistent between an
    input diagram and its normalized form as long as relative order
    among same-signature leaves is preserved, which `normalize_diagram`
    never reorders arbitrarily (see module docstring: causal/positional
    order is preserved throughout).

    Returns
    -------
    frozenset
        `(source, source_port, target, target_port)` tuples, where
        `source`/`target` is `("EXT_IN", j)`, `("EXT_OUT", j)`, or
        `(signature, occurrence_index)` for a real leaf.
    """
    wiring: set = set()
    occurrence_counts: dict = {}

    def tag_for(leaf: Diagram) -> tuple:
        sig = _leaf_signature(leaf)
        idx = occurrence_counts.get(sig, 0)
        occurrence_counts[sig] = idx + 1
        return (sig, idx)

    def trace(node: Diagram, input_sources: list) -> list:
        """Return the list of output-port source descriptions for `node`."""
        if isinstance(node, TensorDiagram):
            outputs = []
            pos = 0
            for element in node.diagrams:
                n_in = element.num_inputs
                outputs.extend(trace(element, input_sources[pos : pos + n_in]))
                pos += n_in
            return outputs
        if isinstance(node, CompositionDiagram):
            active = input_sources
            for i, stage in enumerate(node.diagrams):
                if i > 0:
                    conn = node.connectivity[i - 1]
                    active = [active[conn[k]] for k in range(stage.num_inputs)]
                active = trace(stage, active)
            return active
        # A bare leaf: either an identity wire (pure passthrough, no
        # wiring recorded) or a real gate (record every inbound wire).
        if _is_identity_element(node):
            (source,) = input_sources
            return [source]
        tag = tag_for(node)
        wiring.update((source, 0, tag, p) for p, source in enumerate(input_sources))
        return [(tag, p) for p in range(node.num_outputs)]

    final_outputs = trace(diagram, [("EXT_IN", j) for j in range(diagram.num_inputs)])
    wiring.update((source, 0, ("EXT_OUT", j), 0) for j, source in enumerate(final_outputs))

    return frozenset(wiring)


class TestBailOutCases(unittest.TestCase):
    """The two documented no-op cases: `ContractedDiagram` present, or no leaves."""

    def test_bare_contracted_diagram_is_returned_unchanged(self):
        """A `ContractedDiagram` at the top level is out of scope and untouched.

        `normalize_diagram` is meant to run *before* `expand_two_mode_gates`
        (see module docstring); a `ContractedDiagram` is exactly what that
        expansion produces, so encountering one means normalization already
        ran (or was never applicable), and the function must not attempt to
        re-derive wiring it wasn't designed to trace through.
        """
        expanded = ControlledSumGate(control=2, target=1).expand()
        assert type(expanded).__name__ == "ContractedDiagram"
        result = normalize_diagram(expanded)
        assert result is expanded

    def test_nested_contracted_diagram_is_returned_unchanged(self):
        """The same bail-out fires however deeply the `ContractedDiagram` is nested."""
        expanded = ControlledSumGate(control=2, target=1).expand()
        wrapped = TensorDiagram([expanded, _identity()])
        result = normalize_diagram(wrapped)
        assert result is wrapped

    def test_diagram_with_no_leaves_is_returned_unchanged(self):
        """An empty `TensorDiagram([])` has nothing to normalize."""
        empty = TensorDiagram([])
        result = normalize_diagram(empty)
        assert result is empty


class TestBareLeafDiagrams(unittest.TestCase):
    """The smallest possible inputs: a single wide or narrow leaf."""

    def test_bare_two_mode_gate_is_returned_unwrapped(self):
        """A lone 2-mode gate normalizes to itself, not a padded single-stage composition.

        With no other leaf to share a stage with, there's no identity
        filler to add and only one stage to produce, so the result
        collapses all the way down to the reconstructed gate itself
        (see `normalize_diagram`: `len(stages) == 1` returns `stages[0]`
        directly, and a stage of one element is that element itself,
        not a one-element `TensorDiagram`).
        """
        gate = ControlledSumGate(gain=0.5, control=2, target=1)
        result = normalize_diagram(gate)
        assert isinstance(result, ControlledSumGate)
        assert result.num_inputs == 2
        assert result.num_outputs == 2
        assert result.control == gate.control
        assert result.target == gate.target
        assert _leaf_signatures(result) == _leaf_signatures(gate)

    def test_pure_narrow_chain_stays_a_single_flat_composition(self):
        """A 1-mode gate chain has no wide leaf to split stages around.

        With zero wide leaves, `normalize_diagram` produces exactly one
        type-1 run covering the whole chain -- i.e. the same flat,
        in-order `CompositionDiagram` a hand-written chain would already
        be, just rebuilt from fresh leaf instances.
        """
        chain = CompositionDiagram([
            PhaseRotationGate(pi / 4),
            SqueezingGate(1.5),
            PhaseRotationGate(pi / 3),
        ])
        result = normalize_diagram(chain)
        assert isinstance(result, CompositionDiagram)
        assert result.num_inputs == 1
        assert result.num_outputs == 1
        assert [type(d).__name__ for d in result.diagrams] == [
            "PhaseRotationGate",
            "SqueezingGate",
            "PhaseRotationGate",
        ]
        assert _leaf_signatures(result) == _leaf_signatures(chain)


class TestStageAlternation(unittest.TestCase):
    """Multi-mode circuits: the type-1/type-2 alternation and its causal order."""

    def test_three_mode_mixed_circuit_alternates_narrow_and_wide_stages(self):
        """narrow, wide, wide -- the input's own layering, minus a redundant split.

        The input is built "flat" (four same-width `TensorDiagram`s,
        sequentially composed): a phase rotation on mode 0, a CSUM on
        modes (0, 1), a squeezing gate on mode 2, and a second CSUM on
        modes (1, 2). The squeezing gate has no actual dependency on
        either CSUM (it lives entirely on mode 2, untouched by the first
        CSUM and only read by the second one *afterward*), so the most
        compact correct schedule packs it into the very first stage
        alongside the phase rotation rather than giving it a stage of
        its own in between -- three stages (narrow, wide, wide), not
        five. This is the direct, intended consequence of eliding
        already-input bare identity leaves (`_strip_identity_leaves`)
        before stage assignment runs: with those identity leaves no
        longer cluttering the dependency graph, `_assign_stages` can see
        that mode 2 is genuinely free at time 0 and schedule accordingly
        -- exactly the kind of compaction the "no empty/needlessly-split
        tensors" design rule calls for.
        """
        layer_a = TensorDiagram([PhaseRotationGate(pi / 4), _identity(), _identity()])
        csum1 = ControlledSumGate(control=2, target=1)
        layer_b = TensorDiagram([csum1, _identity()])
        layer_c = TensorDiagram([_identity(), _identity(), SqueezingGate(0.7)])
        csum2 = ControlledSumGate(control=2, target=1)
        layer_d = TensorDiagram([_identity(), csum2])
        circuit = CompositionDiagram([layer_a, layer_b, layer_c, layer_d])

        result = normalize_diagram(circuit)
        assert isinstance(result, CompositionDiagram)
        assert result.num_inputs == 3
        assert result.num_outputs == 3
        assert [_stage_kind(stage) for stage in result.diagrams] == ["narrow", "wide", "wide"]
        wide_stages = [stage for stage in result.diagrams if _stage_kind(stage) == "wide"]
        assert [_wide_leaves_in_stage(s)[0].control for s in wide_stages] == [2, 2]
        assert [_wide_leaves_in_stage(s)[0].target for s in wide_stages] == [1, 1]
        # `connectivity` must connect every one of the 3 stages to its
        # predecessor explicitly (2 boundaries for 3 stages) -- never
        # silently assumed to be the identity map.
        assert set(result.connectivity.keys()) == {0, 1}
        assert _leaf_signatures(result) == _leaf_signatures(circuit)
        assert not any(_stage_is_all_identity(stage) for stage in result.diagrams)
        assert _real_wiring(result) == _real_wiring(circuit)

    def test_ancilla_birth_nested_in_tensor_row_is_normalized_correctly(self):
        """Regression test for a real `connectivity` key/value inversion bug.

        `csum_a` sits inside a `CompositionDiagram` that is itself nested
        inside one row of a wider `TensorDiagram` (`layer1`) -- the exact
        "ancilla prepared mid-circuit" shape that `_build_four_mode_circuit`
        (in `test_optimize.py`) also uses. This is a direct regression
        test for a bug reported against exactly this circuit: the stage
        boundary `connectivity` dict `normalize_diagram` built was keyed
        and valued backwards relative to what
        `CompositionDiagram.__post_init__` actually validates (a
        well-formed-looking bijection that nonetheless wired every
        boundary to the *inverse* of the intended permutation -- silent,
        since a swapped bijection still passes every range check), and
        the ancilla/identity filler rows the input diagram happened to
        carry used to survive as their own pointless all-identity
        stages instead of being cleaned up. `_real_wiring` below is the
        check that actually catches a connectivity inversion (unlike a
        leaf-signature/IO-count check, which a swapped-but-valid
        permutation sails straight through).
        """
        ancilla = PSpider(0, 1, ZxPoly({1: 2}))
        csum_a = ControlledSumGate(control=2, target=1)
        seg_a = CompositionDiagram([TensorDiagram([ancilla, _identity()]), csum_a])
        layer1 = TensorDiagram([seg_a, _identity(), _identity()])
        csum_b = ControlledSumGate(control=2, target=1)
        layer2 = TensorDiagram([_identity(), csum_b, _identity()])
        circuit = CompositionDiagram([layer1, layer2])

        assert circuit.num_inputs == 3
        assert circuit.num_outputs == 4

        result = normalize_diagram(circuit)
        assert isinstance(result, CompositionDiagram)
        assert result.num_inputs == 3
        assert result.num_outputs == 4
        kinds = [_stage_kind(stage) for stage in result.diagrams]
        # csum_a must land in an earlier stage than csum_b (it feeds it),
        # and every other leaf here (the ancilla, the plain identity
        # wires) is narrow -- so the two wide stages must appear in that
        # causal order somewhere in the sequence.
        wide_positions = [i for i, k in enumerate(kinds) if k == "wide"]
        assert len(wide_positions) == 2
        assert wide_positions == sorted(wide_positions)
        assert _leaf_signatures(result) == _leaf_signatures(circuit)
        # No stage may be pure wiring (an "empty tensor") -- the ancilla
        # and its identity partner, and the trailing output-order fix,
        # must be folded into real content rather than surviving as
        # their own inert stage.
        assert not any(_stage_is_all_identity(stage) for stage in result.diagrams)
        # The actual point of this test: every real gate's wiring --
        # including the ancilla feeding `csum_a`'s control leg through a
        # nested `CompositionDiagram`, and `csum_a`'s output eventually
        # reaching `csum_b` -- must match the input's own wiring exactly,
        # not just contain the same multiset of gates.
        assert _real_wiring(result) == _real_wiring(circuit)
        # The ancilla is born on row 0 (it's the first element of
        # `TensorDiagram([ancilla, _identity()])`, itself the first
        # element of `layer1`) and never needs to move again: nothing
        # here ever reorders rows relative to each other, so every
        # stage boundary should be the identity permutation, and
        # `csum_a`'s own stage should carry the ancilla's row (row 0)
        # as one of its two consumed input positions.
        assert all(conn == {k: k for k in range(len(conn))} for conn in result.connectivity.values())
        stage_with_csum_a = result.diagrams[wide_positions[0]]
        assert isinstance(stage_with_csum_a, TensorDiagram)
        assert stage_with_csum_a.diagrams.index(_wide_leaves_in_stage(stage_with_csum_a)[0]) == 0


class TestParallelWideGates(unittest.TestCase):
    """A layer of many mutually-independent 2-mode gates (no causal ordering between them)."""

    def test_twenty_independent_csum_gates_all_survive_normalization(self):
        """20 disjoint-mode-pair CSUM gates: canonical form, one exclusive stage each.

        Wide leaves are always given an entire stage to themselves (see
        module docstring, step 1), even when -- as here -- they have no
        dependency on one another at all: there is no packing of
        independent wide leaves into a shared stage. This also doubles
        as a check that the O(L log L) topological sort and the
        amortized wide-frontier claim (see `_assign_stages`) still
        produce a fully correct, gap-free stage assignment at a size
        where a quadratic implementation would visibly slow down.

        Beyond just "20 wide stages": since every gate sits on its own
        disjoint, already-fixed mode pair with nothing else to reorder
        around, the canonical output keeps gate `k` at row `2*k` in
        *every* stage and never permutes anything -- `TensorDiagram([
        CSUM, id, id, ...]), TensorDiagram([id, id, CSUM, id, ...]),
        ...` -- rather than moving each gate's output block to the
        front of its own stage and compacting every other row after
        it, which used to scramble row order (and so `connectivity`)
        even though there was nothing here that actually needed
        reordering.
        """
        gates = [ControlledSumGate(control=2, target=1) for _ in range(20)]
        circuit = TensorDiagram(gates)

        result = normalize_diagram(circuit)
        assert isinstance(result, CompositionDiagram)
        assert result.num_inputs == 40
        assert result.num_outputs == 40
        assert len(result.diagrams) == 20
        for k, stage in enumerate(result.diagrams):
            wides = _wide_leaves_in_stage(stage)
            assert len(wides) == 1
            assert wides[0].control == 2
            assert wides[0].target == 1
            assert isinstance(stage, TensorDiagram)
            row = stage.diagrams.index(wides[0])
            assert row == 2 * k, f"gate {k} landed on row {row}, expected {2 * k}"
        # Nothing here ever needs reordering -- every boundary should be
        # the plain identity permutation.
        num_ports = circuit.num_inputs
        identity_map = dict.fromkeys(range(num_ports))
        for k in identity_map:
            identity_map[k] = k
        assert all(conn == identity_map for conn in result.connectivity.values())
        assert _leaf_signatures(result) == _leaf_signatures(circuit)
        assert _real_wiring(result) == _real_wiring(circuit)


class TestSemanticContentIsPreserved(unittest.TestCase):
    """Cross-cutting checks that hold for every circuit shape above.

    `normalize_diagram` has no numeric backend to check true semantic
    equivalence against (see module docstring), so these lean on the
    two invariants that are actually checkable: the multiset of real
    (non-identity) leaves, and the input/output wire counts.
    """

    def _circuits(self):  # ruff: ignore[missing-return-type-private-function]
        """A handful of representative diagrams, rebuilt fresh each call.

        Fresh instances matter here: several of these leaves carry an
        auto-generated id, and reusing one instance across two separate
        `normalize_diagram` calls in the same test would make the two
        results share ids in a way a real caller never would.
        """
        chain = CompositionDiagram([PhaseRotationGate(pi / 4), SqueezingGate(1.5)])
        bare_gate = ControlledSumGate(gain=0.5, control=2, target=1)
        ancilla = PSpider(0, 1, ZxPoly({1: 2}))
        seg_a = CompositionDiagram([
            TensorDiagram([ancilla, _identity()]),
            ControlledSumGate(control=2, target=1),
        ])
        layer1 = TensorDiagram([seg_a, _identity(), _identity()])
        layer2 = TensorDiagram([_identity(), ControlledSumGate(control=2, target=1), _identity()])
        ancilla_circuit = CompositionDiagram([layer1, layer2])
        void_containing = TensorDiagram([VoidDiagram(1, 1), Fourier()])
        return [chain, bare_gate, ancilla_circuit, void_containing]

    def test_leaf_signatures_and_io_counts_are_preserved(self):
        """Every representative circuit keeps its real gates and its wire counts."""
        for circuit in self._circuits():
            result = normalize_diagram(circuit)
            assert result.num_inputs == circuit.num_inputs, circuit
            assert result.num_outputs == circuit.num_outputs, circuit
            assert _leaf_signatures(result) == _leaf_signatures(circuit), circuit

    def test_normalizing_twice_does_not_lose_or_duplicate_content(self):
        """Re-normalizing an already-normalized diagram stays content-preserving.

        This deliberately does *not* assert the two results have the same
        number of stages: a second pass can legitimately append its own
        trailing output-order permutation fix even when the first pass's
        natural order didn't need one (or vice versa), which is a
        structural wash, not a content change -- so the only thing
        actually promised (and checked here) is that no real gate is
        lost, duplicated, or mutated, and the wire counts stay put.
        """
        for circuit in self._circuits():
            once = normalize_diagram(circuit)
            twice = normalize_diagram(once)
            assert twice.num_inputs == circuit.num_inputs, circuit
            assert twice.num_outputs == circuit.num_outputs, circuit
            assert _leaf_signatures(twice) == _leaf_signatures(circuit), circuit


def _visualize_normalize_diagram_examples() -> None:
    """Render before/after panels for a few representative circuits.

    Script-mode only: this is never executed by pytest (pytest collects
    the `TestCase` classes above directly and never runs the
    `if __name__ == "__main__":` guard below), so importing or collecting
    this module has no visualization side effects. Saves PNGs under
    `test_images_normalize_diagram/`, one `{name}.png` per example,
    mirroring `visualize_four_mode.py`'s convention for the `optimize()`
    pipeline (each PNG has the original circuit on the left and
    `normalize_diagram`'s output on the right). Intentionally left
    unopened/uninspected here -- run `python3 tests/test_normalize_diagram.py`
    and look at the PNGs directly to review them.
    """
    examples: list[tuple[str, Diagram]] = [
        ("bare_two_mode_gate", ControlledSumGate(gain=0.5, control=2, target=1)),
        (
            "pure_narrow_chain",
            CompositionDiagram([
                PhaseRotationGate(pi / 4),
                SqueezingGate(1.5),
                PhaseRotationGate(pi / 3),
            ]),
        ),
    ]

    layer_a = TensorDiagram([PhaseRotationGate(pi / 4), _identity(), _identity()])
    csum1 = ControlledSumGate(control=2, target=1)
    layer_b = TensorDiagram([csum1, _identity()])
    layer_c = TensorDiagram([_identity(), _identity(), SqueezingGate(0.7)])
    csum2 = ControlledSumGate(control=2, target=1)
    layer_d = TensorDiagram([_identity(), csum2])
    examples.append((
        "three_mode_mixed_circuit",
        CompositionDiagram([layer_a, layer_b, layer_c, layer_d]),
    ))

    ancilla = PSpider(0, 1, ZxPoly({1: 2}))
    csum_a = ControlledSumGate(control=2, target=1)
    seg_a = CompositionDiagram([TensorDiagram([ancilla, _identity()]), csum_a])
    layer1 = TensorDiagram([seg_a, _identity(), _identity()])
    csum_b = ControlledSumGate(control=2, target=1)
    layer2 = TensorDiagram([_identity(), csum_b, _identity()])
    examples.append((
        "ancilla_birth_nested_in_tensor_row",
        CompositionDiagram([layer1, layer2]),
    ))

    gates20 = [ControlledSumGate(control=2, target=1) for _ in range(20)]
    examples.append(("twenty_independent_csum_gates", TensorDiagram(gates20)))

    for name, circuit in examples:
        visualize_before_after(circuit, normalize_diagram(circuit), name, "normalize_diagram")

    print(f"wrote {len(examples)} before/after panel(s) to test_images_normalize_diagram/")


if __name__ == "__main__":
    _visualize_normalize_diagram_examples()
    unittest.main()
