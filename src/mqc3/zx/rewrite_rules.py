"""CV ZX calculus rewrite rules and applications.

This module implements the 10 basic rewrite rules from [3] Sec. IV.A,
the derived rules from Sec. IV.B, and the applications from Sec. V.

References:
-----------
[3] Nagayoshi et al., CV ZX calculus, 2024
"""

import operator

from mqc3.zx.base_gates import (
    CompositionDiagram,
    ContractedDiagram,
    Diagram,
    Fourier,
    Fourier2,
    FourierInv,
    PSpider,
    QSpider,
    TensorDiagram,
)
from mqc3.zx.gates import BeamsplitterGate, DisplacementGate, PhaseRotationGate, SqueezingGate
from mqc3.zx.visualize_base_gates import ZxPoly, is_wiring_diagram

# =============================================================================
# Rewrite Rules (Section IV.A)
# =============================================================================


class RewriteRule:
    """Base class for rewrite rules."""

    def match(self, diagram: Diagram) -> list[tuple]:
        """Find all matches of the rule pattern in the diagram."""
        raise NotImplementedError

    def apply_single(self, diagram: Diagram, match: tuple) -> Diagram:
        """Apply the rule to a specific match."""
        raise NotImplementedError


class IdentityRule(RewriteRule):
    r"""Identity rule (id) from [3] Eq. (69).

    A q-spider with no phase and 1 input/1 output is the identity.
    A p-spider with no phase and 1 input/1 output is also the identity.

    The rule only applies when the identity spider is inside a CompositionDiagram.
    If the identity spider is directly inside a TensorDiagram or ContractedDiagram,
    nothing is done.
    """

    def match(self, diagram: Diagram, path: list[int] | None = None, parent: Diagram | None = None) -> list[list[int]]:
        """Find identity spiders where the parent is a CompositionDiagram.

        Parameters:
        ----------
        diagram : Diagram
            Input Diagram to search.
        path : list[int] | None
            Current path in the nested structure.
        parent : Diagram | None
            Parent of the current diagram.

        Returns:
        -------
        list[list[int]]
            List of paths to identity spiders whose parent is a CompositionDiagram.
        """
        if path is None:
            path = []

        matches = []

        # Check if current diagram is a wiring diagram and parent is CompositionDiagram
        if is_wiring_diagram(diagram) and isinstance(parent, CompositionDiagram):
            matches.append(path.copy())
            return matches

        # Recursive cases - pass self as parent to children
        if isinstance(diagram, (CompositionDiagram, TensorDiagram)):
            for i, sub_diagram in enumerate(diagram.diagrams):
                new_path = [*path, i]
                matches.extend(self.match(sub_diagram, new_path, diagram))

        elif isinstance(diagram, ContractedDiagram):
            # Check first diagram (D1)
            new_path_first = [*path, 0]
            matches.extend(self.match(diagram.first, new_path_first, diagram))

            # Check second diagram (D2)
            new_path_second = [*path, 1]
            matches.extend(self.match(diagram.second, new_path_second, diagram))

        return matches

    def apply_single(self, diagram: Diagram, match_path: list[int]) -> Diagram:
        """Remove an identity spider at the given path.

        The identity spider is only removed if its immediate parent is a
        CompositionDiagram. The result is flattened if the CompositionDiagram
        has one element. ContractedDiagrams are recursively reduced.

        Parameters:
        ----------
        diagram : Diagram
            The diagram to modify.
        match_path : list[int]
            Path to the identity spider to remove.

        Returns:
        -------
        Diagram
            New diagram with the identity spiders removed if applicable,
            or the original diagram if not.
        """
        if not match_path:
            return diagram

        # Navigate to the parent of the identity spider
        parent = diagram
        parent_path = match_path[:-1]
        last_index = match_path[-1]

        for idx in parent_path:
            if isinstance(parent, (CompositionDiagram, TensorDiagram)):
                parent = parent.diagrams[idx]
            elif isinstance(parent, ContractedDiagram):
                parent = parent.first if idx == 0 else parent.second
            else:
                return diagram

        # Check if the parent is a CompositionDiagram
        if not isinstance(parent, CompositionDiagram):
            return diagram

        # Remove the identity spider from the CompositionDiagram
        diagrams = list(parent.diagrams)
        # We don't remove the identy spider if it's the only diagram
        # of the composition
        if len(diagrams) > 1:
            diagrams.pop(last_index)

            # Flatten: return the single element directly
            new_parent = diagrams[0] if len(diagrams) == 1 else CompositionDiagram(diagrams)

            # Reconstruct the diagram along the path
            return self._rebuild_diagram(diagram, parent_path, new_parent)
        return diagram

    def apply_rule(self, diagram: Diagram) -> Diagram:
        """Apply the identity rule to the input diagram.

        Parameters:
        ----------
        diagram : Diagram
            The diagram to modify.

        Returns:
        -------
        Diagram
            New diagram with the identity spiders removed if applicable,
            or the original diagram if not.
        """
        result = diagram
        matches = self.match(diagram)
        if not matches:
            return diagram

        # Group matches by depth
        groups = self._group_by_depth(matches)

        result = diagram

        # Process from deepest to shallowest
        for depth in sorted(groups.keys(), reverse=True):
            paths = groups[depth]

            # Sort by last index in descending order
            # This ensures removing higher indices doesn't affect lower ones
            paths.sort(key=operator.itemgetter(-1), reverse=True)

            for path in paths:
                result = self.apply_single(result, path)

        return result

    def _group_by_depth(self, matches: list[list[int]]) -> dict[int, list[list[int]]]:
        """Group matches by depth (length of the path).

        Parameters:
        ----------
        matches : list[list[int]]
            List of paths from match().

        Returns:
        -------
        dict[int, list[list[int]]]
            Dictionary mapping depth to list of paths.
        """
        groups = {}
        for path in matches:
            depth = len(path)
            if depth not in groups:
                groups[depth] = []
            groups[depth].append(path)
        return groups

    def _flatten_composition(self, diagram: CompositionDiagram) -> Diagram:
        """Flatten a CompositionDiagram with one element.

        Parameters:
        ----------
        diagram : CompositionDiagram
            Composition diagram to flatten.

        Returns:
        -------
        CompositionDiagram
            Flattened CompositionDiagram.
        """
        if len(diagram.diagrams) == 1:
            return diagram.diagrams[0]
        return diagram

    def _reduce_contracted(self, diagram: ContractedDiagram) -> ContractedDiagram:
        """Recursively reduce first and second diagrams of a ContractedDiagram.

        Parameters:
        ----------
        diagram : ContractedDiagram
            Contracted diagram to flatten. It is either the first and/or the
            second diagram of the ContractedDiagram which are flattened.

        Returns:
        -------
        ContractedDiagram
            Flattened ContractedDiagram.
        """
        first = diagram.first
        second = diagram.second

        # Recursively reduce first and second if they are CompositionDiagrams
        if isinstance(first, CompositionDiagram):
            first = self._flatten_composition(first)
        if isinstance(second, CompositionDiagram):
            second = self._flatten_composition(second)

        # If either changed, create a new ContractedDiagram
        if first is not diagram.first or second is not diagram.second:
            return ContractedDiagram(
                first=first,
                second=second,
                I1=diagram.I1,
                I2=diagram.I2,
                J1=diagram.J1,
                J2=diagram.J2,
            )
        return diagram

    def _rebuild_diagram(self, original: Diagram, path: list[int], replacement: Diagram) -> Diagram:
        """Wrapper of _rebuild_recursive function.

        Parameters:
        ----------
        diagram : Diagram
            The diagram to modify.
        path : list[int]
            Path to the location where to replace the current diagram by the
            reduced diagram.
        replacement: Diagram
            Diagram reduced after applying the identity rule. It must be replace
            its former version.

        Returns:
        -------
        Diagram
            New diagram with the identity spiders removed.
        """
        return self._rebuild_recursive(original, path, 0, replacement)

    def _rebuild_recursive(
        self,
        diagram: Diagram,
        path: list[int],
        depth: int,
        replacement: Diagram,
    ) -> Diagram:
        """Recursively reconstruct the diagram along the path.

        Parameters:
        ----------
        diagram : Diagram
            The diagram to modify.
        path : list[int]
            Path to the location where to replace the current diagram by the
            reduced diagram.
        depth: int
            Current depth in the exploration of the diagram. The function must
            go precisely where the diagram to replace is found.
        replacement: Diagram
            Diagram reduced after applying the identity rule. It must be replace
            its former version.

        Returns:
        -------
        Diagram
            New diagram with the identity spiders removed.
        """
        if depth >= len(path):
            return replacement

        idx = path[depth]

        if isinstance(diagram, CompositionDiagram):
            diagrams = list(diagram.diagrams)
            new_sub = self._rebuild_recursive(diagrams[idx], path, depth + 1, replacement)
            diagrams[idx] = new_sub
            result = CompositionDiagram(diagrams)
            return self._flatten_composition(result)

        if isinstance(diagram, TensorDiagram):
            diagrams = list(diagram.diagrams)
            new_sub = self._rebuild_recursive(diagrams[idx], path, depth + 1, replacement)
            diagrams[idx] = new_sub
            return TensorDiagram(diagrams)

        if isinstance(diagram, ContractedDiagram):
            if idx == 0:
                new_first = self._rebuild_recursive(diagram.first, path, depth + 1, replacement)
                # If new_first is a CompositionDiagram with one element, flatten it
                if isinstance(new_first, CompositionDiagram):
                    new_first = self._flatten_composition(new_first)
                result = ContractedDiagram(
                    first=new_first,
                    second=diagram.second,
                    I1=diagram.I1,
                    I2=diagram.I2,
                    J1=diagram.J1,
                    J2=diagram.J2,
                )
            else:
                new_second = self._rebuild_recursive(diagram.second, path, depth + 1, replacement)
                if isinstance(new_second, CompositionDiagram):
                    new_second = self._flatten_composition(new_second)
                result = ContractedDiagram(
                    first=diagram.first,
                    second=new_second,
                    I1=diagram.I1,
                    I2=diagram.I2,
                    J1=diagram.J1,
                    J2=diagram.J2,
                )
            # Recursively reduce the ContractedDiagram
            return self._reduce_contracted(result)

        return diagram


class FusionRule(RewriteRule):
    r"""Fusion rule (f) from [3] Eq. (70).

    Two same-type spiders connected by wires can be fused into a single spider.
    For Q-spiders: two q-spiders connected by wires can be fused with phase addition.
    For P-spiders: two p-spiders connected by wires can be fused with phase addition.

    The rule applies when:
    - Both spiders are the same type (both Q or both P)
    - They are in a ContractedDiagram
    - They are connected via some wires (I1 and I2 matching J1 and J2)
    - The resulting spider has:
        inputs = inputs of first + inputs of second (minus connected wires)
        outputs = outputs of first + outputs of second (minus connected wires)
    - Phase is the sum of the two phases
    """

    def match(self, diagram: Diagram, path: list[int] | None = None, parent: Diagram | None = None) -> list[list[int]]:
        """Find ContractedDiagram containing two same-type spiders.

        Parameters:
        ----------
        diagram : Diagram
            Input Diagram to search.
        path : list[int] | None
            Current path in the nested structure.
        parent : Diagram | None
            Parent of the current diagram.

        Returns:
        -------
        list[list[int]]
            List of paths to ContractedDiagram containing fusible spiders.
        """
        if path is None:
            path = []

        matches = []

        # Check if current diagram is a ContractedDiagram
        if isinstance(parent, ContractedDiagram) and parent == diagram:
            first = parent.first
            second = parent.second
            # Check if both are spiders of the same type
            if (isinstance(first, QSpider) and isinstance(second, QSpider) and self._are_connected(diagram)) or (  # noqa: PLR0916
                isinstance(first, PSpider) and isinstance(second, PSpider) and self._are_connected(diagram)
            ):
                # The spiders are fusible if they share connections
                matches.append(path.copy())
                return matches
            return matches

        # Recursive cases
        if isinstance(diagram, (CompositionDiagram, TensorDiagram)):
            for i, sub_diagram in enumerate(diagram.diagrams):
                new_path = [*path, i]
                matches.extend(self.match(sub_diagram, new_path, diagram))

        elif isinstance(diagram, ContractedDiagram):
            new_path_first = [*path, 0]
            if isinstance(diagram.first, (QSpider, PSpider)) and isinstance(diagram.second, (QSpider, PSpider)):
                new_path = [*path, 0]
                return self.match(diagram, new_path, diagram)
            if isinstance(diagram.first, (CompositionDiagram, TensorDiagram, ContractedDiagram)):
                new_path_first = [*path, 0]
                matches.extend(self.match(diagram.first, new_path_first, diagram))
            if isinstance(diagram.second, (CompositionDiagram, TensorDiagram, ContractedDiagram)):
                new_path_second = [*path, 1]
                matches.extend(self.match(diagram.second, new_path_second, diagram))

        return matches

    def apply_single(self, diagram: Diagram, match_path: list[int]) -> Diagram:  # noqa: C901
        """Fuse two same-type spiders in a ContractedDiagram.

        Parameters:
        ----------
        diagram : Diagram
            The diagram to modify.
        match_path : list[int]
            Path to the ContractedDiagram containing fusible spiders.

        Returns:
        -------
        Diagram
            New diagram with the two spiders fused into one.
        """
        if not match_path:
            return diagram
        # Navigate to the ContractedDiagram
        target = diagram
        if len(match_path) > 1:
            for idx in match_path:
                if isinstance(target, (CompositionDiagram, TensorDiagram)):
                    target = target.diagrams[idx]
                elif isinstance(target, ContractedDiagram):
                    if idx == 0 and not isinstance(target.first, (QSpider, PSpider)):
                        target = target.first
                    if idx == 1:
                        target = target.second
                else:
                    return diagram

        first = target.first
        second = target.second

        # Compute new arities
        new_n_in = len(target.kept_first_inputs) + len(target.kept_second_inputs)
        new_n_out = len(target.kept_first_outputs) + len(target.kept_second_outputs)

        # Phase is the sum
        new_phase = first.phase + second.phase

        # Create fused spider
        if isinstance(first, QSpider):
            fused = QSpider(new_n_in, new_n_out, new_phase)
        else:  # PSpider
            fused = PSpider(new_n_in, new_n_out, new_phase)

        # Replace the ContractedDiagram with the fused spider
        return self._replace_at_path(diagram, match_path, fused)

    def apply_rule(self, diagram: Diagram) -> Diagram:
        """Apply the fusion rule to the input diagram.

        Parameters:
        ----------
        diagram : Diagram
            The diagram to modify.

        Returns:
        -------
        Diagram
            New diagram with the identity spiders inside a composition diagram
            fused  if applicable, or the original diagram if not.
        """
        result = diagram
        matches = self.match(diagram)
        if not matches:
            return diagram

        # Process all matches
        for match_path in matches:
            result = self.apply_single(result, match_path)
        return result

    def _are_connected(self, contracted: ContractedDiagram) -> bool:
        """Check if the two diagrams in ContractedDiagram are connected.

        For fusion, we need at least one connection between the spiders.
        Parameters:
        ----------
        contracted : ContractedDiagram
            Input contracted diagram.

        Returns:
        -------
            bool
        """
        # Check if second spider's outputs connect to first spider's inputs
        return bool(len(contracted.J1) > 0 and len(contracted.J2) > 0) or bool(
            len(contracted.I1) > 0 and len(contracted.I2) > 0
        )

    def _replace_at_path(self, diagram: Diagram, path: list[int], replacement: Diagram) -> Diagram:
        """Replace the diagram at the given path with replacement.

        Parameters:
        ----------
        diagram : Diagram
            The diagram to modify.
        path : list[int]
            Path to the location where to replace the current ContractedDiagram by the
            reduced form.
        replacement: Diagram
            Diagram reduced after applying the fusion rule. It must replace
            its former version.

        Returns:
        -------
        Diagram
            New diagram with the identity spiders in a contracted diagram fused.
        """
        if path == [0]:
            return replacement

        idx = path[0]
        remaining_path = path[1:]

        if isinstance(diagram, CompositionDiagram):
            diagrams = list(diagram.diagrams)
            diagrams[idx] = self._replace_at_path(diagrams[idx], remaining_path, replacement)
            return CompositionDiagram(diagrams)

        if isinstance(diagram, TensorDiagram):
            diagrams = list(diagram.diagrams)
            diagrams[idx] = self._replace_at_path(diagrams[idx], remaining_path, replacement)
            return TensorDiagram(diagrams)

        if isinstance(diagram, ContractedDiagram):
            if idx == 0:
                new_first = self._replace_at_path(diagram.first, remaining_path, replacement)
                result = ContractedDiagram(
                    first=new_first,
                    second=diagram.second,
                    I1=diagram.I1,
                    I2=diagram.I2,
                    J1=diagram.J1,
                    J2=diagram.J2,
                )
            else:
                new_second = self._replace_at_path(diagram.second, remaining_path, replacement)
                result = ContractedDiagram(
                    first=diagram.first,
                    second=new_second,
                    I1=diagram.I1,
                    I2=diagram.I2,
                    J1=diagram.J1,
                    J2=diagram.J2,
                )
            return result

        return diagram


class ChainReductionRule(RewriteRule):
    """Simplifies chains of identical gates using algebraic identities.

    Reduces adjacent same-type gates in CompositionDiagram:
    - Q(ax^n) ∘ Q(bx^n) → Q((a+b)x^n)  (monomial, same degree)
    - P(ax^n) ∘ P(bx^n) → P((a+b)x^n)  (monomial, same degree)
    - R(θ) ∘ R(φ) → R(θ+φ)
    - BS(θ) ∘ BS(φ) → BS(θ+φ)
    - Sq(τ) ∘ Sq(κ) → Sq(τ·κ)
    - D(α) ∘ D(β) → D(α+β)
    - F ∘ F → F², F ∘ F ∘ F ∘ F → I, etc.
    - Finv ∘ Finv → F², Finv ∘ Finv ∘ Finv ∘ Finv → I, etc.
    - F ∘ Finv → I, Finv ∘ F → I
    - F² ∘ F² → I, F² ∘ F² ∘ F² → F², etc.

    """

    def match(self, diagram: Diagram, path: list[int] | None = None, parent: Diagram | None = None) -> list[dict]:  # noqa: C901
        """Find maximal chains of reducible gates in CompositionDiagram.

        Returns:
            dict : dictionary containing
                - 'path': path to the CompositionDiagram containing the chain
                - 'start': starting index of the chain
                - 'end': ending index of the chain (exclusive)
                - 'gate_type': type of gates in the chain
                - 'values': list of values for each gate in the chain

        Strategy:
        1. If current diagram is CompositionDiagram:
           a. Find all chains at this depth level
           b. Recursively explore each sub-diagram that is a container
        2. If current diagram is TensorDiagram or ContractedDiagram:
           a. Recursively explore each sub-diagram that is a container
        """
        if path is None:
            path = []

        matches = []

        # Case 1: Current diagram is a CompositionDiagram
        if isinstance(diagram, CompositionDiagram):
            # Step 1a: Find all chains at this depth level
            chains_at_this_level = self._find_chains_in_composition(diagram)
            matches.extend(
                {
                    "path": path.copy(),
                    "start": chain["start"],
                    "end": chain["end"],  # The last element of the chain is diagram.diagrams[end - 1]
                    "gate_type": chain["gate_type"],
                    "values": chain["values"],
                }
                for chain in chains_at_this_level
            )

            # Step 1b: Recursively explore each sub-diagram that is a container
            for i, sub_diagram in enumerate(diagram.diagrams):
                if isinstance(sub_diagram, (CompositionDiagram, TensorDiagram, ContractedDiagram)):
                    new_path = [*path, i]
                    deeper_matches = self.match(sub_diagram, new_path, diagram)
                    matches.extend(deeper_matches)

        # Case 2: Current diagram is a TensorDiagram
        elif isinstance(diagram, TensorDiagram):
            for i, sub_diagram in enumerate(diagram.diagrams):
                if isinstance(sub_diagram, (CompositionDiagram, TensorDiagram, ContractedDiagram)):
                    new_path = [*path, i]
                    deeper_matches = self.match(sub_diagram, new_path, diagram)
                    matches.extend(deeper_matches)

        # Case 3: Current diagram is a ContractedDiagram
        elif isinstance(diagram, ContractedDiagram):
            if isinstance(diagram.first, (CompositionDiagram, TensorDiagram, ContractedDiagram)):
                new_path_first = [*path, 0]
                matches.extend(self.match(diagram.first, new_path_first, diagram))

            if isinstance(diagram.second, (CompositionDiagram, TensorDiagram, ContractedDiagram)):
                new_path_second = [*path, 1]
                matches.extend(self.match(diagram.second, new_path_second, diagram))

        return matches

    def apply_single(self, diagram: Diagram, match: dict) -> Diagram:
        """Apply chain reduction to a specific match.

        Parameters:
        ----------
        diagram : Diagram
            The diagram to modify.
        match : dict
            Dictionary containing all the information about the chain to
            reduce.

        Returns:
        -------
        Diagram
            New diagram with the chain of reducible gates reduced.
        """
        path = match["path"]
        start = match["start"]
        end = match["end"]
        gate_type = match["gate_type"]
        values = match["values"]

        # Navigate to the CompositionDiagram containing the chain
        target = diagram
        for idx in path:
            if isinstance(target, (CompositionDiagram, TensorDiagram)):
                target = target.diagrams[idx]
            elif isinstance(target, ContractedDiagram):
                target = target.first if idx == 0 else target.second
            else:
                return diagram

        if not isinstance(target, CompositionDiagram):
            return diagram

        # Reduce the chain based on gate type
        reduced_gate = self.reduce_chain(gate_type, values)
        # Replace the chain with the reduced gate
        diagrams = list(target.diagrams)
        diagrams[start:end] = [reduced_gate]
        # Flatten if needed
        new_target = diagrams[0] if len(diagrams) == 1 else CompositionDiagram(diagrams)

        return self._replace_at_path(diagram, path, new_target)

    def apply_rule(self, diagram: Diagram) -> Diagram:
        """Apply the chain reduction rule to the input diagram.

        Parameters:
        ----------
        diagram : Diagram
            The diagram to modify.

        Returns:
        -------
        Diagram
            New diagram with all chains of reducible gates reduced.
        """
        result = diagram
        matches = self.match(diagram)
        if not matches:
            return diagram

        # Group matches by depth (length of path)
        matches_by_depth = {}
        for match in matches:
            depth = len(match["path"])
            if depth not in matches_by_depth:
                matches_by_depth[depth] = []
            matches_by_depth[depth].append(match)

        # Process from deepest to shallowest
        for depth in sorted(matches_by_depth.keys(), reverse=True):
            matches_at_depth = matches_by_depth[depth]
            matches_at_depth.sort(key=operator.itemgetter("start"), reverse=True)

            for match in matches_at_depth:
                result = self.apply_single(result, match)

        return result

    def get_gate_info(self, diagram: Diagram) -> tuple[str | None, any]:  # noqa: C901, PLR0911
        """Extract gate type and value from a diagram.

        Returns:
        -------
        tuple[str | None, any]
            (gate_type, value)
            gate_type: 'Q', 'P', 'R', 'BS', 'Sq', 'D', 'F', 'F2', or None
            value: for Q/P spiders: tuple of (phase_polynomial, degree)
                   where degree is the single degree if monomial, or None if mixed
                   for 'F': 'F' ,'Finv' or F2
        """
        # Q-Spider
        if isinstance(diagram, QSpider):
            # Check if phase is a monomial (single term)
            if len(diagram.phase.coeffs.keys()) == 1:
                degree = next(iter(diagram.phase.coeffs.keys()))
                return ("Q", (diagram.phase, degree, diagram.num_inputs, diagram.num_outputs))
            # Mixed polynomial - cannot chain
            return ("Q", (diagram.phase, None))

        # P-Spider
        if isinstance(diagram, PSpider):
            if len(diagram.phase.coeffs.keys()) == 1:
                degree = next(iter(diagram.phase.coeffs.keys()))
                return ("P", (diagram.phase, degree, diagram.num_inputs, diagram.num_outputs))
            return ("P", (diagram.phase, None))

        # Rotation Gate
        if isinstance(diagram, PhaseRotationGate):
            return ("R", diagram.theta)

        # Beamsplitter Gate
        if isinstance(diagram, BeamsplitterGate):
            return ("BS", diagram.theta)

        # Squeezing Gate
        if isinstance(diagram, SqueezingGate):
            return ("Sq", diagram.tau)

        # Displacement Gate
        if isinstance(diagram, DisplacementGate):
            return ("D", diagram.alpha)

        # Fourier Gates
        if isinstance(diagram, Fourier):
            return ("F", "F")

        if isinstance(diagram, FourierInv):
            return ("F", "Finv")

        # Fourier2 (F²)
        if isinstance(diagram, Fourier2):
            return ("F2", "F2")

        return (None, None)

    def reduce_chain(self, gate_type: str, values: list) -> Diagram:  # noqa: C901, PLR0911, PLR0912
        """Reduce a chain of gates to a single gate.

        Parameters:
        ----------
        gate_type : str
            Type of gates in the chain
        values : list
            List of values for each gate in the chain
            For Q/P: values are (phase, degree) tuples

        Returns:
        -------
        Diagram
            Reduced gate
        """
        identity = QSpider(2, 2, ZxPoly({}))
        if gate_type in {"Q", "P"}:
            identity = QSpider(values[0][2], values[-1][3], ZxPoly({}))
        elif gate_type in {"R", "Sq", "D", "F", "F_pair", "F2"}:
            identity = QSpider(1, 1, ZxPoly({}))

        if gate_type == "Q":
            total_phase = ZxPoly({})
            for v in values:
                total_phase += v[0]
            if total_phase == ZxPoly({}):
                return identity
            return QSpider(values[0][2], values[-1][3], total_phase)

        if gate_type == "P":
            total_phase = ZxPoly({})
            for v in values:
                total_phase += v[0]
            if total_phase == ZxPoly({}):
                return identity
            return PSpider(values[0][2], values[-1][3], total_phase)

        if gate_type == "R":
            total = sum(values)
            if total == 0:
                return identity
            return PhaseRotationGate(total)

        if gate_type == "BS":
            total = sum(values)
            if total == 0:
                return identity
            return BeamsplitterGate(total)

        if gate_type == "Sq":
            total = 1.0
            for v in values:
                total *= v
            if total == 1.0:  # noqa: RUF069
                return identity
            return SqueezingGate(total)

        if gate_type == "D":
            total = sum(values)
            if total == 0:
                return identity
            return DisplacementGate(total)

        if gate_type == "F":
            n = len(values)
            remainder = n % 4

            if remainder == 0:
                return identity
            if remainder == 1:
                if values[0] == "F":
                    return Fourier()
                return FourierInv()
            if remainder == 2:  # noqa: PLR2004
                return Fourier2()
            # Remainder == 3
            if values[0] == "F":
                return FourierInv()
            return Fourier()

        if gate_type == "F_pair":
            return identity

        if gate_type == "F2":
            if len(values) % 2 == 0:
                return identity
            return Fourier2()

        return identity

    def _find_chains_in_composition(self, comp: CompositionDiagram) -> list[dict]:  # noqa: C901, PLR0912
        """Find all maximal chains of reducible gates in a CompositionDiagram.

        Returns:
        -------
            list : list of chains of reducible gates each containing
                - Q: monomial Q spiders with SAME DEGREE
                - P: monomial P spiders with SAME DEGREE
                - R, BS, Sq, D: same-type chains (size >= 2)
                - F: F-F chains (any size), Finv-Finv chains (any size)
                - F-Finv pair (size exactly 2)
                - Finv-F pair (size exactly 2)
                - F2: F2-F2 chains (any size)
        """
        chains = []
        i = 0
        diagrams = comp.diagrams

        while i < len(diagrams):  # noqa: PLR1702
            gate_type, value = self.get_gate_info(diagrams[i])

            if gate_type is not None:
                start = i
                values = [value]

                # Special case: F and Finv pair (size 2 only)
                if gate_type == "F" and i + 1 < len(diagrams):
                    next_type, next_value = self.get_gate_info(diagrams[i + 1])
                    if next_type == "F":  # noqa: SIM102
                        if (value == "F" and next_value == "Finv") or (value == "Finv" and next_value == "F"):
                            chains.append({
                                "start": i,
                                "end": i + 2,
                                "gate_type": "F_pair",
                                "values": [value, next_value],
                            })
                            i += 2
                            continue

                # Regular chain (same type)
                j = i + 1
                while j < len(diagrams):
                    next_type, next_value = self.get_gate_info(diagrams[j])

                    if next_type == gate_type:
                        # For F, only chain same type (F with F, Finv with Finv)
                        if gate_type == "F":
                            if value == next_value:
                                values.append(next_value)
                                j += 1
                            else:
                                break
                        # For Q/P spiders, MUST be monomial with SAME DEGREE
                        elif gate_type in {"Q", "P"}:
                            # value is (phase, degree), next_value is (phase, degree)
                            # Both must have degree not None and be equal
                            if value[1] is not None and next_value[1] is not None and value[1] == next_value[1]:
                                values.append(next_value)
                                j += 1
                            else:
                                # Different degrees or mixed polynomials - break
                                break
                        else:
                            # All other types: R, BS, Sq, D, F2
                            values.append(next_value)
                            j += 1
                    else:
                        break

                # Record the chain if it has at least 2 gates
                if len(values) >= 2:
                    chains.append({"start": start, "end": j, "gate_type": gate_type, "values": values})

                i = max(i + 1, j)
            else:
                i += 1

        return chains

    def _replace_at_path(self, diagram: Diagram, path: list[int], replacement: Diagram) -> Diagram:
        """Replace the diagram at the given path with replacement.

        Parameters:
        ----------
        diagram : Diagram
            The diagram to modify.
        path : list[int]
            Path to the location where to replace the chain of diagrams by the
            reduced form.
        replacement: Diagram
            Diagram reduced after applying the chain reduction rule. It must
            replace its former version.

        Returns:
        -------
        Diagram
            New diagram with chains of reducible gates reduced.
        """
        if not path:
            return replacement

        idx = path[0]
        remaining_path = path[1:]

        if isinstance(diagram, CompositionDiagram):
            diagrams = list(diagram.diagrams)
            diagrams[idx] = self._replace_at_path(diagrams[idx], remaining_path, replacement)
            return self._flatten_composition(CompositionDiagram(diagrams))

        if isinstance(diagram, TensorDiagram):
            diagrams = list(diagram.diagrams)
            diagrams[idx] = self._replace_at_path(diagrams[idx], remaining_path, replacement)
            return TensorDiagram(diagrams)

        if isinstance(diagram, ContractedDiagram):
            if idx == 0:
                new_first = self._replace_at_path(diagram.first, remaining_path, replacement)
                if isinstance(new_first, CompositionDiagram):
                    new_first = self._flatten_composition(new_first)
                return ContractedDiagram(
                    first=new_first,
                    second=diagram.second,
                    I1=diagram.I1,
                    I2=diagram.I2,
                    J1=diagram.J1,
                    J2=diagram.J2,
                )
            new_second = self._replace_at_path(diagram.second, remaining_path, replacement)
            if isinstance(new_second, CompositionDiagram):
                new_second = self._flatten_composition(new_second)
            return ContractedDiagram(
                first=diagram.first,
                second=new_second,
                I1=diagram.I1,
                I2=diagram.I2,
                J1=diagram.J1,
                J2=diagram.J2,
            )

        return diagram

    def _flatten_composition(self, diagram: CompositionDiagram) -> Diagram:
        """Flatten a CompositionDiagram with one element.

        Parameters:
        ----------
        diagram : CompositionDiagram
            Composition diagram to flatten.

        Returns:
        -------
        CompositionDiagram
            Flattened CompositionDiagram.
        """
        if len(diagram.diagrams) == 1:
            return diagram.diagrams[0]
        return diagram
