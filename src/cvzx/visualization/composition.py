"""Drawing `CompositionDiagram` (sequential) and `TensorDiagram` (parallel) containers."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from cvzx.ir.base import CompositionDiagram, ContractedDiagram, TensorDiagram
from cvzx.visualization.geometry import Position, reorder_positions, require

if TYPE_CHECKING:
    from collections.abc import Sequence

    import matplotlib.pyplot as plt

    from cvzx.visualization.protocol import Visualizer


class _CompositionMixin:
    """Draws sequential (`CompositionDiagram`) and parallel (`TensorDiagram`) containers."""

    def _draw_composition(  # ruff: ignore[complex-structure, too-many-branches, too-many-arguments, too-many-locals, too-many-positional-arguments]
        self: Visualizer,
        ax: plt.Axes,
        diagram: CompositionDiagram,
        x: float = 0,
        y: float = 0,
        comp_idx: int | None = None,
        sub_comp_idx: int | None = None,
        is_sub_tensor: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument]
        input_positions: list[Position] | None = None,
        radius: float | None = None,
        draw_kept_inputs: Sequence[int] | None = None,
        draw_kept_outputs: Sequence[int] | None = None,
    ) -> tuple[list[Position], list[Position], float | list[float]]:
        """Draw a composition diagram.

        Parameters
        ----------
        ax : plt.Axes
            Matplotlib axes
        diagram : Diagram
            The sub-diagram to draw
        x: float
            First coordinate of the diagram
        y: float
            Second coordinate of the diagram
        comp_idx: int | None
            Index of a sub-diagram inside a composition. In a composition,
            we don't need to draw the input/output wires of all sub-diagrams.
        sub_comp_idx: int | None
            Index of a proper diagram inside a sub-composition diagram. Along with
            the comp_idx, their value determine whether to draw input and/or output
            wires on the proper diagram.
        input_positions: list[Position] | None
            List of positions received from the sub-diagram that precedes the
            current diagram in a composition diagram. It will help to draw
            the input wires from that sub-diagram to the current diagram.
            In case list is empty, it means the sub-diagram is not part of a
            composition diagram.
        is_sub_tensor: bool | None
            Boolean tag to indicate that a composition is a sub-diagram of a
            tensor diagram.
        radius: float | None
            This is the variable used to define the parameters:
            arrow_length, horizontal span and the horizontal spacing. So it is
            the unit variable of the diagram. It is initialized as None because
            one can not have access to self in argument definition.
        draw_kept_inputs: Sequence[int] | None
            List of indices of input wires from bottom to top not involved in a
            partial trace. This parameter is computed inside _draw_contracted_diagram.
        draw_kept_outputs: Sequence[int] | None
            List of indices of output wires from bottom to top not involved in a
            partial trace. This parameter is computed inside _draw_contracted_diagram.

        Returns
        -------
        list[Position]
            Lists of output indices that will serve as the input indices of the
            next diagram if there is any.
        list[Position]
            Lists of proper input indices, i.e. different from input indices
            received from a sub-diagram.
        float | list[float]
            radius of the first diagram of a composition diagram it can be a
            list of radiuses if it's a tensor diagram.
        """
        if radius is None:
            radius = self.config.node_radius
        diagram_length = len(diagram.diagrams)
        if diagram_length == 0:
            return [], [], radius

        if draw_kept_inputs is None:
            draw_kept_inputs = range(diagram.num_inputs)
            draw_kept_outputs = range(diagram.num_outputs)
        # draw_kept_inputs and draw_kept_outputs are always provided together.
        draw_kept_outputs = require(draw_kept_outputs, "draw_kept_outputs must be set alongside draw_kept_inputs")
        output_radius: float | list[float] = radius
        # Draw each sub-diagram at its position
        # is_sub_tensor is True means that we are inside a composition block
        # which is part of a tensor diagram therefore we must resize the
        # sub_hor_span, spacing and radius
        if is_sub_tensor:
            sub_hor_span = (
                6 * radius / diagram_length
                if comp_idx != -1 and comp_idx is not None
                else 18 * radius / (2 * diagram_length + 1)
            )
        else:
            sub_hor_span = 6 * radius
        sub_radius = sub_hor_span / 6
        sub_hor_spacing = 4 * sub_radius
        sub_x = x + 3 * (radius - sub_radius)
        # Calculate positions for each sub-diagram
        # Composition Diagrams are from drawn left to right
        spacing_i = sub_hor_spacing
        output_positions: list[Position] | None = input_positions
        init_input_positions: list[Position] = []
        for i, sub_diagram in enumerate(diagram.diagrams):
            spacing_i -= sub_hor_spacing
            idx = i
            input_pos = output_positions
            if is_sub_tensor:
                sub_comp_idx = idx
            else:
                comp_idx = idx
            if i == 0:
                sent_kept_inputs = draw_kept_inputs
                sent_kept_outputs: Sequence[int] = range(sub_diagram.num_outputs)
            elif i == diagram_length - 1:
                # output_positions was set by the previous iteration (i > 0 here).
                output_positions = require(output_positions, "output_positions must be set for i > 0")
                input_pos = reorder_positions(diagram.connectivity[i - 1], output_positions)
                sent_kept_inputs = range(sub_diagram.num_inputs)
                sent_kept_outputs = draw_kept_outputs
                if is_sub_tensor:
                    sub_comp_idx = -1
                else:
                    comp_idx = -1
            else:
                output_positions = require(output_positions, "output_positions must be set for i > 0")
                input_pos = reorder_positions(diagram.connectivity[i - 1], output_positions)
                sent_kept_inputs = range(sub_diagram.num_inputs)
                sent_kept_outputs = range(sub_diagram.num_outputs)
            output_positions, init_input_pos, sub_output_radius = self._draw_sub_diagram(
                ax,
                sub_diagram,
                sub_x + spacing_i,
                y,
                comp_idx=comp_idx,
                sub_comp_idx=sub_comp_idx,
                is_sub_tensor=is_sub_tensor,
                input_positions=input_pos,
                radius=sub_radius,
                draw_kept_inputs=sent_kept_inputs,
                draw_kept_outputs=sent_kept_outputs,
            )
            # The input positions of a composition diagram are
            # the input positions of its first sub-diagram
            if i == 0:
                init_input_positions = init_input_pos
                output_radius = sub_output_radius
        # The loop always runs at least once since diagram_length > 0 (checked above),
        # so output_positions has always been reassigned by _draw_sub_diagram by now.
        output_positions = require(output_positions, "output_positions must be set after a non-empty composition")
        return output_positions, init_input_positions, output_radius

    def _draw_tensor(  # ruff: ignore[complex-structure, too-many-branches, too-many-arguments, too-many-locals, too-many-statements, too-many-positional-arguments]
        self: Visualizer,
        ax: plt.Axes,
        diagram: TensorDiagram,
        x: float = 0,
        y: float = 0,
        comp_idx: int | None = None,
        sub_comp_idx: int | None = None,
        is_sub_tensor: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument]
        input_positions: list[Position] | None = None,
        radius: float | None = None,
        draw_kept_inputs: Sequence[int] | None = None,
        draw_kept_outputs: Sequence[int] | None = None,
    ) -> tuple[list[Position], list[Position], list[float]]:
        """Draw a tensor diagram (parallel).

        Parameters
        ----------
        ax : plt.Axes
            Matplotlib axes
        diagram : Diagram
            The sub-diagram to draw
        x: float
            First coordinate of the diagram
        y: float
            Second coordinate of the diagram
        comp_idx: int | None
            Index of a sub-diagram inside a composition. In a composition,
            we don't need to draw the input/output wires of all sub-diagrams.
        sub_comp_idx: int | None
            Index of a proper diagram inside a sub-composition diagram. Along with
            the comp_idx, their value determine whether to draw input and/or output
            wires on the proper diagram.
        is_sub_tensor: bool | None
            Boolean tag to indicate that a composition is a sub-diagram of a tensor
            diagram. It is a relevant argument for _draw_tensor because a
            composition can contain a tensor diagram.
        input_positions: list[Position] | None
            List of positions received from the sub-diagram that precedes the
            current diagram in a composition diagram. It will help to draw
            the input wires from that sub-diagram to the current diagram.
            In case list is empty, it means the sub-diagram is not part of a
            composition diagram.
        radius: float | None
            This is the variable used to define the parameters:
            arrow_length, horizontal span and the horizontal spacing. So it is
            the unit variable of the diagram. It is initialized as None because
            one can not have access to self in argument definition.
        draw_kept_inputs: Sequence[int] | None
            List of indices of input wires from bottom to top not involved in a
            partial trace. This parameter is computed inside _draw_contracted_diagram.
        draw_kept_outputs: Sequence[int] | None
            List of indices of output wires from bottom to top not involved in a
            partial trace. This parameter is computed inside _draw_contracted_diagram.

        Returns
        -------
        list[Position]
            Lists of output indices that will serve as the input indices of the
            next diagram if there is any.
        list[Position]
            Lists of proper input indices, i.e. different from input indices
            received from a sub-diagram.
        list[float]
            the list of radiuses of the tensor diagram.
        """
        if len(diagram.diagrams) == 0:
            return [], [], []
        # Calculate positions for each sub-diagram
        # Tensor Diagrams are drawn top to bottom
        output_positions: list[Position] = []
        init_input_positions: list[Position] = []
        output_radius: list[float] = []
        inp_ind = 0
        out_ind = 0
        if radius is None:
            radius = self.config.node_radius
            vertical_spacing = self.config.vertical_spacing
        else:
            vertical_spacing = self.config.vertical_factor * radius
        # input positions are listed from the bottom to the top and since
        # we draw from top to bottom in draw_tensor we must reverse
        # input_positions
        if input_positions is not None:
            input_positions.reverse()
        if draw_kept_inputs is not None:
            # draw_kept_inputs and draw_kept_outputs are always provided together.
            draw_kept_outputs = require(draw_kept_outputs, "draw_kept_outputs must be set alongside draw_kept_inputs")
            num_inputs = diagram.num_inputs
            draw_kept_inputs = sorted(num_inputs - i - 1 for i in draw_kept_inputs)
            num_outputs = diagram.num_outputs
            draw_kept_outputs = sorted(num_outputs - i - 1 for i in draw_kept_outputs)
        contract_shift = 0.0
        h = vertical_spacing
        for sub_diagram in diagram.diagrams:
            is_sub_tensor = isinstance(sub_diagram, CompositionDiagram)
            if contract_shift != 0:
                h -= contract_shift
            else:
                h -= vertical_spacing
            # Draw sub-diagram with local coordinates
            sub_input_positions: list[Position] | None = None
            sub_kept_inputs: list[int] | None = None
            sub_kept_outputs: list[int] | None = None
            if input_positions is not None:
                sub_input_positions = input_positions[inp_ind : inp_ind + sub_diagram.num_inputs]
                # We must reverse back sub_input_positions
                sub_input_positions.reverse()
            if draw_kept_inputs is not None and draw_kept_outputs is not None:
                sub_kept_inputs = []
                for ind in range(inp_ind, inp_ind + sub_diagram.num_inputs):
                    if ind in draw_kept_inputs:
                        sub_kept_inputs.append(ind - inp_ind)
                sub_kept_outputs = []
                for ind in range(out_ind, out_ind + sub_diagram.num_outputs):
                    if ind in draw_kept_outputs:
                        sub_kept_outputs.append(ind - out_ind)
                # We must reverse back the lits above
                sub_kept_inputs = sorted(sub_diagram.num_inputs - p - 1 for p in sub_kept_inputs)
                sub_kept_outputs = sorted(sub_diagram.num_outputs - p - 1 for p in sub_kept_outputs)
            # Find local_kept_inputs and local_kept_outputs
            output_pos, init_input_pos, sub_radius = self._draw_sub_diagram(
                ax,
                sub_diagram,
                x,
                y + h,
                comp_idx=comp_idx,
                sub_comp_idx=sub_comp_idx,
                is_sub_tensor=is_sub_tensor,
                input_positions=sub_input_positions,
                radius=radius,
                draw_kept_outputs=sub_kept_outputs,
                draw_kept_inputs=sub_kept_inputs,
            )
            if isinstance(sub_diagram, ContractedDiagram):
                contract_shift, _ = self.vertical_shift_in_contraction(sub_diagram)
                contract_shift += 2 * self.config.node_radius
            else:
                contract_shift = 0.0
            output_positions = output_pos + output_positions
            init_input_positions = init_input_pos + init_input_positions
            # NOTE: sub_radius is float | list[float] in general — it is only a
            # list[float] when sub_diagram is a CompositionDiagram whose own first
            # element is itself a TensorDiagram, a case this bookkeeping (one radius
            # per top-level tensor element) does not actually support; that pre-existing
            # limitation is left as-is here rather than changed as part of this typing
            # pass.
            output_radius = [cast("float", sub_radius), *output_radius]
            inp_ind += sub_diagram.num_inputs
            out_ind += sub_diagram.num_outputs
        return output_positions, init_input_positions, output_radius
