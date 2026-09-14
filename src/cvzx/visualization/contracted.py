"""Drawing `ContractedDiagram` (partial trace with connections)."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from cvzx.ir.base import CompositionDiagram, ContractedDiagram, Diagram, Swap, TensorDiagram
from cvzx.visualization.geometry import Position, normalize_radiuses

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cvzx.visualization.protocol import Visualizer


class _ContractedMixin:
    """Draws a `ContractedDiagram` (partial trace with connections)."""

    def _draw_contracted(  # ruff: ignore[complex-structure, too-many-branches, too-many-arguments, too-many-locals, too-many-statements, too-many-positional-arguments]
        self: Visualizer,
        ax: plt.Axes,
        diagram: ContractedDiagram,
        x: float = 0,
        y: float = 0,
        comp_idx: int | None = None,
        sub_comp_idx: int | None = None,
        is_sub_tensor: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument]
        input_positions: list[Position] | None = None,
        radius: float | None = None,
        draw_kept_inputs: Sequence[int] | None = None,
        draw_kept_outputs: Sequence[int] | None = None,
    ) -> tuple[list[Position], list[Position], float]:
        """Draw a contracted diagram (feedback connections).

        Special care is needed because:
            - First diagram (D1) is drawn
            - Second diagram (D2) is drawn
            - Forward connections (I1→I2) are drawn as wires
            - Feedback connections (J2→J1) are drawn as wires
            - External wires use kept_first_inputs, kept_first_outputs,
              kept_second_inputs, kept_second_outputs

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
            diagram. It is a relevant argument for _draw_contracted because a
            composition can contain a contracted diagram.
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
        float
            radius of either the proper diagram, of the first diagram of a
            composition diagram or the list of radiuses of a tensor diagram.
        """
        if radius is None:
            radius = self.config.node_radius
        # Draw second diagram (D2)
        # We draw from bottom to top
        num_d2_out_wires = len(diagram.kept_second_outputs) + len(diagram.J2)
        num_d2_input_wires = len(diagram.kept_second_inputs) + len(diagram.I2)
        draw_kept_second_inputs: list[int] = [num_d2_input_wires - i - 1 for i in diagram.kept_second_inputs]
        draw_kept_second_inputs.sort()
        draw_kept_second_outputs: list[int] = [num_d2_out_wires - i - 1 for i in diagram.kept_second_outputs]
        draw_kept_second_outputs.sort()
        # We take into account draw_kept_inputs and draw_kept_outputs while drawing
        # the diagram D2
        if draw_kept_inputs is not None:
            num_sec_inputs = len(draw_kept_second_inputs)
            for elt in range(len(draw_kept_second_inputs) - 1, -1, -1):
                if elt not in draw_kept_inputs:
                    draw_kept_second_inputs.pop(elt)
        if draw_kept_outputs is not None:
            num_sec_outputs = len(draw_kept_second_outputs)
            for elt in range(len(draw_kept_second_outputs) - 1, -1, -1):
                if elt not in draw_kept_outputs:
                    draw_kept_second_outputs.pop(elt)
        # We do not need to reverse input_positions like in draw_tensor because
        is_sub_tensor = isinstance(diagram.second, CompositionDiagram)
        sec_in_positions = input_positions[: len(draw_kept_second_inputs)] if input_positions is not None else None
        first_shift, _ = self.vertical_shift_in_contraction(diagram.first, is_sub_tensor=True)
        spacing = 2 * self.config.vertical_spacing
        y2 = y - spacing - first_shift
        x2 = x
        output_positions_2, input_positions_2, output_radius_2 = self._draw_sub_diagram(
            ax,
            diagram.second,
            x2,
            y2,
            comp_idx=comp_idx,
            sub_comp_idx=sub_comp_idx,
            is_sub_tensor=is_sub_tensor,
            input_positions=sec_in_positions,
            radius=radius,
            draw_kept_inputs=draw_kept_second_inputs,
            draw_kept_outputs=draw_kept_second_outputs,
        )

        # Draw first diagram (D1)
        num_d1_out_wires = len(diagram.kept_first_outputs) + len(diagram.I1)
        num_d1_input_wires = len(diagram.kept_first_inputs) + len(diagram.J1)
        # Wires are drawn from bottom to top
        draw_kept_first_inputs: list[int] = [num_d1_input_wires - i - 1 for i in diagram.kept_first_inputs]
        draw_kept_first_inputs.sort()
        draw_kept_first_outputs: list[int] = [num_d1_out_wires - i - 1 for i in diagram.kept_first_outputs]
        draw_kept_first_outputs.sort()
        # We take into account draw_kept_inputs and draw_kept_outputs while drawing
        # the diagram D1
        if draw_kept_inputs is not None:
            for elt in range(len(draw_kept_first_inputs) - 1, -1, -1):
                if elt + num_sec_inputs not in draw_kept_inputs:
                    draw_kept_first_inputs.pop(elt)
        if draw_kept_outputs is not None:
            for elt in range(len(draw_kept_first_outputs) - 1, -1, -1):
                if elt + num_sec_outputs not in draw_kept_outputs:
                    draw_kept_first_outputs.pop(elt)
        is_sub_tensor = isinstance(diagram.first, CompositionDiagram)
        fir_in_positions = input_positions[len(draw_kept_second_inputs) :] if input_positions is not None else None
        x1, y1 = x, y
        output_positions_1, input_positions_1, output_radius_1 = self._draw_sub_diagram(
            ax,
            diagram.first,
            x1,
            y1,
            comp_idx=comp_idx,
            sub_comp_idx=sub_comp_idx,
            is_sub_tensor=is_sub_tensor,
            input_positions=fir_in_positions,
            radius=radius,
            draw_kept_inputs=draw_kept_first_inputs,
            draw_kept_outputs=draw_kept_first_outputs,
        )

        color_in = self.config.colors["contraction_in"]
        color_out = self.config.colors["contraction_out"]
        if isinstance(output_radius_2, (int, float)):
            output_radius_2 = [output_radius_2]
        if isinstance(output_radius_1, (int, float)):
            output_radius_1 = [output_radius_1]
        # We must make sure there is no equal output radiuses
        new_output_radius_2 = normalize_radiuses(output_radius_2)
        new_output_radius_1 = normalize_radiuses(output_radius_1)
        J_dict = {diagram.J2[i]: diagram.J1[i] for i in range(len(diagram.J1))}  # ruff: ignore[non-lowercase-variable-in-function]
        I_dict = {diagram.I1[i]: diagram.I2[i] for i in range(len(diagram.I2))}  # ruff: ignore[non-lowercase-variable-in-function]
        # Let's compute the distance between D1 and D2
        box_dist = self.config.contraction_shift
        # Draw feedback connections (J2 → J1)
        # J2: indices of outputs from second diagram
        # J1: indices of inputs from first diagram
        # Connection order: J2[k] connects to J1[k]
        # First we draw incoming arrows J2[k]s of D2
        second_diagrams: list[Diagram] = (
            list(deepcopy(diagram.second.diagrams)) if isinstance(diagram.second, TensorDiagram) else [diagram.second]
        )
        second_diagrams.reverse()
        k = 0
        J2_info = {}  # ruff: ignore[non-lowercase-variable-in-function]
        for j, sub_diagram in enumerate(second_diagrams):
            # A sub-diagram with no outputs at all (e.g. a `VoidDiagram`
            # or a bare effect) has nothing to offset -- the loops below
            # never index into `x_offset_2_i` in that case anyway, since
            # `range(k, k + 0)` never yields, so an empty list is exactly
            # the right stand-in (avoids popping off an empty/singleton
            # list further down).
            if sub_diagram.num_outputs:
                x_offset_2_i = [
                    float(v) for v in np.linspace(0, 2 * new_output_radius_2[j], sub_diagram.num_outputs + 2)
                ]
                x_offset_2_i.pop(0)
                x_offset_2_i.pop(1)
            else:
                x_offset_2_i = []
            sub_output_positions_2 = output_positions_2[k : k + sub_diagram.num_outputs]
            sub_J2 = []  # ruff: ignore[non-lowercase-variable-in-function]
            sub_J2_dict = {}  # ruff: ignore[non-lowercase-variable-in-function]
            # This is to allow the arrow to appear when there is a Swap
            padding = isinstance(sub_diagram, Swap)
            padding_coef = 0.4
            for ind in range(k, k + sub_diagram.num_outputs):
                if num_d2_out_wires - ind - 1 in diagram.J2:
                    sub_J2.append(ind - k)
                    sub_J2_dict[ind - k] = num_d2_out_wires - ind - 1
            # We draw from bottom to top
            x_offset_2_i.reverse()
            J2_points = {}  # ruff: ignore[non-lowercase-variable-in-function]
            for i in sub_J2:
                # We draw from bottom to top
                sec_point_i = sub_output_positions_2[i]
                ax.plot(
                    [sec_point_i[0], sec_point_i[0] - x_offset_2_i[i]],
                    [sec_point_i[1], sec_point_i[1]],
                    linewidth=self.config.wire_width,
                    color=color_in,
                )
                sec_arrow_i_1 = patches.FancyArrowPatch(
                    (sec_point_i[0] - x_offset_2_i[i], sec_point_i[1]),
                    (sec_point_i[0] - x_offset_2_i[i], y2 + output_radius_2[j] * (1 + padding_coef * padding)),
                    arrowstyle="->",
                    ec=color_in,
                    mutation_scale=20,
                    linewidth=self.config.wire_width,
                )
                ax.add_patch(sec_arrow_i_1)
                J2_points[i] = (
                    sec_point_i[0] - x_offset_2_i[i],
                    y2 + output_radius_2[j] * (1 + padding_coef * padding),
                )
            J2_info[j] = (sub_J2, sub_J2_dict, J2_points)
            k += sub_diagram.num_outputs

        # Then the receptors arrows J1[k]s of D1
        # We will draw from the last element of J1 to the first
        first_diagrams: list[Diagram] = (
            list(deepcopy(diagram.first.diagrams)) if isinstance(diagram.first, TensorDiagram) else [diagram.first]
        )
        first_diagrams.reverse()
        k = 0
        J1_info = {}  # ruff: ignore[non-lowercase-variable-in-function]
        J1_sub_diag_mapping = {}  # ruff: ignore[non-lowercase-variable-in-function]
        for j, sub_diagram in enumerate(first_diagrams):
            # See the analogous `x_offset_2_i` guard above: a sub-diagram
            # with no inputs has nothing to offset.
            if sub_diagram.num_inputs:
                x_offset_1_i = [
                    float(v) for v in np.linspace(0, 2 * new_output_radius_1[j], sub_diagram.num_inputs + 2)
                ]
                x_offset_1_i.pop(0)
                x_offset_1_i.pop(1)
            else:
                x_offset_1_i = []
            sub_input_positions_1 = input_positions_1[k : k + sub_diagram.num_inputs]
            sub_J1 = []  # ruff: ignore[non-lowercase-variable-in-function]
            sub_J1_dict = {}  # ruff: ignore[non-lowercase-variable-in-function]
            sub_J1_dict_inv = {}  # ruff: ignore[non-lowercase-variable-in-function]
            for ind in range(k, k + sub_diagram.num_inputs):
                if num_d1_input_wires - ind - 1 in diagram.J1:
                    sub_J1.append(ind - k)
                    sub_J1_dict[ind - k] = num_d1_input_wires - ind - 1
                    sub_J1_dict_inv[num_d1_input_wires - ind - 1] = ind - k
                    J1_sub_diag_mapping[num_d1_input_wires - ind - 1] = j
            # We draw from bottom to top
            J1_points = {}  # ruff: ignore[non-lowercase-variable-in-function]
            padding = isinstance(sub_diagram, Swap)
            padding_coef = 0.4
            for i in sub_J1:
                # We draw from bottom to top
                fir_point_i = sub_input_positions_1[i]
                # We move fir_point_i to the end of the arrow
                fir_point_i = (fir_point_i[0] - 2 * output_radius_1[j], fir_point_i[1])
                ax.plot(
                    [fir_point_i[0] + x_offset_1_i[i], fir_point_i[0] + x_offset_1_i[i]],
                    [
                        y1 - first_shift - output_radius_1[-1] * (1 + padding_coef * padding),
                        fir_point_i[1],
                    ],
                    linewidth=self.config.wire_width,
                    color=color_in,
                )
                J1_points[i] = (fir_point_i[0] + x_offset_1_i[i], fir_point_i[1])
                fir_arrow_i_1 = patches.FancyArrowPatch(
                    (fir_point_i[0] + x_offset_1_i[i], fir_point_i[1]),
                    (fir_point_i[0], fir_point_i[1]),
                    arrowstyle="->",
                    ec=color_in,
                    mutation_scale=20,
                    linewidth=self.config.wire_width,
                )
                ax.add_patch(fir_arrow_i_1)
            J1_info[j] = (sub_J1, (sub_J1_dict, sub_J1_dict_inv), J1_points)
            k += sub_diagram.num_inputs

        # Now we link the arrows J1[k], J2[k]
        y_offset_J = [float(v) for v in np.linspace(0, box_dist / 2, len(diagram.J2) + 2)]  # ruff: ignore[non-lowercase-variable-in-function]
        y_offset_J.pop(0)
        y_offset_J.pop(-1)
        for j, sub_diagram in enumerate(second_diagrams):
            sub_J2 = J2_info[j][0]  # ruff: ignore[non-lowercase-variable-in-function]
            sub_J2_dict = J2_info[j][1]  # ruff: ignore[non-lowercase-variable-in-function]
            J2_points = J2_info[j][2]  # ruff: ignore[non-lowercase-variable-in-function]
            for i in sub_J2:
                # We draw from down to top
                starting_point = J2_points[i]
                ax.plot(
                    [starting_point[0], starting_point[0]],
                    [starting_point[1], starting_point[1] + y_offset_J[diagram.J2.index(sub_J2_dict[i])]],
                    linewidth=self.config.wire_width,
                    color=color_in,
                )
                fir_sub_diag_ind = J1_sub_diag_mapping[J_dict[sub_J2_dict[i]]]
                sub_J1_info = J1_info[fir_sub_diag_ind]  # ruff: ignore[non-lowercase-variable-in-function]
                sub_J1_ind = sub_J1_info[1][1][J_dict[sub_J2_dict[i]]]  # ruff: ignore[non-lowercase-variable-in-function]
                end_point = sub_J1_info[2][sub_J1_ind]
                ax.plot(
                    [starting_point[0], end_point[0]],
                    [
                        starting_point[1] + y_offset_J[diagram.J2.index(sub_J2_dict[i])],
                        starting_point[1] + y_offset_J[diagram.J2.index(sub_J2_dict[i])],
                    ],
                    linewidth=self.config.wire_width,
                    color=color_in,
                )
                ax.plot(
                    [end_point[0], end_point[0]],
                    [starting_point[1] + y_offset_J[diagram.J2.index(sub_J2_dict[i])], end_point[1]],
                    linewidth=self.config.wire_width,
                    color=color_in,
                )
            k += sub_diagram.num_inputs

        # Draw forward connections (I1 → I2)
        # I1: indices of outputs from first diagram
        # I2: indices of inputs from second diagram
        # Connection order: I1[k] connects to I2[k]
        # I must recompute x_offset and y_offset
        I1_info = {}  # ruff: ignore[non-lowercase-variable-in-function]
        I1_sub_diag_mapping = {}  # ruff: ignore[non-lowercase-variable-in-function]
        k = 0
        for j, sub_diagram in enumerate(first_diagrams):
            # See the analogous `x_offset_2_i` guard above: a sub-diagram
            # with no outputs has nothing to offset.
            if sub_diagram.num_outputs:
                x_offset_1_i = [
                    float(v) for v in np.linspace(0, 2 * new_output_radius_1[j], sub_diagram.num_outputs + 2)
                ]
                x_offset_1_i.pop(0)
                x_offset_1_i.pop(1)
            else:
                x_offset_1_i = []
            sub_output_positions_1 = output_positions_1[k : k + sub_diagram.num_outputs]
            sub_I1 = []  # ruff: ignore[non-lowercase-variable-in-function]
            sub_I1_dict = {}  # ruff: ignore[non-lowercase-variable-in-function]
            for ind in range(k, k + sub_diagram.num_outputs):
                if num_d1_out_wires - ind - 1 in diagram.I1:
                    sub_I1.append(ind - k)
                    sub_I1_dict[ind - k] = num_d1_out_wires - ind - 1
                    I1_sub_diag_mapping[num_d1_out_wires - ind - 1] = j
            # This is to allow the arrow to appear when when there is a Swap
            padding = isinstance(sub_diagram, Swap)
            padding_coef = 0.4
            # We draw from bottom to top
            I1_points = {}  # ruff: ignore[non-lowercase-variable-in-function]
            for i in sub_I1:
                # We draw from down to top
                fir_point_i = sub_output_positions_1[i]
                ax.plot(
                    [fir_point_i[0], fir_point_i[0] - x_offset_1_i[i]],
                    [fir_point_i[1], fir_point_i[1]],
                    linewidth=self.config.wire_width,
                    color=color_out,
                )
                fir_arrow_i_1 = patches.FancyArrowPatch(
                    (fir_point_i[0] - x_offset_1_i[i], fir_point_i[1]),
                    (
                        fir_point_i[0] - x_offset_1_i[i],
                        y1 - first_shift - output_radius_1[-1] * (1 + padding_coef * padding),
                    ),
                    arrowstyle="->",
                    ec=color_out,
                    mutation_scale=20,
                    linewidth=self.config.wire_width,
                )
                I1_points[i] = (
                    fir_point_i[0] - x_offset_1_i[i],
                    y1 - first_shift - output_radius_1[-1] * (1 + padding_coef * padding),
                )
                ax.add_patch(fir_arrow_i_1)
            I1_info[j] = (sub_I1, sub_I1_dict, I1_points)
            k += sub_diagram.num_outputs
        # Then we draw arrows I2[k]
        I2_info = {}  # ruff: ignore[non-lowercase-variable-in-function]
        I2_sub_diag_mapping = {}  # ruff: ignore[non-lowercase-variable-in-function]
        k = 0
        for j, sub_diagram in enumerate(second_diagrams):
            # See the analogous `x_offset_2_i` guard above: a sub-diagram
            # with no inputs has nothing to offset.
            if sub_diagram.num_inputs:
                x_offset_2_i = [
                    float(v) for v in np.linspace(0, 2 * new_output_radius_2[j], sub_diagram.num_inputs + 2)
                ]
                x_offset_2_i.pop(0)
                x_offset_2_i.pop(1)
            else:
                x_offset_2_i = []
            sub_input_positions_2 = input_positions_2[k : k + sub_diagram.num_inputs]
            sub_I2 = []  # ruff: ignore[non-lowercase-variable-in-function]
            sub_I2_dict = {}  # ruff: ignore[non-lowercase-variable-in-function]
            sub_I2_dict_inv = {}  # ruff: ignore[non-lowercase-variable-in-function]
            for ind in range(k, k + sub_diagram.num_inputs):
                if num_d2_input_wires - ind - 1 in diagram.I2:
                    sub_I2.append(ind - k)
                    sub_I2_dict[ind - k] = num_d2_input_wires - ind - 1
                    sub_I2_dict_inv[num_d2_input_wires - ind - 1] = ind - k
                    I2_sub_diag_mapping[num_d2_input_wires - ind - 1] = j
            padding = isinstance(sub_diagram, Swap)
            padding_coef = 0.4
            # We draw from bottom to top
            I2_points = {}  # ruff: ignore[non-lowercase-variable-in-function]
            for i in sub_I2:
                # We draw from down to top
                sec_point_i = sub_input_positions_2[i]
                # We move sec_point_i to the end of the arrow
                sec_point_i = (sec_point_i[0] - 2 * output_radius_2[j], sec_point_i[1])
                ax.plot(
                    [sec_point_i[0] + x_offset_2_i[i], sec_point_i[0] + x_offset_2_i[i]],
                    [
                        y2 + output_radius_2[j] * (1 + padding_coef * padding),
                        sec_point_i[1],
                    ],
                    linewidth=self.config.wire_width,
                    color=color_out,
                )
                I2_points[i] = (sec_point_i[0] + x_offset_2_i[i], sec_point_i[1])
                sec_arrow_i_2 = patches.FancyArrowPatch(
                    (sec_point_i[0] + x_offset_2_i[i], sec_point_i[1]),
                    (sec_point_i[0], sec_point_i[1]),
                    arrowstyle="->",
                    ec=color_out,
                    mutation_scale=20,
                    linewidth=self.config.wire_width,
                )
                ax.add_patch(sec_arrow_i_2)
            I2_info[j] = (sub_I2, (sub_I2_dict, sub_I2_dict_inv), I2_points)
            k += sub_diagram.num_inputs
        # Then we link arrows I1[k] , I2[k]
        y_offset_I = [float(v) for v in np.linspace(0, box_dist / 2, len(diagram.I1) + 2)]  # ruff: ignore[non-lowercase-variable-in-function]
        y_offset_I.pop(0)
        y_offset_I.pop(-1)
        k = 0
        for j, sub_diagram in enumerate(first_diagrams):
            sub_I1 = I1_info[j][0]  # ruff: ignore[non-lowercase-variable-in-function]
            sub_I1_dict = I1_info[j][1]  # ruff: ignore[non-lowercase-variable-in-function]
            I1_points = I1_info[j][2]  # ruff: ignore[non-lowercase-variable-in-function]
            for i in sub_I1:
                # We draw from down to top
                starting_point = I1_points[i]
                ax.plot(
                    [starting_point[0], starting_point[0]],
                    [starting_point[1], starting_point[1] - y_offset_I[diagram.I1.index(sub_I1_dict[i])]],
                    linewidth=self.config.wire_width,
                    color=color_out,
                )
                sec_sub_diag_ind = I2_sub_diag_mapping[I_dict[sub_I1_dict[i]]]
                sub_I2_info = I2_info[sec_sub_diag_ind]  # ruff: ignore[non-lowercase-variable-in-function]
                sub_I2_ind = sub_I2_info[1][1][I_dict[sub_I1_dict[i]]]  # ruff: ignore[non-lowercase-variable-in-function]
                end_point = sub_I2_info[2][sub_I2_ind]
                ax.plot(
                    [starting_point[0], end_point[0]],
                    [
                        starting_point[1] - y_offset_I[diagram.I1.index(sub_I1_dict[i])],
                        starting_point[1] - y_offset_I[diagram.I1.index(sub_I1_dict[i])],
                    ],
                    linewidth=self.config.wire_width,
                    color=color_out,
                )
                ax.plot(
                    [end_point[0], end_point[0]],
                    [starting_point[1] - y_offset_I[diagram.I1.index(sub_I1_dict[i])], end_point[1]],
                    linewidth=self.config.wire_width,
                    color=color_out,
                )
            k += sub_diagram.num_inputs
        # We take outputs from bottom to top
        draw_kept_second_inputs = [num_d2_input_wires - i - 1 for i in diagram.kept_second_inputs]
        draw_kept_second_inputs.sort()
        draw_kept_second_outputs = [num_d2_out_wires - i - 1 for i in diagram.kept_second_outputs]
        draw_kept_second_outputs.sort()
        draw_kept_first_inputs = [num_d1_input_wires - i - 1 for i in diagram.kept_first_inputs]
        draw_kept_first_inputs.sort()
        draw_kept_first_outputs = [num_d1_out_wires - i - 1 for i in diagram.kept_first_outputs]
        draw_kept_first_outputs.sort()
        output_positions = [output_positions_2[i] for i in draw_kept_second_outputs]
        output_positions += [output_positions_1[i] for i in draw_kept_first_outputs]
        # We also return the initial input positions
        init_input_positions = [input_positions_2[i] for i in draw_kept_second_inputs]
        init_input_positions += [input_positions_1[i] for i in draw_kept_first_inputs]
        return output_positions, init_input_positions, radius
