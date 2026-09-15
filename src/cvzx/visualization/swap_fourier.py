"""Drawing `Swap`, the Fourier gate family, and feedforward annotation arrows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib import patches

from cvzx.ir.base import Diagram, Fourier2, FourierInv, Swap

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cvzx.visualization.geometry import Position
    from cvzx.visualization.protocol import Visualizer


class _SwapFourierMixin:
    """Draws `Swap`, `Fourier`/`Fourier2`/`FourierInv`, and feedforward arrows."""

    def _draw_swap(  # ruff: ignore[complex-structure, too-many-arguments, too-many-positional-arguments, too-many-branches]
        self: Visualizer,
        ax: plt.Axes,
        x: float,
        y: float,
        diagram: Swap | None = None,
        comp_idx: int | None = None,
        sub_comp_idx: int | None = None,
        input_positions: list[Position] | None = None,
        radius: float | None = None,
        draw_kept_inputs: Sequence[int] | None = None,
        draw_kept_outputs: Sequence[int] | None = None,
    ) -> tuple[list[Position], list[Position], float]:
        """Draw a swap node.

        Parameters
        ----------
        ax : plt.Axes
            Matplotlib axes
        x: float
            First coordinate of the diagram
        y: float
            Second coordinate of the diagram
        diagram: Swap | None
            The `Swap` instance being drawn. When its `void_input_port` is
            set, the diagonal/arrows for that port are omitted -- that
            port's whole path is known to dead-end in `VoidDiagram` filler
            (see `FusionRule`'s disguised-composition rewrite), so drawing
            it would only clutter the figure with an uninteresting wire.
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
            radius of the swap node.
        """
        if radius is None:
            radius = self.config.node_radius
        arrow_length = 2 * radius
        void_input_port = diagram.void_input_port if diagram is not None else None
        # Draw X shape -- line below connects input1<->output0 (skip when
        # `void_input_port == 1`), the one after connects input0<->output1
        # (skip when `void_input_port == 0`).
        if void_input_port != 1:
            ax.plot([x - radius, x + radius], [y - radius, y + radius], "k-", linewidth=self.config.wire_width)
        if void_input_port != 0:
            ax.plot([x - radius, x + radius], [y + radius, y - radius], "k-", linewidth=self.config.wire_width)
        # We will draw input and output wires depending of the block is part of
        # a composition diagram
        draw_in_wires = True
        draw_out_wires = True
        if (  # ruff: ignore[too-many-boolean-expressions]
            comp_idx is not None and comp_idx != -1
        ) or ((comp_idx == -1 or comp_idx is None) and (sub_comp_idx is not None and sub_comp_idx != -1)):
            draw_out_wires = False
        if draw_kept_inputs is None:
            draw_kept_inputs = [0, 1]
        if draw_kept_outputs is None:
            draw_kept_outputs = [0, 1]
        if void_input_port is not None:
            # The void-bound port's own input arrow, and the paired output
            # arrow its diagonal used to feed (input0<->output1,
            # input1<->output0), both lead nowhere interesting.
            draw_kept_inputs = [p for p in draw_kept_inputs if p != void_input_port]
            draw_kept_outputs = [p for p in draw_kept_outputs if p != 1 - void_input_port]
        # Draw inputs arrows
        init_input_positions = [
            (x + radius + arrow_length, y - radius),
            (x + radius + arrow_length, y + radius),
        ]
        if draw_in_wires:
            if input_positions is None:
                # input_positions is empty only for the first element of a composition
                # or a single Swap diagram
                input_positions = init_input_positions
            # A hack to not draw incoming arrows from void diagrams (mirrors
            # `_draw_proper_diagram`'s own `if input_pos:` guard): a
            # position can be `()` -- VoidDiagram's own drawing forces its
            # output positions to empty tuples -- even when `void_input_port`
            # itself wasn't set or doesn't line up, since which position
            # ends up void here depends on the *predecessor*'s own layout,
            # not anything this `Swap` (or its constructor) controls.
            if 0 in draw_kept_inputs and input_positions[0]:
                input1 = patches.FancyArrowPatch(
                    input_positions[0],
                    (x + radius, y - radius),
                    arrowstyle="->",
                    ec="black",
                    mutation_scale=20,
                    linewidth=self.config.wire_width,
                )
                ax.add_patch(input1)
            if 1 in draw_kept_inputs and input_positions[1]:
                input2 = patches.FancyArrowPatch(
                    input_positions[1],
                    (x + radius, y + radius),
                    arrowstyle="->",
                    ec="black",
                    mutation_scale=20,
                    linewidth=self.config.wire_width,
                )
                ax.add_patch(input2)

        # Draw Outputs
        # We return outputs from bottom to top
        output_positions = [
            (x - radius, y - radius),
            (x - radius, y + radius),
        ]
        if draw_out_wires:
            if 0 in draw_kept_outputs:
                output1 = patches.FancyArrowPatch(
                    output_positions[0],
                    (x - radius - arrow_length, y - radius),
                    arrowstyle="->",
                    ec="black",
                    mutation_scale=20,
                    linewidth=self.config.wire_width,
                )
                ax.add_patch(output1)
            if 1 in draw_kept_outputs:
                output2 = patches.FancyArrowPatch(
                    output_positions[1],
                    (x - radius - arrow_length, y + radius),
                    arrowstyle="->",
                    ec="black",
                    mutation_scale=20,
                    linewidth=self.config.wire_width,
                )
                ax.add_patch(output2)
        return output_positions, init_input_positions, radius

    def _draw_fourier(  # ruff: ignore[complex-structure, too-many-arguments, too-many-positional-arguments]
        self: Visualizer,
        ax: plt.Axes,
        x: float,
        y: float,
        diagram: Diagram,
        comp_idx: int | None = None,
        sub_comp_idx: int | None = None,
        input_positions: list[Position] | None = None,
        radius: float | None = None,
        draw_kept_inputs: Sequence[int] | None = None,
        draw_kept_outputs: Sequence[int] | None = None,
    ) -> tuple[list[Position], list[Position], float]:
        """Draw a Fourier node.

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
            radius of the Fourier node.
        """
        if radius is None:
            radius = self.config.node_radius
        arrow_length = 2 * radius
        symbol = "F"
        resize_symbol = 0.8
        if isinstance(diagram, FourierInv):
            symbol = "F†"
            ax.plot([x - radius, x - radius], [y - radius, y + radius], "k-", linewidth=self.config.wire_width)
            ax.plot([x - radius, x + radius], [y + radius, y], "k-", linewidth=self.config.wire_width)
            ax.plot([x - radius, x + radius], [y - radius, y], "k-", linewidth=self.config.wire_width)
            ax.text(x, y, symbol, ha="center", va="center", fontsize=resize_symbol * self.config.fontsize * radius)
        elif isinstance(diagram, Fourier2):
            symbol = "F²"
            ax.plot([x - radius, x], [y, y + radius], "k-", linewidth=self.config.wire_width)
            ax.plot([x - radius, x], [y, y - radius], "k-", linewidth=self.config.wire_width)
            ax.plot([x + radius, x], [y, y - radius], "k-", linewidth=self.config.wire_width)
            ax.plot([x + radius, x], [y, y + radius], "k-", linewidth=self.config.wire_width)
            ax.text(x, y, symbol, ha="center", va="center", fontsize=resize_symbol * self.config.fontsize * radius)
        else:
            ax.plot([x + radius, x + radius], [y - radius, y + radius], "k-", linewidth=self.config.wire_width)
            ax.plot([x + radius, x - radius], [y + radius, y], "k-", linewidth=self.config.wire_width)
            ax.plot([x + radius, x - radius], [y - radius, y], "k-", linewidth=self.config.wire_width)
            ax.text(x, y, symbol, ha="center", va="center", fontsize=resize_symbol * self.config.fontsize * radius)

        # We will draw input and output wires depending of the block is part of
        # a composition diagram
        draw_in_wires = True
        draw_out_wires = True
        if (  # ruff: ignore[too-many-boolean-expressions]
            comp_idx is not None and comp_idx != -1
        ) or ((comp_idx == -1 or comp_idx is None) and (sub_comp_idx is not None and sub_comp_idx != -1)):
            draw_out_wires = False
        if draw_kept_inputs is None:
            draw_kept_inputs = [0]
        if draw_kept_outputs is None:
            draw_kept_outputs = [0]
        # Draw inputs wires
        init_input_positions = [(x + radius + arrow_length, y)]
        if draw_in_wires:
            if input_positions is None:
                input_positions = init_input_positions
            # A hack to not draw incoming arrows from void diagrams -- mirrors
            # `_draw_proper_diagram`'s own `if input_pos:` guard, see `_draw_swap`.
            if 0 in draw_kept_inputs and input_positions[0]:
                input1 = patches.FancyArrowPatch(
                    input_positions[0],
                    (x + radius, y),
                    arrowstyle="->",
                    ec="black",
                    mutation_scale=20,
                    linewidth=self.config.wire_width,
                )
                ax.add_patch(input1)
        # Draw output wires
        output_positions = [(x - radius, y)]
        if draw_out_wires and 0 in draw_kept_outputs:
            output1 = patches.FancyArrowPatch(
                output_positions[0],
                (x - radius - arrow_length, y),
                arrowstyle="->",
                ec="black",
                mutation_scale=20,
                linewidth=self.config.wire_width,
            )
            ax.add_patch(output1)
        return output_positions, init_input_positions, radius

    def _draw_feedforward(self: Visualizer, ax: plt.Axes) -> None:
        """Draw the classical link corresponding to feedforwards.

        Parameters
        ----------
        ax : plt.Axes
            Matplotlib axes
        """
        for node_id in self.reg.feedforward_nodes:
            if self.graph.nodes[node_id]["feedforward"]:
                x2, y2 = self.graph.nodes[node_id]["pos"]
                r2 = self.graph.nodes[node_id]["radius"]
                for meas_node in self.graph.nodes[node_id]["measurement_ids"]:
                    if meas_node in self.graph.nodes:
                        x1, y1 = self.graph.nodes[meas_node]["pos"]
                        r1 = self.graph.nodes[meas_node]["radius"]
                        # Depending of the relative vertical position we change the
                        # edges of the arrow of the classical link
                        pos1 = (x1 + r1, y1 + r1)
                        pos2 = (x2 - r2, y2 - r2)
                        if y1 > y2:
                            pos1 = (x1 + r1, y1 - r1)
                            pos2 = (x2 - r2, y2 + r2)
                        arrow = patches.FancyArrowPatch(
                            pos1,
                            pos2,
                            arrowstyle="->",
                            linestyle="dashed",
                            ec="green",
                            mutation_scale=20,
                            linewidth=self.config.wire_width,
                        )
                        ax.add_patch(arrow)
