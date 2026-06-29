"""Visualization for CV ZX diagrams.

This module provides visualization capabilities for:
    - ProperDiagram (spiders, gates)
    - CompositionDiagram (sequential composition)
    - TensorDiagram (parallel composition)
    - ContractedDiagram (partial trace with connections)

The visualizer uses matplotlib to draw diagrams in a way that respects
the input/output wire ordering and connection indices.
"""

import random as r
from copy import deepcopy
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from mqc3.zx.base_gates import (
    CompositionDiagram,
    ContractedDiagram,
    Diagram,
    Fourier,
    Fourier2,
    FourierInv,
    ProperDiagram,
    PSpider,
    QSpider,
    ScalarDiagram,
    Swap,
    TensorDiagram,
    ZxPoly,
)


@dataclass
class VisualizerConfig:
    """Configuration for diagram visualization.

    Layout Rules (l(D) - horizontal space occupied by diagram D):
    ------------------------------------------------------------
    Rule 0 (Base cases):
        - If D is a ProperDiagram, then l(D) = 1.
        - If D is a CompositionDiagram where every child is a ProperDiagram,
          then l(D) = |D| (number of child diagrams).

    Rule 1 (TensorDiagram with only ProperDiagrams):
        - If D is a TensorDiagram and every child is a ProperDiagram,
          then l(D) = 1 (sub-diagrams are packed vertically, same horizontal space).

    Rule 2 (TensorDiagram containing CompositionDiagram):
        - If D is a TensorDiagram and there exists a child D' that is a CompositionDiagram,
          then l(D') = 1 (overriding Rule 0). Composition diagrams inside a tensor
          diagram are resized to occupy 1 horizontal unit.

    Rule 3 (General CompositionDiagram):
        - If D is a CompositionDiagram, then l(D) = sum_{child in D.diagrams} l(child).

    Spacing Formulas (derived, not configurable):
    --------------------------------------------
    arrow_length = 2 * node_radius
        The length of input/output wires extending from nodes.

    horizontal_spacing = node_radius * 2 + arrow_length * 2
        Horizontal spacing between sub-diagrams inside a composition diagram.
        This ensures proper separation for sequential placement left to right.

    vertical_spacing = node_radius * vertical_factor
        Vertical spacing between sub-diagrams inside a tensor diagram.
        vertical_factor must be > 2 to prevent overlap. Default is 2.2.

    Attributes:
    ----------
    node_radius : float
        Each proper diagram is drawn inside a square of radius node_radius.

    vertical_factor : float
        Factor for vertical spacing between sub-diagrams in a tensor diagram.
        Must be > 2. Default is 2.2.

    wire_width : float
        Width of wire lines (stroke thickness).

    fontsize : float
        Font size for text labels.

    colors : dict
        Color mapping for spider types (q, p, macronode, non_gaussian, etc.).
    """

    node_radius: float = 5
    vertical_factor: float = 3
    wire_width: float = 1.5
    fontsize: float = 10.0

    colors: dict = field(
        default_factory=lambda: {
            "q": "lightgreen",
            "p": "lightcoral",
            "macronode": "lightblue",
            "non_gaussian": "gold",
            "contraction_out": "gray",
            "contraction_in": "violet",
            "boundary": "white",
            "default": "white",
        }
    )

    def __post_init__(self) -> None:
        """Validate parameters and compute derived spacing values.

        Raises:
        ------
        ValueError:
            If vertical_factor <= 2.
        """
        if self.vertical_factor <= 2:  # noqa: PLR2004
            msg = f"vertical_factor must be > 2, got {self.vertical_factor}"
            raise ValueError(msg)

        # Derived spacing values (strict formulas)
        self.arrow_length = 2 * self.node_radius
        self.horizontal_spacing = self.node_radius * 2 + self.arrow_length * 2
        self.vertical_spacing = self.node_radius * self.vertical_factor

    def get_horizontal_space(self, diagram: Diagram) -> int:  # noqa: PLR0911
        """Compute the horizontal space l(D) occupied by a diagram.

        This implements the layout rules defined in the class docstring.

        Parameters
        ----------
        diagram : Diagram
            The diagram to measure.

        Returns:
        -------
        int
            Horizontal space occupied by the diagram (in units).
        """
        # Rule 0: ProperDiagram
        if isinstance(diagram, ProperDiagram):
            return 1

        # Rule 1 & 2: TensorDiagram
        if isinstance(diagram, TensorDiagram):
            # Check if any child is a CompositionDiagram (Rule 2)
            for child in diagram.diagrams:
                if isinstance(child, CompositionDiagram):
                    return 1
            # Rule 1: All children are ProperDiagrams -> l(D) = 1
            return 1

        # Rule 0 & 3: CompositionDiagram
        if isinstance(diagram, CompositionDiagram):
            total = 0
            for child in diagram.diagrams:
                total += self.get_horizontal_space(child)
            return total

        # ContractedDiagram: treat as atomic (horizontal space = 1)
        if isinstance(diagram, ContractedDiagram):
            return 1

        # ScalarDiagram: no horizontal space
        if isinstance(diagram, ScalarDiagram):
            return 0

        # Default fallback
        return 1


class DiagramVisualizer:
    """Visualizer for CV ZX diagrams.

    This class handles layout and drawing of various diagram types,
    ensuring proper wire connections and index ordering.
    """

    def __init__(self, config: VisualizerConfig | None = None) -> None:
        """Initialize visualizer with configuration."""
        self.config = config or VisualizerConfig()

    def visualize(self, diagram: Diagram, title: str = "") -> plt.Figure:
        """Visualize a diagram.

        Parameters:
        ----------
        diagram : Diagram
            The diagram to visualize.
        title : str
            Title for the figure.

        Returns:
        -------
        plt.Figure
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_aspect("equal")
        ax.axis("off")

        if title:
            ax.set_title(title, fontsize=self.config.fontsize)

        if isinstance(diagram, ProperDiagram):
            self._draw_proper_diagram(ax, diagram)
        elif isinstance(diagram, CompositionDiagram):
            self._draw_composition(ax, diagram)
        elif isinstance(diagram, TensorDiagram):
            self._draw_tensor(ax, diagram)
        elif isinstance(diagram, ContractedDiagram):
            self._draw_contracted(ax, diagram)
        elif isinstance(diagram, ScalarDiagram):
            self._draw_scalar(ax)
        else:
            ax.text(
                0.5, 0.5, f"Unknown diagram type: {type(diagram)}", ha="center", va="center", transform=ax.transAxes
            )

        plt.tight_layout()
        return fig

    def _draw_proper_diagram(  # noqa: C901, PLR0912, PLR0913, PLR0914, PLR0915, PLR0917
        self,
        ax: plt.Axes,
        diagram: ProperDiagram,
        x: float = 0,
        y: float = 0,
        comp_idx: int | None = None,
        sub_comp_idx: int | None = None,
        input_positions: list | None = None,
        radius: float | None = None,
        kept_inputs: list | None = None,
        kept_outputs: list | None = None,
    ) -> tuple[list[float], list[float] | None, float | list[float]]:
        """Draw a proper diagram (single node).

        Parameters:
        ----------
        ax : plt.Axes
            Matplotlib axes
        diagram : Diagram
            The sub-diagram to draw
        x: float
            First coordinate of the diagram
        y: float
            Second coordinate of the diagram
        comp_idx: int | None = None
            Index of a sub-diagram inside a composition. In a composition,
            we don't need to draw the input/output wires of all sub-diagrams.
        sub_comp_idx: int | None = None
            Index of a proper diagram inside a sub-composition diagram. Along with
            the comp_idx, their value determine whether to draw input and/or output
            wires on the proper diagram.
        input_positions: list | None
            List of positions received from the sub-diagram that precedes the
            current diagram in a composition diagram. It will help to draw
            the input wires from that sub-diagram to the current diagram.
            In case list is empty, it means the sub-diagram is not part of a
            composition diagram.
        radius: float | None
            This is the variable used to find any other meaningful parameter:
            arrow_length, horizontal span and the horizontal spacing. So it is
            the unit variable of the diagram.

        Returns:
        -------
        list[tuple[float]]
            Lists of output indices that will serve as the input indices of the
            next diagram if there is any.
        list[tuple[float]] | None
            Lists of proper input indices, i.e. different from input indices
            received from a sub-diagram.
        float | list[float]
            radius of either the proper diagram, of the first diagram of a
            composition diagram or the list of radiuses of a tensor diagram.
        """
        output_positions = []
        # Determine spider type
        # if isinstance(diagram, (MacronodeSpider, NonGaussianSpider, QSpider, PSpider)):
        if isinstance(diagram, (QSpider, PSpider)):
            if radius is None:
                radius = self.config.node_radius
            arrow_length = 2 * radius
            spider_type = self._get_spider_type(diagram)
            color = self.config.colors.get(spider_type, self.config.colors["default"])

            # Draw node
            pivot = (x - radius, y - radius)
            width = 2 * radius
            height = 2 * radius
            box = patches.Rectangle(pivot, width, height, facecolor=color)
            ax.add_patch(box)

            # Draw phase if present
            if hasattr(diagram, "phase") and diagram.phase and not diagram.phase.is_zero():
                phase_str = self._format_phase(diagram.phase)
                ax.text(
                    x,
                    y,
                    phase_str,
                    ha="center",
                    va="top",
                    fontsize=self.config.fontsize,
                )
            # We will draw input and output wires depending of the block is part of
            # a composition diagram
            draw_in_wires = True
            draw_out_wires = True
            if (  # noqa: PLR0916
                comp_idx is not None and comp_idx != -1
            ) or ((comp_idx == -1 or comp_idx is None) and (sub_comp_idx is not None and sub_comp_idx != -1)):
                draw_out_wires = False
            if kept_inputs is None:
                kept_inputs = range(diagram.num_inputs)
            if kept_outputs is None:
                kept_outputs = range(diagram.num_outputs)
            # Draw input wires (left side)
            y_offset = list(np.linspace(0, height, diagram.num_inputs + 2))
            # Remove the two edges
            y_offset.pop(0)
            y_offset.pop(-1)
            init_input_positions = [
                (pivot[0] + width + arrow_length, pivot[1] + y_offset[i]) for i in range(diagram.num_inputs)
            ]
            if draw_in_wires:
                if input_positions is None:
                    # input_positions is empty only for the first element of a composition
                    # or a single proper diagram
                    input_positions = init_input_positions
                for i in range(diagram.num_inputs):
                    if i in kept_inputs:
                        if sub_comp_idx is not None:
                            input_pos = (
                                input_positions[i]
                                if (sub_comp_idx == 0 and (comp_idx is None or comp_idx == 0))
                                else input_positions[kept_inputs.index(i)]
                            )
                        else:
                            input_pos = (
                                input_positions[i]
                                if (comp_idx is None or comp_idx == 0)
                                else input_positions[kept_inputs.index(i)]
                            )
                        input_i = patches.FancyArrowPatch(
                            input_pos,
                            (pivot[0] + arrow_length, pivot[1] + y_offset[i]),
                            arrowstyle="->",
                            ec="black",
                            mutation_scale=20,
                            linewidth=self.config.wire_width,
                        )
                        ax.add_patch(input_i)

            # Draw output wires (right side)
            y_offset = list(np.linspace(0, height, diagram.num_outputs + 2))
            # Remove the two edges
            y_offset.pop(0)
            y_offset.pop(-1)
            output_positions = [(pivot[0], pivot[1] + y_offset[i]) for i in range(diagram.num_outputs)]
            if draw_out_wires:
                for i in range(diagram.num_outputs):
                    if i in kept_outputs:
                        output_i = patches.FancyArrowPatch(
                            output_positions[i],
                            (pivot[0] - arrow_length, pivot[1] + y_offset[i]),
                            arrowstyle="->",
                            ec="black",
                            mutation_scale=20,
                            linewidth=self.config.wire_width,
                        )
                        ax.add_patch(output_i)
                ax.plot()
        elif isinstance(diagram, Swap):
            output_positions, init_input_positions, radius = self._draw_swap(
                ax, x, y, comp_idx, sub_comp_idx, input_positions, radius, kept_inputs, kept_outputs
            )  # TODO: add the kept_input management
        elif isinstance(diagram, (Fourier, FourierInv, Fourier2)):
            output_positions, init_input_positions, radius = self._draw_fourier(
                ax, x, y, diagram, comp_idx, sub_comp_idx, input_positions, radius, kept_inputs, kept_outputs
            )  # TODO: update the sub_comp_idx and add the kept_input management
        return output_positions, init_input_positions, radius

    def _draw_composition(  # noqa: C901, PLR0912, PLR0913, PLR0914, PLR0917
        self,
        ax: plt.Axes,
        diagram: CompositionDiagram,
        x: float = 0,
        y: float = 0,
        comp_idx: int | None = None,
        sub_comp_idx: int | None = None,
        is_sub_comp: bool = False,  # noqa: FBT001, FBT002
        input_positions: list | None = None,
        radius: float | None = None,
        kept_inputs: list | None = None,
        kept_outputs: list | None = None,
    ) -> tuple[list[float], list[float] | None, float | list[float]]:
        """Draw a composition diagram.

        Parameters:
        ----------
        ax : plt.Axes
            Matplotlib axes
        diagram : Diagram
            The sub-diagram to draw
        x: float
            First coordinate of the diagram
        y: float
            Second coordinate of the diagram
        comp_idx: int | None = None
            Index of a sub-diagram inside a composition. In a composition,
            we don't need to draw the input/output wires of all sub-diagrams.
        sub_comp_idx: int | None = None
            Index of a proper diagram inside a sub-composition diagram. Along with
            the comp_idx, their value determine whether to draw input and/or output
            wires on the proper diagram.
        input_positions: list | None
            List of positions received from the sub-diagram that precedes the
            current diagram in a composition diagram. It will help to draw
            the input wires from that sub-diagram to the current diagram.
            In case list is empty, it means the sub-diagram is not part of a
            composition diagram.
        is_sub_comp: bool
            It is true when a composition is a sub-diagram.
        radius: float | None
            This is the variable used to find any other meaningful parameter:
            arrow_length, horizontal span and the horizontal spacing. So it is
            the unit variable of the diagram.

        Returns:
        -------
        list[tuple[float]]
            Lists of output indices that will serve as the input indices of the
            next diagram if there is any.
        list[tuple[float]] | None
            Lists of proper input indices, i.e. different from input indices
            received from a sub-diagram.
        float | list[float]
            radius of either the proper diagram, of the first diagram of a
            composition diagram or the list of radiuses of a tensor diagram.
        """
        n = len(diagram.diagrams)
        if n == 0:
            return None

        if radius is None:
            radius = self.config.node_radius
        if kept_inputs is None:
            kept_inputs = range(diagram.num_inputs)
            kept_outputs = range(diagram.num_outputs)
        output_radius = radius
        # Draw each sub-diagram at its position
        # If is_sub_comp is True means that we are inside a composition block
        # which is part of a tensor product therefore we must resize the
        # sub_hor_span, spacing and radius
        if is_sub_comp:
            cs = len(diagram.diagrams)
            sub_hor_span = 6 * radius / cs if comp_idx != -1 and comp_idx is not None else 18 * radius / (2 * cs + 1)
        else:
            sub_hor_span = 6 * radius
        sub_radius = sub_hor_span / 6
        sub_hor_spacing = 4 * sub_radius
        sub_x = x + 3 * (radius - sub_radius)
        # Calculate positions for each sub-diagram
        # Composition is drawn left to right
        start_x = 0
        output_positions = input_positions
        init_input_positions = []
        for i, sub_diagram in enumerate(diagram.diagrams):
            d = start_x - i * sub_hor_spacing
            idx = i
            if is_sub_comp:
                sub_comp_idx = idx
            else:
                comp_idx = idx
            if i == 0:
                sent_kept_inputs = kept_inputs
                sent_kept_outputs = range(diagram.num_outputs)
            elif i == n - 1:
                sent_kept_inputs = range(diagram.num_inputs)
                sent_kept_outputs = kept_outputs
                if is_sub_comp:
                    sub_comp_idx = -1
                else:
                    comp_idx = -1
            else:
                sent_kept_inputs = range(diagram.num_inputs)
                sent_kept_outputs = range(diagram.num_outputs)
            output_positions, init_input_pos, radius = self._draw_sub_diagram(
                ax,
                sub_diagram,
                sub_x + d,
                y,
                comp_idx=comp_idx,
                sub_comp_idx=sub_comp_idx,
                is_sub_comp=is_sub_comp,
                input_positions=output_positions,
                radius=sub_radius,
                kept_inputs=sent_kept_inputs,
                kept_outputs=sent_kept_outputs,
            )
            # The input positions of a composition diagram is
            # the input positions of its first sub-diagram
            if i == 0:
                init_input_positions = init_input_pos
                output_radius = radius
        return output_positions, init_input_positions, output_radius

    def _draw_tensor(  # noqa: C901, PLR0912, PLR0913, PLR0914, PLR0915, PLR0917
        self,
        ax: plt.Axes,
        diagram: TensorDiagram,
        x: float = 0,
        y: float = 0,
        comp_idx: int | None = None,
        sub_comp_idx: int | None = None,
        is_sub_comp: bool = False,  # noqa: FBT001, FBT002
        input_positions: list | None = None,
        radius: float | None = None,
        kept_inputs: list | None = None,
        kept_outputs: list | None = None,
    ) -> tuple[list[float], list[float] | None, float | list[float]]:
        """Draw a tensor diagram (parallel).

        Parameters:
        ----------
        ax : plt.Axes
            Matplotlib axes
        diagram : Diagram
            The sub-diagram to draw
        x: float
            First coordinate of the diagram
        y: float
            Second coordinate of the diagram
        comp_idx: int | None = None
            Index of a sub-diagram inside a composition. In a composition,
            we don't need to draw the input/output wires of all sub-diagrams.
        sub_comp_idx: int | None = None
            Index of a proper diagram inside a sub-composition diagram. Along with
            the comp_idx, their value determine whether to draw input and/or output
            wires on the proper diagram.
        is_sub_comp: bool
            It is true when a composition is a sub-diagram.
        input_positions: list | None
            List of positions received from the sub-diagram that precedes the
            current diagram in a composition diagram. It will help to draw
            the input wires from that sub-diagram to the current diagram.
            In case list is empty, it means the sub-diagram is not part of a
            composition diagram.
        radius: float | None
            This is the variable used to find any other meaningful parameter:
            arrow_length, horizontal span and the horizontal spacing. So it is
            the unit variable of the diagram.

        Returns:
        -------
        list[tuple[float]]
            Lists of output indices that will serve as the input indices of the
            next diagram if there is any.
        list[tuple[float]] | None
            Lists of proper input indices, i.e. different from input indices
            received from a sub-diagram.
        float | list[float]
            radius of either the proper diagram, of the first diagram of a
            composition diagram or the list of radiuses of a tensor diagram.
        """
        n = len(diagram.diagrams)
        if n == 0:
            return []

        # Calculate positions for each sub-diagram
        # Tensor is drawn top to bottom
        output_positions = []
        init_input_positions = []
        output_radius = []
        inp_ind = 0
        out_ind = 0
        if radius is None:
            radius = self.config.node_radius
            vertical_spacing = self.config.vertical_spacing
        else:
            vertical_spacing = self.config.vertical_factor * radius
        # input_positions are from the bottom to the top and since
        # we draw from top to bottom in draw_tensor we must reverse
        # input_positions
        if input_positions is not None:
            input_positions.reverse()
        if kept_inputs is not None:
            num_inputs = diagram.num_inputs
            kept_inputs = [num_inputs - i - 1 for i in kept_inputs]
            kept_inputs.sort()
            num_outputs = diagram.num_outputs
            kept_outputs = [num_outputs - i - 1 for i in kept_outputs]
            kept_outputs.sort()
        contract_shift = 0
        h = vertical_spacing
        for sub_diagram in diagram.diagrams:
            is_sub_comp = isinstance(sub_diagram, CompositionDiagram)
            if contract_shift != 0:
                h -= contract_shift
            else:
                h -= vertical_spacing
            # Draw sub-diagram with local coordinates
            sub_input_positions = None
            sub_kept_inputs = None
            sub_kept_outputs = None
            if input_positions is not None:
                sub_input_positions = input_positions[inp_ind : inp_ind + sub_diagram.num_inputs]
                # We must reverse back sub_input_positions
                sub_input_positions.reverse()
            if kept_inputs is not None:
                sub_kept_inputs = []
                for ind in range(inp_ind, inp_ind + sub_diagram.num_inputs):
                    if ind in kept_inputs:
                        sub_kept_inputs.append(ind - inp_ind)
                sub_kept_outputs = []
                for ind in range(out_ind, out_ind + sub_diagram.num_outputs):
                    if ind in kept_outputs:
                        sub_kept_outputs.append(ind - out_ind)
                # We must reverse back the lits above
                sub_kept_inputs = [sub_diagram.num_inputs - p - 1 for p in sub_kept_inputs]
                sub_kept_outputs = [sub_diagram.num_outputs - p - 1 for p in sub_kept_outputs]
                sub_kept_inputs.sort()
                sub_kept_outputs.sort()
            # Find local_kept_inputs and local_kept_outputs
            output_pos, init_input_pos, radius = self._draw_sub_diagram(
                ax,
                sub_diagram,
                x,
                y + h,
                comp_idx=comp_idx,
                sub_comp_idx=sub_comp_idx,
                is_sub_comp=is_sub_comp,
                input_positions=sub_input_positions,
                radius=radius,
                kept_outputs=sub_kept_outputs,
                kept_inputs=sub_kept_inputs,
            )
            if isinstance(sub_diagram, ContractedDiagram):
                contract_shift = radius[1]
                radius = radius[0]
            else:
                contract_shift = 0
            output_positions = output_pos + output_positions
            init_input_positions = init_input_pos + init_input_positions
            output_radius = [radius, *output_radius]
            inp_ind += sub_diagram.num_inputs
            out_ind += sub_diagram.num_outputs
        return output_positions, init_input_positions, output_radius

    def _draw_contracted(  # noqa: C901, PLR0912, PLR0913, PLR0914, PLR0915, PLR0917
        self,
        ax: plt.Axes,
        diagram: ContractedDiagram,
        x: float = 0,
        y: float = 0,
        comp_idx: int | None = None,
        sub_comp_idx: int | None = None,
        is_sub_comp: int | None = None,
        input_positions: list | None = None,
        radius: float | None = None,
    ) -> tuple[list[float], list[float] | None, float | list[float]]:
        """Draw a contracted diagram (feedback connections).

        Special care is needed because:
            - First diagram (D1) is drawn
            - Second diagram (D2) is drawn
            - Forward connections (I1→I2) are drawn as wires
            - Feedback connections (J2→J1) are drawn as wires
            - External wires use kept_first_inputs, kept_first_outputs,
              kept_second_inputs, kept_second_outputs

        Parameters:
        ----------
        ax : plt.Axes
            Matplotlib axes
        diagram : Diagram
            The sub-diagram to draw
        x: float
            First coordinate of the diagram
        y: float
            Second coordinate of the diagram
        comp_idx: int | None = None
            Index of a sub-diagram inside a composition. In a composition,
            we don't need to draw the input/output wires of all sub-diagrams.
        sub_comp_idx: int | None = None
            Index of a proper diagram inside a sub-composition diagram. Along with
            the comp_idx, their value determine whether to draw input and/or output
            wires on the proper diagram.
        is_sub_comp: bool
            It is true when a composition is a sub-diagram.
        input_positions: list | None
            List of positions received from the sub-diagram that precedes the
            current diagram in a composition diagram. It will help to draw
            the input wires from that sub-diagram to the current diagram.
            In case list is empty, it means the sub-diagram is not part of a
            composition diagram.
        radius: float | None
            This is the variable used to find any other meaningful parameter:
            arrow_length, horizontal span and the horizontal spacing. So it is
            the unit variable of the diagram.

        Returns:
        -------
        list[tuple[float]]
            Lists of output indices that will serve as the input indices of the
            next diagram if there is any.
        list[tuple[float]] | None
            Lists of proper input indices, i.e. different from input indices
            received from a sub-diagram.
        tuple[float, float]
            radius of either the proper diagram, of the first diagram of a
            composition diagram or the list of radiuses of a tensor diagram.
            And the shift of the contracted diagram. This shift will allow
            to draw a contracted diagram properly when it is inside a tensor
            diagram
        """
        if radius is None:
            radius = self.config.node_radius
        # Draw second diagram (D2)
        is_sub_comp = isinstance(diagram.second, CompositionDiagram)
        num_d2_out_wires = len(diagram.kept_second_outputs) + len(diagram.J2)
        num_d2_input_wires = len(diagram.kept_second_inputs) + len(diagram.I2)
        draw_kept_second_inputs = [num_d2_input_wires - i - 1 for i in diagram.kept_second_inputs]
        draw_kept_second_inputs.sort()
        draw_kept_second_outputs = [num_d2_out_wires - i - 1 for i in diagram.kept_second_outputs]
        draw_kept_second_outputs.sort()
        # We do not need to reverse input_positions like in draw_tensor because
        # We draw from bottom to top
        sec_in_positions = input_positions[: len(draw_kept_second_inputs)] if input_positions is not None else None
        y_shift1 = 0
        y_shift2 = 0
        if isinstance(diagram.second, TensorDiagram):
            y_shift2 += (len(diagram.second.diagrams) - 1) * (self.config.vertical_spacing)
        if isinstance(diagram.first, TensorDiagram):
            y_shift1 += (len(diagram.first.diagrams) - 1) * (self.config.vertical_spacing)
        spacing = 2 * self.config.vertical_spacing
        y2 = y - spacing - y_shift1
        x2 = x
        output_positions_2, input_positions_2, output_radius_2 = self._draw_sub_diagram(
            ax,
            diagram.second,
            x2,
            y2,
            comp_idx=comp_idx,
            sub_comp_idx=sub_comp_idx,
            is_sub_comp=is_sub_comp,
            input_positions=sec_in_positions,
            radius=radius,
            kept_inputs=draw_kept_second_inputs,
            kept_outputs=draw_kept_second_outputs,
        )

        # Draw first diagram (D1)
        is_sub_comp = isinstance(diagram.first, CompositionDiagram)
        num_d1_out_wires = len(diagram.kept_first_outputs) + len(diagram.I1)
        num_d1_input_wires = len(diagram.kept_first_inputs) + len(diagram.J1)
        # Wires are drawn from bottom to top
        draw_kept_first_inputs = [num_d1_input_wires - i - 1 for i in diagram.kept_first_inputs]
        draw_kept_first_inputs.sort()
        draw_kept_first_outputs = [num_d1_out_wires - i - 1 for i in diagram.kept_first_outputs]
        draw_kept_first_outputs.sort()
        fir_in_positions = input_positions[len(draw_kept_second_inputs) :] if input_positions is not None else None
        x1, y1 = x, y
        output_positions_1, input_positions_1, output_radius_1 = self._draw_sub_diagram(
            ax,
            diagram.first,
            x1,
            y1,
            comp_idx=comp_idx,
            sub_comp_idx=sub_comp_idx,
            is_sub_comp=is_sub_comp,
            input_positions=fir_in_positions,
            radius=radius,
            kept_inputs=draw_kept_first_inputs,
            kept_outputs=draw_kept_first_outputs,
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
        J_dict = {diagram.J2[i]: diagram.J1[i] for i in range(len(diagram.J1))}
        I_dict = {diagram.I1[i]: diagram.I2[i] for i in range(len(diagram.I2))}
        # Let's compute the distance between D1 and D2
        box_dist = spacing - 2 * radius
        # Draw feedback connections (J2 → J1)
        # J2: indices of outputs from second diagram
        # J1: indices of inputs from first diagram
        # Connection order: J2[k] connects to J1[k]
        # First we draw incoming arrows J2[k]s of D2
        second_diagrams = (
            deepcopy(diagram.second.diagrams) if isinstance(diagram.second, TensorDiagram) else [diagram.second]
        )
        second_diagrams.reverse()
        k = 0
        J2_info = {}  # noqa: N806
        for j, sub_diagram in enumerate(second_diagrams):
            x_offset_2_i = list(np.linspace(0, 2 * new_output_radius_2[j], sub_diagram.num_outputs + 2))
            x_offset_2_i.pop(0)
            x_offset_2_i.pop(1)
            sub_output_positions_2 = output_positions_2[k : k + sub_diagram.num_outputs]
            sub_J2 = []  # noqa: N806
            sub_J2_dict = {}  # noqa: N806
            # This is to allow the arrow to appear when when there is a Swap
            padding = isinstance(sub_diagram, Swap)
            padding_coef = 0.4
            for ind in range(k, k + sub_diagram.num_outputs):
                if num_d2_out_wires - ind - 1 in diagram.J2:
                    sub_J2.append(ind - k)
                    sub_J2_dict[ind - k] = num_d2_out_wires - ind - 1
            # We draw from bottom to top
            x_offset_2_i.reverse()
            J2_points = {}  # noqa: N806
            for i in sub_J2:
                # We draw from down to top
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
        first_diagrams = (
            deepcopy(diagram.first.diagrams) if isinstance(diagram.first, TensorDiagram) else [diagram.first]
        )
        first_diagrams.reverse()
        k = 0
        J1_info = {}  # noqa: N806
        J1_sub_diag_mapping = {}  # noqa: N806
        for j, sub_diagram in enumerate(first_diagrams):
            x_offset_1_i = list(np.linspace(0, 2 * new_output_radius_1[j], sub_diagram.num_inputs + 2))
            x_offset_1_i.pop(0)
            x_offset_1_i.pop(1)
            sub_input_positions_1 = input_positions_1[k : k + sub_diagram.num_inputs]
            sub_J1 = []  # noqa: N806
            sub_J1_dict = {}  # noqa: N806
            sub_J1_dict_inv = {}  # noqa: N806
            for ind in range(k, k + sub_diagram.num_inputs):
                if num_d1_input_wires - ind - 1 in diagram.J1:
                    sub_J1.append(ind - k)
                    sub_J1_dict[ind - k] = num_d1_input_wires - ind - 1
                    sub_J1_dict_inv[num_d1_input_wires - ind - 1] = ind - k
                    J1_sub_diag_mapping[num_d1_input_wires - ind - 1] = j
            # We draw from bottom to top
            J1_points = {}  # noqa: N806
            padding = isinstance(sub_diagram, Swap)
            padding_coef = 0.4
            for i in sub_J1:
                # We draw from down to top
                fir_point_i = sub_input_positions_1[i]
                # We move fir_point_i to the end of the arrow
                fir_point_i = (fir_point_i[0] - 2 * output_radius_1[j], fir_point_i[1])
                ax.plot(
                    [fir_point_i[0] + x_offset_1_i[i], fir_point_i[0] + x_offset_1_i[i]],
                    [
                        y1
                        - self.config.vertical_spacing * (len(first_diagrams) - 1)
                        - output_radius_1[-1] * (1 + padding_coef * padding),
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
        y_offset_J = list(np.linspace(0, box_dist / 2, len(diagram.J2) + 2))
        y_offset_J.pop(0)
        y_offset_J.pop(-1)
        for j, sub_diagram in enumerate(second_diagrams):
            sub_J2 = J2_info[j][0]
            sub_J2_dict = J2_info[j][1]
            J2_points = J2_info[j][2]
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
                sub_J1_info = J1_info[fir_sub_diag_ind]
                sub_J1_ind = sub_J1_info[1][1][J_dict[sub_J2_dict[i]]]
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
        I1_info = {}  # noqa: N806
        I1_sub_diag_mapping = {}  # noqa: N806
        k = 0
        for j, sub_diagram in enumerate(first_diagrams):
            x_offset_1_i = list(np.linspace(0, 2 * new_output_radius_1[j], sub_diagram.num_outputs + 2))
            x_offset_1_i.pop(0)
            x_offset_1_i.pop(1)
            sub_output_positions_1 = output_positions_1[k : k + sub_diagram.num_outputs]
            sub_I1 = []  # noqa: N806
            sub_I1_dict = {}  # noqa: N806
            for ind in range(k, k + sub_diagram.num_outputs):
                if num_d1_out_wires - ind - 1 in diagram.I1:
                    sub_I1.append(ind - k)
                    sub_I1_dict[ind - k] = num_d1_out_wires - ind - 1
                    I1_sub_diag_mapping[num_d1_out_wires - ind - 1] = j
            # This is to allow the arrow to appear when when there is a Swap
            padding = isinstance(sub_diagram, Swap)
            padding_coef = 0.4
            # We draw from bottom to top
            I1_points = {}  # noqa: N806
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
                        y1
                        - self.config.vertical_spacing * (len(first_diagrams) - 1)
                        - output_radius_1[-1] * (1 + padding_coef * padding),
                    ),
                    arrowstyle="->",
                    ec=color_out,
                    mutation_scale=20,
                    linewidth=self.config.wire_width,
                )
                I1_points[i] = (
                    fir_point_i[0] - x_offset_1_i[i],
                    y1
                    - self.config.vertical_spacing * (len(first_diagrams) - 1)
                    - output_radius_1[-1] * (1 + padding_coef * padding),
                )
                ax.add_patch(fir_arrow_i_1)
            I1_info[j] = (sub_I1, sub_I1_dict, I1_points)
            k += sub_diagram.num_outputs
        # Then we draw arrows I2[k]
        I2_info = {}  # noqa: N806
        I2_sub_diag_mapping = {}  # noqa: N806
        k = 0
        for j, sub_diagram in enumerate(second_diagrams):
            x_offset_2_i = list(np.linspace(0, 2 * new_output_radius_2[j], sub_diagram.num_inputs + 2))
            x_offset_2_i.pop(0)
            x_offset_2_i.pop(1)
            sub_input_positions_2 = input_positions_2[k : k + sub_diagram.num_inputs]
            sub_I2 = []  # noqa: N806
            sub_I2_dict = {}  # noqa: N806
            sub_I2_dict_inv = {}  # noqa: N806
            for ind in range(k, k + sub_diagram.num_inputs):
                if num_d2_input_wires - ind - 1 in diagram.I2:
                    sub_I2.append(ind - k)
                    sub_I2_dict[ind - k] = num_d2_input_wires - ind - 1
                    sub_I2_dict_inv[num_d2_input_wires - ind - 1] = ind - k
                    I2_sub_diag_mapping[num_d2_input_wires - ind - 1] = j
            padding = isinstance(sub_diagram, Swap)
            padding_coef = 0.4
            # We draw from bottom to top
            I2_points = {}  # noqa: N806
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
        y_offset_I = list(np.linspace(0, box_dist / 2, len(diagram.I1) + 2))
        y_offset_I.pop(0)
        y_offset_I.pop(-1)
        k = 0
        for j, sub_diagram in enumerate(first_diagrams):
            sub_I1 = I1_info[j][0]
            sub_I1_dict = I1_info[j][1]
            I1_points = I1_info[j][2]
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
                sub_I2_info = I2_info[sec_sub_diag_ind]
                sub_I2_ind = sub_I2_info[1][1][I_dict[sub_I1_dict[i]]]
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
        output_positions = [output_positions_2[i] for i in draw_kept_second_outputs]
        output_positions += [output_positions_1[i] for i in draw_kept_first_outputs]
        # We also return the initial input positions
        init_input_positions = [input_positions_2[i] for i in draw_kept_second_inputs]
        init_input_positions += [input_positions_1[i] for i in draw_kept_first_inputs]
        return output_positions, init_input_positions, (radius, y_shift1 + y_shift2 + 1.5 * spacing)

    def _draw_sub_diagram(  # noqa: PLR0913, PLR0917
        self,
        ax: plt.Axes,
        diagram: Diagram,
        x: float,
        y: float,
        comp_idx: int | None = None,
        sub_comp_idx: int | None = None,
        is_sub_comp: bool = False,  # noqa: FBT001, FBT002
        input_positions: list | None = None,
        radius: float | None = None,
        kept_inputs: list | None = None,
        kept_outputs: list | None = None,
    ) -> tuple[list[float], list[float] | None, float | list[float]]:
        """Draw a sub-diagram at specified coordinates and return port positions.

        Parameters:
        ----------
        ax : plt.Axes
            Matplotlib axes
        diagram : Diagram
            The sub-diagram to draw
        x: float
            First coordinate of the sub-diagram
        y: float
            Second coordinate of the sub-diagram
        comp_idx: int | None = None
            Index of a sub-diagram inside a composition. In a composition,
            we don't need to draw the input/output wires of all sub-diagrams.
        sub_comp_idx: int | None = None
            Index of a proper diagram inside a sub-composition diagram. Along with
            the comp_idx, their value determine whether to draw input and/or output
            wires on the proper diagram.
        is_sub_comp: bool = False
            This variable is not None when a composition is part of a tensor
            diagram. And in that case we need to resize this composition
            diagram so that it fits in the horizontal spacing of the tensor
            diagram.
        input_positions: list | None
            List of positions received from the sub-diagram that precedes the
            current diagram in a composition diagram. It will help to draw
            the input wires from that sub-diagram to the current diagram.
            In case list is empty, it means the sub-diagram is not part of a
            composition diagram.
        radius: float | None
            This is the variable used to find any other meaningful parameter:
            arrow_length, horizontal span and the horizontal spacing. So it is
            the unit variable of the diagram.
        ...

        Returns:
        -------
        list[tuple[float]]
            Lists of output indices that will serve as the input indices of the
            next diagram if there is any.
        list[tuple[float]] | None
            Lists of proper input indices, i.e. different from input indices
            received from a sub-diagram.
        float | list[float]
            radius of either the proper diagram, of the first diagram of a
            composition diagram or the list of radiuses of a tensor diagram.
        """
        if radius is None:
            radius = self.config.node_radius
        if isinstance(diagram, ProperDiagram):
            return self._draw_proper_diagram(
                ax, diagram, x, y, comp_idx, sub_comp_idx, input_positions, radius, kept_inputs, kept_outputs
            )
        if isinstance(diagram, CompositionDiagram):
            return self._draw_composition(
                ax,
                diagram,
                x,
                y,
                comp_idx,
                sub_comp_idx,
                is_sub_comp,
                input_positions,
                radius,
                kept_inputs,
                kept_outputs,
            )
        if isinstance(diagram, TensorDiagram):
            return self._draw_tensor(
                ax,
                diagram,
                x,
                y,
                comp_idx,
                sub_comp_idx,
                is_sub_comp,
                input_positions,
                radius,
                kept_inputs,
                kept_outputs,
            )
        if isinstance(diagram, ContractedDiagram):
            return self._draw_contracted(
                ax,
                diagram,
                x,
                y,
                comp_idx,
                sub_comp_idx,
                is_sub_comp,
                input_positions,
                radius,
            )
        return [], [], radius

    def _draw_swap(  # noqa: C901, PLR0913, PLR0917
        self,
        ax: plt.Axes,
        x: float,
        y: float,
        comp_idx: int | None = None,
        sub_comp_idx: int | None = None,
        input_positions: list | None = None,
        radius: float | None = None,
        kept_inputs: list | None = None,
        kept_outputs: list | None = None,
    ) -> tuple[list[float], list[float] | None, float | list[float]]:
        """Draw a swap node.

        Parameters:
        ----------
        ax : plt.Axes
            Matplotlib axes
        diagram : Diagram
            The sub-diagram to draw
        x: float
            First coordinate of the diagram
        y: float
            Second coordinate of the diagram
        comp_idx: int | None = None
            Index of a sub-diagram inside a composition. In a composition,
            we don't need to draw the input/output wires of all sub-diagrams.
        sub_comp_idx: int | None = None
            Index of a proper diagram inside a sub-composition diagram. Along with
            the comp_idx, their value determine whether to draw input and/or output
            wires on the proper diagram.
        input_positions: list | None
            List of positions received from the sub-diagram that precedes the
            current diagram in a composition diagram. It will help to draw
            the input wires from that sub-diagram to the current diagram.
            In case list is empty, it means the sub-diagram is not part of a
            composition diagram.
        radius: float | None
            This is the variable used to find any other meaningful parameter:
            arrow_length, horizontal span and the horizontal spacing. So it is
            the unit variable of the diagram.

        Returns:
        -------
        list[tuple[float]]
            Lists of output indices that will serve as the input indices of the
            next diagram if there is any.
        list[tuple[float]] | None
            Lists of proper input indices, i.e. different from input indices
            received from a sub-diagram.
        float | list[float]
            radius of either the proper diagram, of the first diagram of a
            composition diagram or the list of radiuses of a tensor diagram.
        """
        if radius is None:
            radius = self.config.node_radius
        arrow_length = 2 * radius
        # Draw X shape
        ax.plot([x - radius, x + radius], [y - radius, y + radius], "k-", linewidth=self.config.wire_width)
        ax.plot([x - radius, x + radius], [y + radius, y - radius], "k-", linewidth=self.config.wire_width)
        # We will draw input and output wires depending of the block is part of
        # a composition diagram
        draw_in_wires = True
        draw_out_wires = True
        if (  # noqa: PLR0916
            comp_idx is not None and comp_idx != -1
        ) or ((comp_idx == -1 or comp_idx is None) and (sub_comp_idx is not None and sub_comp_idx != -1)):
            draw_out_wires = False
        if kept_inputs is None:
            kept_inputs = [0, 1]
        if kept_outputs is None:
            kept_outputs = [0, 1]
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
            if 0 in kept_inputs:
                input1 = patches.FancyArrowPatch(
                    input_positions[0],
                    (x + radius, y - radius),
                    arrowstyle="->",
                    ec="black",
                    mutation_scale=20,
                    linewidth=self.config.wire_width,
                )
                ax.add_patch(input1)
            if 1 in kept_inputs:
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
            if 0 in kept_outputs:
                output1 = patches.FancyArrowPatch(
                    output_positions[0],
                    (x - radius - arrow_length, y - radius),
                    arrowstyle="->",
                    ec="black",
                    mutation_scale=20,
                    linewidth=self.config.wire_width,
                )
                ax.add_patch(output1)
            if 1 in kept_outputs:
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

    def _draw_fourier(  # noqa: C901, PLR0913, PLR0917
        self,
        ax: plt.Axes,
        x: float,
        y: float,
        diagram: Diagram,
        comp_idx: int | None = None,
        sub_comp_idx: int | None = None,
        input_positions: list | None = None,
        radius: float | None = None,
        kept_inputs: list | None = None,
        kept_outputs: list | None = None,
    ) -> tuple[list[float], list[float] | None, float | list[float]]:
        """Draw a Fourier node.

        Parameters:
        ----------
        ax : plt.Axes
            Matplotlib axes
        diagram : Diagram
            The sub-diagram to draw
        x: float
            First coordinate of the diagram
        y: float
            Second coordinate of the diagram
        comp_idx: int | None = None
            Index of a sub-diagram inside a composition. In a composition,
            we don't need to draw the input/output wires of all sub-diagrams.
        sub_comp_idx: int | None = None
            Index of a proper diagram inside a sub-composition diagram. Along with
            the comp_idx, their value determine whether to draw input and/or output
            wires on the proper diagram.
        input_positions: list | None
            List of positions received from the sub-diagram that precedes the
            current diagram in a composition diagram. It will help to draw
            the input wires from that sub-diagram to the current diagram.
            In case list is empty, it means the sub-diagram is not part of a
            composition diagram.
        radius: float | None
            This is the variable used to find any other meaningful parameter:
            arrow_length, horizontal span and the horizontal spacing. So it is
            the unit variable of the diagram.

        Returns:
        -------
        list[tuple[float]]
            Lists of output indices that will serve as the input indices of the
            next diagram if there is any.
        list[tuple[float]] | None
            Lists of proper input indices, i.e. different from input indices
            received from a sub-diagram.
        float | list[float]
            radius of either the proper diagram, of the first diagram of a
            composition diagram or the list of radiuses of a tensor diagram.
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
        if (  # noqa: PLR0916
            comp_idx is not None and comp_idx != -1
        ) or ((comp_idx == -1 or comp_idx is None) and (sub_comp_idx is not None and sub_comp_idx != -1)):
            draw_out_wires = False
        if kept_inputs is None:
            kept_inputs = [0]
        if kept_outputs is None:
            kept_outputs = [0]
        # Draw inputs wires
        init_input_positions = [(x + radius + arrow_length, y)]
        if draw_in_wires:
            if input_positions is None:
                input_positions = init_input_positions
            if 0 in kept_inputs:
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
        if draw_out_wires and 0 in kept_outputs:
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

    def _draw_scalar(self, ax: plt.Axes) -> None:
        """Draw a scalar diagram (closed loop)."""
        ax.text(
            0.5,
            0.5,
            "Scalar (closed diagram)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=self.config.fontsize,
            style="italic",
        )

    def _format_phase(self, phase: ZxPoly) -> str:
        """Format phase polynomial for display.

        Parameters:
        ----------
        phase: ZxPoly
            Real polynomial describing a p/q spider.

        Returns:
        -------
        str
            String display of the phase polynomial
        """
        if phase.is_zero():
            return ""
        terms = []
        max_size = 20
        for d, c in sorted(phase.coeffs.items()):
            if d == 0:
                terms.append(f"{c:.2f}")
            elif d == 1:
                terms.append(f"{c:.2f}x")
            else:
                terms.append(f"{c:.2f}x^{d}")
        result = " + ".join(terms)
        if len(result) > max_size:
            result = result[:17] + "..."
        return result

    def _get_spider_type(self, diagram: ProperDiagram) -> str:
        """Get spider type string from diagram instance.

        Parameters:
        ----------
            diagram: ProperDiagram
                Input proper diagram.

        Returns:
        -------
            str
                Type of the input proper diagram.
        """
        if isinstance(diagram, QSpider):
            return "q"
        if isinstance(diagram, PSpider):
            return "p"
        if isinstance(diagram, Swap):
            return "swap"
        if isinstance(diagram, (Fourier, FourierInv, Fourier2)):
            return "fourier"
        return "default"


def visualize(diagram: Diagram, title: str = "", config: VisualizerConfig | None = None) -> plt.Figure:
    """Convenience function to visualize a diagram.

    Parameters:
    ----------
    diagram : Diagram
        The diagram to visualize.
    title : str
        Title for the figure.
    config : VisualizerConfig | None
        Visualization configuration.

    Returns:
    -------
    plt.Figure
        Matplotlib figure.
    """
    visualizer = DiagramVisualizer(config)
    return visualizer.visualize(diagram, title)


def normalize_radiuses(radiuses: list[float]) -> list[float]:
    """Normalize list of radiuses.

    The list of radiuses corresponds to sub-diagrams of a tensor diagram
    which is part of a contracted diagram. The normalization allows to
    avoid line superpositions when connecting outputs/inputs of D1 to
    inputs/outputs of D2 of the contracted diagram.

    Parameters:
    ----------
    radiuses: list[float]
        List of input radiuses.

    Returns:
    -------
    list[float]
        List of radiuses normalized.
    """
    min_val = min(radiuses)
    max_val = max(radiuses)
    val = r.randint(2, 10)
    if min_val == max_val:
        max_val *= (val + 1) / (val + 2)
        min_val = ((val - 1) / val) * max_val
    else:
        max_val *= (val + 1) / (val + 2)
    output = list(np.linspace(min_val, max_val, len(radiuses) + 1))
    output.pop(0)
    return output


if __name__ == "__main__":
    # Build proper diagrams
    p = ZxPoly({1: 2, 2: 4})
    q = ZxPoly({1: 2, 3: 4, 5: 7})

    a = Fourier()
    b = Fourier2()
    c = FourierInv()
    d_1 = QSpider(1, 1, p)
    d_11 = QSpider(3, 2, p)
    d_12 = QSpider(4, 3, p)
    d_13 = QSpider(5, 5, p)
    d_2 = PSpider(1, 1, q)
    d_21 = PSpider(3, 2, p)
    d_22 = PSpider(3, 4, p)
    d_23 = PSpider(5, 5, p)
    e_1 = Swap()

    # Valid test cases (respecting input/output counts)

    # # 1. Single proper diagram
    # fig1 = visualize(a, "Fourier")
    # fig1_1 = visualize(b, "Fourier2")
    # fig1_2 = visualize(c, "Inverse Fourier")
    # fig1_3 = visualize(d_1, "Single QSpider 1")
    # fig1_31 = visualize(d_11, "Single QSpider 2")
    # fig1_32 = visualize(d_12, "Single QSpider 3")
    # fig1_33 = visualize(d_13, "Single QSpider 4")
    # fig1_4 = visualize(d_2, "Single PSpider 1")
    # fig1_41 = visualize(d_21, "Single PSpider 2")
    # fig1_42 = visualize(d_22, "Single PSpider 3")
    # fig1_43 = visualize(d_23, "Single PSpider 4")
    # fig1_5 = visualize(e_1, "Swap")
    # fig1.savefig("Fourier")
    # fig1_1.savefig("Fourier2")
    # fig1_2.savefig("Inverse Fourier")
    # fig1_3.savefig("Single QSpider 1")
    # fig1_31.savefig("Single QSpider 2")
    # fig1_32.savefig("Single QSpider 3")
    # fig1_33.savefig("Single QSpider 4")
    # fig1_4.savefig("Single PSpider 1")
    # fig1_41.savefig("Single PSpider 2")
    # fig1_42.savefig("Single PSpider 3")
    # fig1_43.savefig("Single PSpider 4")
    # fig1_5.savefig("Swap")

    # # 2. Simple composition: QSpider (1 output) followed by Fourier (1 input)
    comp1 = d_1.compose(a)  # Valid: 1→1
    # fig2 = visualize(comp1, "Composition: Fourier then QSpider")
    # fig2.savefig("Composition: Fourier then QSpider")

    # # 2.1 Complex composition
    # comp2 = b.compose(comp1)
    # comp2 = c.compose(comp2)
    # comp2 = d_2.compose(comp2)
    # fig2_1 = visualize(comp2, "Complex Composition")
    # fig2_1.savefig("Complex Composition")

    # # 3. Tensor of two proper diagrams
    # tensor1 = a.tensor(b)  # Fourier ⊗ Fourier2
    # fig3 = visualize(tensor1, "Tensor: Fourier ⊗ Fourier2")
    # fig3.savefig("Tensor: Fourier ⊗ Fourier2")

    # # 3.1 Complex tensor product
    # tensor2 = e_1.tensor(tensor1)
    # tensor2 = d_22.tensor(tensor2)
    # tensor2 = d_13.tensor(tensor2)
    # fig3_1 = visualize(tensor2, "Complex Tensor Diagram")
    # fig3_1.savefig("Complex Tensor Diagram")

    # # 4. Composition of tensor with swap: need 2 outputs → 2 inputs
    # two_fouriers = a.tensor(a)  # Fourier ⊗ Fourier (2 outputs)
    # comp_swap = two_fouriers.compose(e_1)  # Valid: 2→2
    # fig4 = visualize(comp_swap, "Composition: Swap then (F ⊗ F)")
    # fig4.savefig("Composition: Swap then (F ⊗ F)")

    # 5. Nested composition: (c ∘ a) ∘ b
    # nested_comp = d_1.compose(a).compose(b)
    # fig5 = visualize(nested_comp, "Nested composition: (c ∘ a) ∘ b")
    # fig5.savefig("Nested composition: (c ∘ a) ∘ b")

    # 6. Tensor containing composition
    tensor_with_comp = a.tensor(comp1)  # F ⊗ (c ∘ a) - each has 1 output
    fig6 = visualize(tensor_with_comp, "Tensor containing composition")
    fig6.savefig("Tensor containing composition")

    # 7. Complex: (F ⊗ F) composed with Swap, then composed with (F ⊗ F)
    left = a.tensor(a)  # 2 outputs
    middle = left.compose(e_1)  # 2 outputs after swap
    right = a.tensor(a)  # 2 inputs
    full = middle.compose(right)  # 2→2
    fig7 = visualize(full, "Full circuit: (F⊗F) ∘ Swap ∘ (F⊗F)")
    fig7.savefig("Full circuit: (F⊗F) ∘ Swap ∘ (F⊗F)")

    # # 8. Big Complex diagram
    e = a.tensor(b)
    e = e.tensor(b)
    d2 = e_1.tensor(a)
    f = e.compose(d2)
    g = d2.compose(f)
    g = d2.compose(g)
    a2 = QSpider(3, 3, p)
    b2 = CompositionDiagram([
        a2,
        a2,
    ])
    b2 = b2.compose(a2)
    b2 = b2.compose(a2)
    g = g.compose(b2)
    h = a.compose(b)
    h = h.compose(b)
    j = QSpider(4, 4, q)
    k = j.tensor(g)
    i = a.tensor(g)
    # fig8 = visualize(i, "Complex Diagram")
    # fig8.savefig("Complex Diagram.png")

    # 9. Proper Contracted diagram
    c1 = ContractedDiagram(d_13, d_23, [0, 1, 4], [1, 2, 3], [0, 1, 2], [1, 2, 4])
    c1_1 = ContractedDiagram(e_1, c, [0], [0], [], [])
    c1_2 = ContractedDiagram(d_12, e_1, [1], [1], [0, 2], [0, 1])
    c1_3 = ContractedDiagram(b, d_21, [0], [1], [0], [0])
    # fig9 = visualize(c1, "Proper Contracted Diagram")
    # fig9_1 = visualize(c1_1, "Proper Contracted Diagram: Swap and Inv Fourier")
    # fig9_2 = visualize(c1_2, "Proper Contracted Diagram: QSpider and Swap")
    # fig9_3 = visualize(c1_3, "Proper Contracted Diagram: Fourier2 and PSpider")
    # fig9.savefig("Proper Contracted Diagram")
    # fig9_1.savefig("Proper Contracted Diagram: Swap and Inv Fourier")
    # fig9_2.savefig("Proper Contracted Diagram: QSpider and Swap")
    # fig9_3.savefig("Proper Contracted Diagram: Fourier2 and PSpider")

    # 10. Proper Contracted diagram inside a Composition Diagram
    e_2 = e_1.tensor(a)
    e_2 = e_2.tensor(b)
    d1 = c1.compose(e_2)
    d1 = e_2.compose(d1)
    e_3 = b.tensor(c)
    # fig10 = visualize(d1, "Proper Contracted Diagram inside a Composition Diagram 1")
    # fig10_1 = visualize(c1_1.compose(e_3), "Proper Contracted Diagram inside a Composition Diagram 2")
    # fig10_2 = visualize(e_3.compose(c1_2), "Proper Contracted Diagram inside a Composition Diagram 3")
    # fig10_3 = visualize(a.compose(c1_3.compose(e_3)), "Proper Contracted Diagram inside a Composition Diagram 3")
    # fig10.savefig("Proper Contracted Diagram inside a Composition Diagram 1")
    # fig10_1.savefig("Proper Contracted Diagram inside a Composition Diagram 2")
    # fig10_2.savefig("Proper Contracted Diagram inside a Composition Diagram 3")
    # fig10_3.savefig("Proper Contracted Diagram inside a Composition Diagram 4")

    # 11. Proper Contracted diagram inside a Tensor Diagram
    d_14 = QSpider(5, 5, 20 * (p + q))
    d1 = c1.tensor(d_14)
    d1 = d_23.tensor(d1)
    d1 = c1_1.tensor(d1)
    # fig11 = visualize(d1, "Proper Contracted Diagram inside a Tensor Diagram")
    # fig11.savefig("Proper Contracted Diagram inside a Tensor Diagram")

    # 12. Contracted diagram composed of Composition Diagram(s)
    c2 = d_13.compose(d_23)
    c2 = d_14.compose(c2)
    d2 = ContractedDiagram(c2, d_13, [0, 1, 4], [1, 2, 3], [0, 1, 2], [1, 2, 4])
    d3 = ContractedDiagram(d_13, c2, [0, 1, 4], [1, 2, 3], [0, 1, 2], [1, 2, 4])
    # fig12_1 = visualize(d2, "Contracted diagram composed of Composition Diagrams 1")
    # fig12_1.savefig("Contracted diagram composed of Composition Diagrams 1")
    # fig12_2 = visualize(d3, "Contracted diagram composed of Composition Diagrams 2")
    # fig12_2.savefig("Contracted diagram composed of Composition Diagrams 2")

    # 13. Contracted diagram composed of Tensor Diagram(s)
    c2 = d_13.tensor(e_1)
    c3 = d_22.tensor(a)
    c3 = c3.tensor(b)
    c3 = c3.tensor(c)
    c4 = d_12.tensor(a)
    d2 = ContractedDiagram(c2, c3, [0, 1, 4, 6], [1, 2, 3, 4], [0, 1, 2, 5], [0, 1, 2, 5])
    d3 = ContractedDiagram(d_13, c2, [0, 1, 4], [1, 2, 3], [0, 1, 2], [1, 2, 4])
    d4 = ContractedDiagram(c3, c4, [0, 1, 2, 6], [0, 1, 2, 4], [4, 5], [1, 3])
    # fig13_1 = visualize(d2, "Contracted diagram composed of Tensor Diagrams 1")
    # fig13_1.savefig("Contracted diagram composed of Tensor Diagrams 1")
    # fig13_2 = visualize(d3, "Contracted diagram composed of Tensor Diagrams 2")
    # fig13_2.savefig("Contracted diagram composed of Tensor Diagrams 2")
    # fig13_3 = visualize(d4, "Contracted diagram composed of Tensor Diagrams 3")
    # fig13_3.savefig("Contracted diagram composed of Tensor Diagrams 3")
