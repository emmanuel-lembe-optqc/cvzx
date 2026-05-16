"""Visualization for CV ZX diagrams.

This module provides visualization capabilities for:
    - ProperDiagram (spiders, gates)
    - CompositionDiagram (sequential composition)
    - TensorDiagram (parallel composition)
    - ContractedDiagram (partial trace with connections)

The visualizer uses matplotlib to draw diagrams in a way that respects
the input/output wire ordering and connection indices.
"""

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
        Radius of spider circles (in points or display units).

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
    vertical_factor: float = 2.2
    wire_width: float = 1.5
    fontsize: float = 10.0

    colors: dict = field(
        default_factory=lambda: {
            "q": "lightgreen",
            "p": "lightcoral",
            "macronode": "lightblue",
            "non_gaussian": "gold",
            "swap": "gray",
            "fourier": "lavender",
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
        elif isinstance(diagram, ScalarDiagram):
            self._draw_scalar(ax)
        else:
            ax.text(
                0.5, 0.5, f"Unknown diagram type: {type(diagram)}", ha="center", va="center", transform=ax.transAxes
            )

        plt.tight_layout()
        return fig

    def _draw_proper_diagram(  # noqa: C901, PLR0913, PLR0914, PLR0917
        self,
        ax: plt.Axes,
        diagram: ProperDiagram,
        x: float = 0,
        y: float = 0,
        comp_idx: int | None = None,
        input_positions: list | None = None,
        radius: float | None = None,
    ) -> list[tuple]:
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
            Index of a sub-diagram inside a composition. Depending on its
            value we draw input or output wires. In a composition, we don't
            need to draw the input/output wires of all sub-diagrams. This
            variable allows to monitor it.
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
            if comp_idx is not None and comp_idx != -1:
                draw_out_wires = False
            # Draw input wires (left side)
            if draw_in_wires:
                y_offset = list(np.linspace(0, height, diagram.num_inputs + 2))
                # Remove the two edges
                y_offset.pop(0)
                y_offset.pop(-1)
                if input_positions is None:
                    # input_positions is empty only for the first element of a composition
                    # or a single proper diagram
                    input_positions = [
                        (pivot[0] + width + arrow_length, pivot[1] + y_offset[i]) for i in range(diagram.num_inputs)
                    ]
                for i in range(diagram.num_inputs):
                    input_i = patches.FancyArrowPatch(
                        input_positions[i],
                        (pivot[0] + width, pivot[1] + y_offset[i]),
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
            output_positions = self._draw_swap(ax, x, y, comp_idx, input_positions, radius)
        elif isinstance(diagram, (Fourier, FourierInv, Fourier2)):
            output_positions = self._draw_fourier(ax, x, y, diagram, comp_idx, input_positions, radius)
        return output_positions

    def _draw_composition(  # noqa: PLR0913, PLR0917
        self,
        ax: plt.Axes,
        diagram: CompositionDiagram,
        x: float = 0,
        y: float = 0,
        comp_idx: int | None = None,
        input_positions: list | None = None,
        comp_size: int | None = None,
        hor_spacing: float | None = None,
        radius: float | None = None,
    ) -> list[tuple] | None:
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
            Index of a sub-diagram inside a composition. Depending on its
            value we draw input or output wires. In a composition, we don't
            need to draw the input/output wires of all sub-diagrams. This
            variable allows to monitor it.
        input_positions: list | None
            List of positions received from the sub-diagram that precedes the
            current diagram in a composition diagram. It will help to draw
            the input wires from that sub-diagram to the current diagram.
            In case list is empty, it means the sub-diagram is not part of a
            composition diagram.
        comp_size: int | None
            This variable is not None when a composition is part of a tensor
            diagram. And in that case we need to resize this composition
            diagram so that it fits in the horizontal spacing of the tensor
            diagram. It is inside this function that it is changed to an int.
        hor_spacing: float | None
            Horizontal spacing between sub-diagrams inside a composition diagram.
        radius: float | None
            This is the variable used to find any other meaningful parameter:
            arrow_length, horizontal span and the horizontal spacing. So it is
            the unit variable of the diagram.

        Returns:
        -------
        list[tuple[float]]
            Lists of output indices that will serve as the input indices of the
            next diagram if there is any.
        """
        n = len(diagram.diagrams)
        if n == 0:
            return None

        # Draw each sub-diagram at its position
        # comp_size is not None and radius is None means that we are inside
        # a composition block which is part of a tensor product therefore we must
        # resize the sub_diag_hor_span, spacing and radius
        # If comp_size is not None and radius is not None therefore we are
        # inside a sub_composition diagram of a composition diagram of a tensor diagram
        # Therefore we must not resize the sub_diag_hor_span, spacing and radius because there
        # are given to the sub_composition diagram by the composition diagram
        if comp_size is not None and radius is None:
            sub_diag_hor_span = 18 * self.config.node_radius / (2 * comp_size + 1)
            hor_spacing = 2 * sub_diag_hor_span / 3
            radius = sub_diag_hor_span / 6
            x += 3 * self.config.node_radius - sub_diag_hor_span / 2
        # If comp_size is None, therefore we are inside a composition diagram which
        # is not part of a tensor diagram. So no need to resize.
        elif comp_size is None:
            sub_diag_hor_span = 6 * self.config.node_radius
            hor_spacing = 2 * sub_diag_hor_span / 3
            radius = self.config.node_radius
        # Calculate positions for each sub-diagram
        # Composition is drawn left to right
        start_x = 0
        sub_diagram_size = 1
        for i, sub_diagram in enumerate(diagram.diagrams):
            d = start_x - i * hor_spacing * sub_diagram_size
            idx = i
            if i == n - 1:
                idx = comp_idx if comp_idx is not None else -1
            if isinstance(sub_diagram, CompositionDiagram):
                # If comp_size is not None we must resize
                if comp_size is not None:
                    comp_size = len(sub_diagram.diagrams)
                    sub_sub_diag_hor_span = 18 * radius / (2 * comp_size + 1)
                    sub_hor_spacing = 2 * sub_sub_diag_hor_span / 3
                    sub_radius = sub_sub_diag_hor_span / 6
                    x2 = x + d
                    x2 += 3 * radius - sub_sub_diag_hor_span / 2
                    input_positions = self._draw_sub_diagram(
                        ax,
                        sub_diagram,
                        x2,
                        y,
                        comp_idx=idx,
                        input_positions=input_positions,
                        comp_size=comp_size,
                        hor_spacing=sub_hor_spacing,
                        radius=sub_radius,
                    )
                # if comp_size is not None, no need to resize
                else:
                    # self.config.get_horizontal_space(CompositionDiagram)
                    sub_diagram_size = self.config.get_horizontal_space(sub_diagram)
                    input_positions = self._draw_sub_diagram(
                        ax,
                        sub_diagram,
                        x + d,
                        y,
                        comp_idx=idx,
                        input_positions=input_positions,
                        comp_size=comp_size,
                        hor_spacing=hor_spacing,
                        radius=radius,
                    )
            else:
                input_positions = self._draw_sub_diagram(
                    ax,
                    sub_diagram,
                    x + d,
                    y,
                    comp_idx=idx,
                    input_positions=input_positions,
                    comp_size=comp_size,
                    hor_spacing=hor_spacing,
                    radius=radius,
                )
        return input_positions

    def _draw_tensor(  # noqa: PLR0913, PLR0917
        self,
        ax: plt.Axes,
        diagram: TensorDiagram,
        x: float = 0,
        y: float = 0,
        comp_idx: int | None = None,
        input_positions: list | None = None,
        comp_size: int | None = None,
        hor_spacing: float | None = None,
        radius: float | None = None,
    ) -> None:
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
            Index of a sub-diagram inside a composition. Depending on its
            value we draw input or output wires. In a composition, we don't
            need to draw the input/output wires of all sub-diagrams. This
            variable allows to monitor it.
        input_positions: list | None
            List of positions received from the sub-diagram that precedes the
            current diagram in a composition diagram. It will help to draw
            the input wires from that sub-diagram to the current diagram.
            In case list is empty, it means the sub-diagram is not part of a
            composition diagram.
        comp_size: int | None
            This variable is not None when a composition is part of a tensor
            diagram. And in that case we need to resize this composition
            diagram so that it fits in the horizontal spacing of the tensor
            diagram. It is inside this function that it is changed to an int.
            It is an important argument because of a tensor diagram is a sub-diagram of a
            composition diagram which
        hor_spacing: float | None
            Horizontal spacing between sub-diagrams inside a composition diagram.
            It is not None when the tensor diagram is not part of a composition
            diagram.
        radius: float | None
            This is the variable used to find any other meaningful parameter:
            arrow_length, horizontal span and the horizontal spacing. So it is
            the unit variable of the diagram.

        Returns:
        -------
        list[tuple[float]]
            Lists of output indices that will serve as the input indices of the
            next diagram if there is any.
        """
        n = len(diagram.diagrams)
        if n == 0:
            return []

        # Calculate positions for each sub-diagram
        # Tensor is drawn top to bottom
        start_y = 0
        output_positions = []
        j = 0
        if radius is None:
            radius = self.config.node_radius
            vertical_spacing = self.config.vertical_spacing
        else:
            vertical_spacing = 2.2 * radius
        for i, sub_diagram in enumerate(diagram.diagrams):
            if isinstance(sub_diagram, CompositionDiagram) and comp_size is None:
                comp_size = len(sub_diagram.diagrams)
            h = start_y - i * vertical_spacing
            # Draw sub-diagram with local coordinates
            sub_input_positions = (
                input_positions[j : j + sub_diagram.num_inputs] if input_positions is not None else None
            )

            output_positions += self._draw_sub_diagram(
                ax,
                sub_diagram,
                x,
                y + h,
                comp_idx=comp_idx,
                input_positions=sub_input_positions,
                comp_size=comp_size,
                hor_spacing=hor_spacing,
                radius=radius,
            )
            j = sub_diagram.num_inputs
        return output_positions

    def _draw_sub_diagram(  # noqa: PLR0913, PLR0917
        self,
        ax: plt.Axes,
        diagram: Diagram,
        x: float,
        y: float,
        comp_idx: int | None = None,
        input_positions: list | None = None,
        comp_size: int | None = None,
        hor_spacing: float | None = None,
        radius: float | None = None,
    ) -> tuple[list[float], list[float]]:
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
            Index of a sub-diagram inside a composition. Depending on its
            value we draw input or output wires. In a composition, we don't
            need to draw the input/output wires of all sub-diagrams. This
            variable allows to monitor it.
        input_positions: list | None
            List of positions received from the sub-diagram that precedes the
            current diagram in a composition diagram. It will help to draw
            the input wires from that sub-diagram to the current diagram.
            In case list is empty, it means the sub-diagram is not part of a
            composition diagram.
        comp_size: int | None
            This variable is not None when a composition is part of a tensor
            diagram. And in that case we need to resize this composition
            diagram so that it fits in the horizontal spacing of the tensor
            diagram.
        hor_spacing: float | None
            Horizontal spacing between sub-diagrams inside a composition diagram.
        radius: float | None
            This is the variable used to find any other meaningful parameter:
            arrow_length, horizontal span and the horizontal spacing. So it is
            the unit variable of the diagram.

        Returns:
        -------
        list[tuple[float]]
            Lists of output indices that will serve as the input indices of the
            next sub-diagram if there is any.
        """
        if comp_size is None or radius is None or hor_spacing is None:
            sub_diag_hor_span = self.config.horizontal_spacing
            hor_spacing = 2 * sub_diag_hor_span / 3
            radius = self.config.node_radius
        if isinstance(diagram, ProperDiagram):
            return self._draw_proper_diagram(ax, diagram, x, y, comp_idx, input_positions, radius)
        if isinstance(diagram, CompositionDiagram):
            return self._draw_composition(ax, diagram, x, y, comp_idx, input_positions, comp_size, hor_spacing, radius)
        if isinstance(diagram, TensorDiagram):
            return self._draw_tensor(ax, diagram, x, y, comp_idx, input_positions, comp_size, hor_spacing, radius)
        return [], []

    def _draw_swap(  # noqa: PLR0913, PLR0917
        self,
        ax: plt.Axes,
        x: float,
        y: float,
        comp_idx: int | None = None,
        input_positions: list | None = None,
        radius: float | None = None,
    ) -> list[tuple]:
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
            Index of a sub-diagram inside a composition. Depending on its
            value we draw input or output wires. In a composition, we don't
            need to draw the input/output wires of all sub-diagrams. This
            variable allows to monitor it.
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
        if comp_idx is not None and comp_idx != -1:
            draw_out_wires = False
        # Draw inputs arrows
        if draw_in_wires:
            if input_positions is None:
                input_positions = [
                    (x + radius + arrow_length, y + radius),
                    (x + radius + arrow_length, y - radius),
                ]
            input1 = patches.FancyArrowPatch(
                input_positions[0],
                (x + radius, y + radius),
                arrowstyle="->",
                ec="black",
                mutation_scale=20,
                linewidth=self.config.wire_width,
            )
            input2 = patches.FancyArrowPatch(
                input_positions[1],
                (x + radius, y - radius),
                arrowstyle="->",
                ec="black",
                mutation_scale=20,
                linewidth=self.config.wire_width,
            )
            ax.add_patch(input1)
            ax.add_patch(input2)

        # Draw Outputs
        output_positions = [
            (x - radius, y + radius),
            (x - radius, y - radius),
        ]
        if draw_out_wires:
            output1 = patches.FancyArrowPatch(
                output_positions[0],
                (x - radius - arrow_length, y + radius),
                arrowstyle="->",
                ec="black",
                mutation_scale=20,
                linewidth=self.config.wire_width,
            )
            output2 = patches.FancyArrowPatch(
                output_positions[1],
                (x - radius - arrow_length, y - radius),
                arrowstyle="->",
                ec="black",
                mutation_scale=20,
                linewidth=self.config.wire_width,
            )
            ax.add_patch(output1)
            ax.add_patch(output2)
        return output_positions

    def _draw_fourier(  # noqa: PLR0913, PLR0917
        self,
        ax: plt.Axes,
        x: float,
        y: float,
        diagram: Diagram,
        comp_idx: int | None = None,
        input_positions: list | None = None,
        radius: float | None = None,
    ) -> list[tuple]:
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
            Index of a sub-diagram inside a composition. Depending on its
            value we draw input or output wires. In a composition, we don't
            need to draw the input/output wires of all sub-diagrams. This
            variable allows to monitor it.
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
        """
        if radius is None:
            radius = self.config.node_radius
        arrow_length = 2 * radius
        symbol = "F"
        if isinstance(diagram, FourierInv):
            symbol = "F†"
            ax.plot([x - radius, x - radius], [y - radius, y + radius], "k-", linewidth=self.config.wire_width)
            ax.plot([x - radius, x + radius], [y + radius, y], "k-", linewidth=self.config.wire_width)
            ax.plot([x - radius, x + radius], [y - radius, y], "k-", linewidth=self.config.wire_width)
            ax.text(x, y, symbol, ha="center", va="center", fontsize=self.config.fontsize * radius)
        elif isinstance(diagram, Fourier2):
            symbol = "F²"
            ax.plot([x - radius, x], [y, y + radius], "k-", linewidth=self.config.wire_width)
            ax.plot([x - radius, x], [y, y - radius], "k-", linewidth=self.config.wire_width)
            ax.plot([x + radius, x], [y, y - radius], "k-", linewidth=self.config.wire_width)
            ax.plot([x + radius, x], [y, y + radius], "k-", linewidth=self.config.wire_width)
            ax.text(x, y, symbol, ha="center", va="center", fontsize=self.config.fontsize * radius)
        else:
            ax.plot([x + radius, x + radius], [y - radius, y + radius], "k-", linewidth=self.config.wire_width)
            ax.plot([x + radius, x - radius], [y + radius, y], "k-", linewidth=self.config.wire_width)
            ax.plot([x + radius, x - radius], [y - radius, y], "k-", linewidth=self.config.wire_width)
            ax.text(x, y, symbol, ha="center", va="center", fontsize=self.config.fontsize * radius)

        # We will draw input and output wires depending of the block is part of
        # a composition diagram
        draw_in_wires = True
        draw_out_wires = True
        if comp_idx is not None and comp_idx != -1:
            draw_out_wires = False
        # Draw inputs wires
        if draw_in_wires:
            if input_positions is None:
                input_positions = [(x + radius + arrow_length, y)]
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
        if draw_out_wires:
            output1 = patches.FancyArrowPatch(
                output_positions[0],
                (x - radius - arrow_length, y),
                arrowstyle="->",
                ec="black",
                mutation_scale=20,
                linewidth=self.config.wire_width,
            )
            ax.add_patch(output1)
        return output_positions

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


if __name__ == "__main__":
    # Build proper diagrams
    p = ZxPoly({1: 2, 2: 4})

    a = Fourier()
    b = Fourier2()
    c = QSpider(1, 1, p)
    d = Swap()

    # Valid test cases (respecting input/output counts)

    # 1. Single proper diagram
    fig1 = visualize(c, "Single QSpider")
    fig1.savefig("Single QSpider")

    # 2. Simple composition: QSpider (1 output) followed by Fourier (1 input)
    comp1 = c.compose(a)  # Valid: 1→1
    fig2 = visualize(comp1, "Composition: QSpider then Fourier")
    fig2.savefig("Composition: QSpider then Fourier")

    # 3. Tensor of two proper diagrams
    tensor1 = a.tensor(b)  # Fourier ⊗ Fourier2
    fig3 = visualize(tensor1, "Tensor: Fourier ⊗ Fourier2")
    fig3.savefig("Tensor: Fourier ⊗ Fourier2")

    # 4. Composition of tensor with swap: need 2 outputs → 2 inputs
    # Fourier has 1 output, so tensor of two Fouriers has 2 outputs
    two_fouriers = a.tensor(a)  # Fourier ⊗ Fourier (2 outputs)
    comp_swap = two_fouriers.compose(d)  # Valid: 2→2
    fig4 = visualize(comp_swap, "Composition: (F ⊗ F) then Swap")
    fig4.savefig("Composition: (F ⊗ F) then Swap")

    # 5. Nested composition: (c ∘ a) ∘ b
    nested_comp = c.compose(a).compose(b)
    fig5 = visualize(nested_comp, "Nested composition: (c ∘ a) ∘ b")
    fig5.savefig("Nested composition: (c ∘ a) ∘ b")

    # 6. Tensor containing composition
    tensor_with_comp = a.tensor(comp1)  # F ⊗ (c ∘ a) - each has 1 output
    fig6 = visualize(tensor_with_comp, "Tensor containing composition")
    fig6.savefig("Tensor containing composition")

    # 7. Complex: (F ⊗ F) composed with Swap, then composed with (F ⊗ F)
    left = a.tensor(a)  # 2 outputs
    middle = left.compose(d)  # 2 outputs after swap
    right = a.tensor(a)  # 2 inputs
    full = middle.compose(right)  # 2→2
    fig7 = visualize(full, "Full circuit: (F⊗F) ∘ Swap ∘ (F⊗F)")
    fig7.savefig("Full circuit: (F⊗F) ∘ Swap ∘ (F⊗F)")
