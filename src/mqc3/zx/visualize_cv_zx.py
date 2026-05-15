"""Visualization for CV ZX diagrams.

This module provides visualization capabilities for:
    - ProperDiagram (spiders, gates)
    - CompositionDiagram (sequential composition)
    - TensorDiagram (parallel composition)
    - ContractedDiagram (partial trace with connections)

The visualizer uses matplotlib to draw diagrams in a way that respects
the input/output wire ordering and connection indices.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

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
    # MacronodeSpider,
    # NonGaussianSpider,
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

    Attributes:
    ----------
    node_radius : float
        Radius of spider circles
    arrow_length: float
        Length of input/output wires
    wire_width : float
        Width of wire lines
    fontsize : float
        Font size for text labels
    colors : dict
        Color mapping for spider types
    symbols : dict
        Symbol mapping for spider types
    """

    node_radius: float = 0.4
    wire_width: float = 1.5
    fontsize: float = 10.0
    arrow_length: float = 2 * node_radius
    horizontal_spacing: float = node_radius * 2 + arrow_length * 2
    vertical_spacing: float = node_radius * 2.2

    colors: dict = field(
        default_factory=lambda: {
            "q": "lightgreen",
            "p": "lightcoral",
            "macronode": "lightblue",
            "non_gaussian": "gold",
            "fourier": "lavender",
            "swap": "gray",
            "boundary": "white",
            "default": "white",
        }
    )

    edge_colors: dict = field(
        default_factory=lambda: {
            "q": "darkgreen",
            "p": "darkred",
            "macronode": "blue",
            "non_gaussian": "orange",
            "default": "black",
        }
    )

    symbols: dict = field(
        default_factory=lambda: {
            "q": "●",
            "p": "○",
            "macronode": "■",
            "non_gaussian": "★",
            "fourier": "F",
            "swap": "×",
            "boundary": "◉",
            "default": "?",
        }
    )


class DiagramVisualizer:
    """Visualizer for CV ZX diagrams.

    This class handles layout and drawing of various diagram types,
    ensuring proper wire connections and index ordering.
    """

    def __init__(self, config: Optional[VisualizerConfig] = None):
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
            ax.set_title(title, fontsize=14)

        if isinstance(diagram, ProperDiagram):
            self._draw_proper_diagram(ax, diagram)
        elif isinstance(diagram, CompositionDiagram):
            self._draw_composition(ax, diagram)
        elif isinstance(diagram, TensorDiagram):
            self._draw_tensor(ax, diagram)
        elif isinstance(diagram, ContractedDiagram):
            self._draw_contracted(ax, diagram)
        elif isinstance(diagram, ScalarDiagram):
            self._draw_scalar(ax, diagram)
        else:
            ax.text(
                0.5, 0.5, f"Unknown diagram type: {type(diagram)}", ha="center", va="center", transform=ax.transAxes
            )

        plt.tight_layout()
        return fig

    def _draw_proper_diagram(
        self,
        ax: plt.Axes,
        diagram: ProperDiagram,
        x: float = 0,
        y: float = 0,
        comp_idx: int | None = None,
        tensor_idx: int | None = None,
        input_positions: list = [],
        radius: float | None = None,
    ) -> list[tuple]:
        """Draw a proper diagram (single node)."""
        output_positions = []
        # Determine spider type
        # if isinstance(diagram, (MacronodeSpider, NonGaussianSpider, QSpider, PSpider)):
        if isinstance(diagram, (QSpider, PSpider)):
            if radius is None:
                radius = self.config.node_radius
            arrow_length = 2 * radius
            print("arrow length", arrow_length)
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
            print("Inputs of the spider", input_positions, draw_out_wires)
            # Draw input wires (left side)
            # print(comp_idx, color, draw_in_wires, draw_out_wires)
            if draw_in_wires:
                y_offset = list(np.linspace(0, height, diagram.num_inputs + 2))
                # Remove the two edges
                y_offset.pop(0)
                y_offset.pop(-1)
                if not input_positions:
                    # input_positions is empty only for the first element of a composition
                    # or a single proper diagram
                    input_positions = [
                        (pivot[0] + width + arrow_length, pivot[1] + y_offset[i]) for i in range(diagram.num_inputs)
                    ]
                assert len(input_positions) == diagram.num_inputs
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
            print("Inputs of the SWAP", input_positions)
            output_positions = self._draw_swap(ax, x, y, comp_idx, tensor_idx, input_positions)
        elif isinstance(diagram, (Fourier, FourierInv, Fourier2)):
            output_positions = self._draw_fourier(ax, x, y, diagram, comp_idx, tensor_idx, input_positions)
        return output_positions

    def _draw_composition(
        self,
        ax: plt.Axes,
        diagram: CompositionDiagram,
        x: float = 0,
        y: float = 0,
        comp_idx: int | None = None,
        input_positions: list = [],
        comp_size: int | None = None,
        sub_diagram_span: float | None = None,
        spacing: float | None = None,
        radius: float | None = None,
    ) -> list[tuple] | None:
        """Draw a composition diagram (sequential)."""
        n = len(diagram.diagrams)
        if n == 0:
            return None

        # Calculate positions for each sub-diagram
        # Composition is drawn left to right
        start_x = 0
        print("Inputs of the composition", input_positions)
        # Draw each sub-diagram at its position
        sub_diagram_size = 1
        # comp_size is not None and sub_diagram_span is None means that we are inside
        # a composition block which is part of a tensor product therefore we must
        # compute sub_diagram_size, spacing and radius
        # If comp_size is not None and sub_diagram_span is not None therefore we are
        # inside a sub_composition diagram of a composition diagram of a tensor diagram
        # Therefore we must not compute sub_diagram_size, spacing and radius because there
        # are given to the sub_composition diagram by the composition diagram
        if comp_size is not None and sub_diagram_span is None:
            sub_diagram_span = 18 * self.config.node_radius / (2 * comp_size + 1)
            spacing = 2 * sub_diagram_span / 3
            radius = sub_diagram_span / 6
            x += 3 * self.config.node_radius - sub_diagram_span / 2
        # If comp_size is None, therefore we are inside a composition diagram which
        # is not part of a tensor diagram. So no need to resize.
        elif comp_size is None:
            sub_diagram_span = 6 * self.config.node_radius
            spacing = 2 * sub_diagram_span / 3
            radius = self.config.node_radius
        sub_diagram_size = 1
        for i, sub_diagram in enumerate(diagram.diagrams):
            d = start_x - i * spacing * sub_diagram_size
            idx = i
            if i == n - 1:
                idx = comp_idx if comp_idx is not None else -1
            if isinstance(sub_diagram, CompositionDiagram):
                sub_diagram_size = len(sub_diagram.diagrams)
                # sub_sub_diagram_span = 18 * radius / (2 * comp_size + 1)
                # sub_spacing = 2 * sub_sub_diagram_span / 3
                # sub_radius = sub_sub_diagram_span / 6
                # x2 = x + d - 3 * sub_radius + sub_sub_diagram_span / 2
                # # sub_d = start_x - i * sub_spacing * (sub_sub_diagram_size)
                # input_positions = self._draw_subdiagram(
                #     ax,
                #     sub_diagram,
                #     x + d,
                #     y,
                #     comp_idx=idx,
                #     input_positions=input_positions,
                #     sub_diagram_span=sub_diagram_span,
                #     spacing=spacing,
                #     radius=radius,
                # )
            # else:
            print("inside composition", sub_diagram_span, radius)
            input_positions = self._draw_subdiagram(
                ax,
                sub_diagram,
                x + d,
                y,
                comp_idx=idx,
                input_positions=input_positions,
                comp_size=comp_size,
                sub_diagram_span=sub_diagram_span,
                spacing=spacing,
                radius=radius,
            )
        print("End composition", input_positions)
        return input_positions

    def _draw_tensor(
        self,
        ax: plt.Axes,
        diagram: TensorDiagram,
        x: float = 0,
        y: float = 0,
        comp_idx: int | None = None,
        input_positions: list = [],
    ) -> None:
        """Draw a tensor diagram (parallel)."""
        n = len(diagram.diagrams)
        if n == 0:
            return

        # Calculate positions for each sub-diagram
        # Tensor is drawn top to bottom
        start_y = 0
        output_positions = []
        j = 0
        print("input positions to the tensor", input_positions)
        comp_size = None
        for i, sub_diagram in enumerate(diagram.diagrams):
            if isinstance(sub_diagram, CompositionDiagram):
                comp_size = len(sub_diagram.diagrams)
            h = start_y - i * self.config.vertical_spacing
            # Draw sub-diagram with local coordinates
            output_positions += self._draw_subdiagram(
                ax,
                sub_diagram,
                x,
                y + h,
                comp_idx=comp_idx,
                tensor_idx=i,
                input_positions=input_positions[j : j + sub_diagram.num_inputs],
                comp_size=comp_size,
            )
            j = sub_diagram.num_inputs
        print("Output positions from the tensor", output_positions)
        return output_positions

    def _draw_contracted(self, ax: plt.Axes, diagram: ContractedDiagram) -> None:
        """Draw a contracted diagram (feedback connections).

        Special care is needed because:
            - First diagram (D1) is drawn
            - Second diagram (D2) is drawn
            - Forward connections (I1→I2) are drawn as wires
            - Feedback connections (J2→J1) are drawn as wires
            - External wires use kept_first_inputs, kept_first_outputs,
              kept_second_inputs, kept_second_outputs
        """
        # Positions
        x1 = -self.config.horizontal_spacing / 2
        x2 = self.config.horizontal_spacing / 2
        y_center = 0

        # Draw first diagram (D1)
        fig1 = self._draw_subdiagram(
            ax,
            diagram.first,
            x1,
            y_center,
            kept_inputs=diagram.kept_first_inputs,
            kept_outputs=diagram.kept_first_outputs,
        )
        in_ports1, out_ports1 = fig1

        # Draw second diagram (D2)
        fig2 = self._draw_subdiagram(
            ax,
            diagram.second,
            x2,
            y_center,
            kept_inputs=diagram.kept_second_inputs,
            kept_outputs=diagram.kept_second_outputs,
        )
        in_ports2, out_ports2 = fig2

        # Draw forward connections (I1 → I2)
        # I1: indices of outputs from first diagram
        # I2: indices of inputs from second diagram
        # Connection order: I1[k] connects to I2[k]
        for _, (out_idx, in_idx) in enumerate(zip(diagram.I1, diagram.I2, strict=False)):
            if out_idx < len(out_ports1) and in_idx < len(in_ports2):
                y1 = out_ports1[out_idx]
                y2 = in_ports2[in_idx]
                # Draw wire from output of D1 to input of D2
                ax.annotate(
                    "",
                    xy=(x2 - self.config.node_radius, y2),
                    xytext=(x1 + self.config.node_radius, y1),
                    arrowprops={"arrowstyle": "->", "color": "blue", "linewidth": self.config.wire_width},
                )

        # Draw feedback connections (J2 → J1)
        # J2: indices of outputs from second diagram
        # J1: indices of inputs from first diagram
        # Connection order: J2[k] connects to J1[k]
        for k, (out_idx, in_idx) in enumerate(zip(diagram.J2, diagram.J1)):
            if out_idx < len(out_ports2) and in_idx < len(in_ports1):
                y2 = out_ports2[out_idx]
                y1 = in_ports1[in_idx]
                # Draw wire from output of D2 to input of D1 (feedback, above)
                # Use a curved arrow to distinguish from forward connection
                ax.annotate(
                    "",
                    xy=(x1 - self.config.node_radius, y1),
                    xytext=(x2 + self.config.node_radius, y2),
                    arrowprops={
                        "arrowstyle": "->",
                        "color": "red",
                        "connectionstyle": "arc3,rad=0.3",
                        "linewidth": self.config.wire_width,
                    },
                )

        # Draw external input wires (kept inputs of D1 and D2)
        for y in in_ports1:
            ax.plot(
                [-x1 - self.config.node_radius - 0.5, -self.config.node_radius],
                [y, y],
                "k-",
                linewidth=self.config.wire_width,
            )

        # Draw external output wires (kept outputs of D1 and D2)
        # Note: D1 outputs that are not in I1 are external
        for i, y in enumerate(out_ports1):
            if i not in diagram.I1:
                ax.plot(
                    [x1 + self.config.node_radius, x1 + self.config.node_radius + 0.5],
                    [y, y],
                    "k-",
                    linewidth=self.config.wire_width,
                )

        for y in out_ports2:
            if i not in diagram.J2:
                ax.plot(
                    [x2 + self.config.node_radius, x2 + self.config.node_radius + 0.5],
                    [y, y],
                    "k-",
                    linewidth=self.config.wire_width,
                )

    def _draw_subdiagram(
        self,
        ax: plt.Axes,
        diagram: Diagram,
        x: float,
        y: float,
        comp_idx: int | None = None,
        tensor_idx: int | None = None,
        input_positions: list = [],
        comp_size: int | None = None,
        sub_diagram_span: float | None = None,
        spacing: float | None = None,
        radius: float | None = None,
        kept_inputs: Optional[Sequence[int]] = None,
        kept_outputs: Optional[Sequence[int]] = None,
    ) -> tuple[list[float], list[float]]:
        """Draw a sub-diagram at specified coordinates and return port positions.

        Parameters:
        ----------
        ax : plt.Axes
            Matplotlib axes
        diagram : Diagram
            The sub-diagram to draw
        x, y : float
            Center coordinates for the sub-diagram
        kept_inputs : Optional[Sequence[int]]
            Indices of inputs to keep (for ContractedDiagram)
        kept_outputs : Optional[Sequence[int]]
            Indices of outputs to keep (for ContractedDiagram)

        Returns:
        -------
        tuple[list[float], list[float]]
            Lists of y-coordinates for input ports and output ports
        """
        if comp_size is None:
            print("I am here")
            sub_diagram_span = self.config.horizontal_spacing
            spacing = 2 * self.config.horizontal_spacing / 3
            radius = self.config.node_radius
        if isinstance(diagram, ProperDiagram):
            return self._draw_proper_diagram(ax, diagram, x, y, comp_idx, tensor_idx, input_positions, radius)
        if isinstance(diagram, CompositionDiagram):
            return self._draw_composition(
                ax, diagram, x, y, comp_idx, input_positions, comp_size, sub_diagram_span, spacing, radius
            )
        if isinstance(diagram, TensorDiagram):
            return self._draw_tensor(ax, diagram, x, y, comp_idx, input_positions)
        # if isinstance(diagram, ContractedDiagram):
        #     return self._draw_contracted_at(ax, diagram, x, y, kept_inputs, kept_outputs)
        return [], []

    def _draw_swap(
        self,
        ax: plt.Axes,
        x: float,
        y: float,
        comp_idx: int | None = None,
        tensor_idx: int | None = None,
        input_positions: list = [],
    ) -> list[tuple]:
        """Draw a swap node."""
        r = self.config.node_radius
        # Draw X shape
        ax.plot([x - r, x + r], [y - r, y + r], "k-", linewidth=self.config.wire_width)
        ax.plot([x - r, x + r], [y + r, y - r], "k-", linewidth=self.config.wire_width)

        # We will draw input and output wires depending of the block is part of
        # a composition diagram
        draw_in_wires = True
        draw_out_wires = True
        if comp_idx is not None and comp_idx != -1:
            draw_out_wires = False
        # Draw inputs arrows
        if draw_in_wires:
            if not input_positions:
                input_positions = [
                    (x + r + self.config.arrow_length, y + r),
                    (x + r + self.config.arrow_length, y - r),
                ]
            assert len(input_positions) == 2
            input1 = patches.FancyArrowPatch(
                input_positions[0],
                (x + r, y + r),
                arrowstyle="->",
                ec="black",
                mutation_scale=20,
                linewidth=self.config.wire_width,
            )
            input2 = patches.FancyArrowPatch(
                input_positions[1],
                (x + r, y - r),
                arrowstyle="->",
                ec="black",
                mutation_scale=20,
                linewidth=self.config.wire_width,
            )
            ax.add_patch(input1)
            ax.add_patch(input2)

        # Draw Outputs
        output_positions = [
            (x - r, y + r),
            (x - r, y - r),
        ]
        if draw_out_wires:
            output1 = patches.FancyArrowPatch(
                output_positions[0],
                (x - r - self.config.arrow_length, y + r),
                arrowstyle="->",
                ec="black",
                mutation_scale=20,
                linewidth=self.config.wire_width,
            )
            output2 = patches.FancyArrowPatch(
                output_positions[1],
                (x - r - self.config.arrow_length, y - r),
                arrowstyle="->",
                ec="black",
                mutation_scale=20,
                linewidth=self.config.wire_width,
            )
            ax.add_patch(output1)
            ax.add_patch(output2)
        return output_positions

    def _draw_fourier(
        self,
        ax: plt.Axes,
        x: float,
        y: float,
        diagram: Diagram,
        comp_idx: int | None = None,
        tensor_idx: int | None = None,
        input_positions: list = [],
    ) -> list[tuple]:
        """Draw a Fourier node."""
        r = self.config.node_radius
        symbol = "F"
        if isinstance(diagram, FourierInv):
            symbol = "F†"
            ax.plot([x - r, x - r], [y - r, y + r], "k-", linewidth=self.config.wire_width)
            ax.plot([x - r, x + r], [y + r, y], "k-", linewidth=self.config.wire_width)
            ax.plot([x - r, x + r], [y - r, y], "k-", linewidth=self.config.wire_width)
            ax.text(x, y, symbol, ha="center", va="center", fontsize=14 / r)
        elif isinstance(diagram, Fourier2):
            symbol = "F²"
            ax.plot([x - r, x], [y, y + r], "k-", linewidth=self.config.wire_width)
            ax.plot([x - r, x], [y, y - r], "k-", linewidth=self.config.wire_width)
            ax.plot([x + r, x], [y, y - r], "k-", linewidth=self.config.wire_width)
            ax.plot([x + r, x], [y, y + r], "k-", linewidth=self.config.wire_width)
            ax.text(x, y, symbol, ha="center", va="center", fontsize=14 / r)
        else:
            ax.plot([x + r, x + r], [y - r, y + r], "k-", linewidth=self.config.wire_width)
            ax.plot([x + r, x - r], [y + r, y], "k-", linewidth=self.config.wire_width)
            ax.plot([x + r, x - r], [y - r, y], "k-", linewidth=self.config.wire_width)
            ax.text(x, y, symbol, ha="center", va="center", fontsize=14 / r)

        # We will draw input and output wires depending of the block is part of
        # a composition diagram
        draw_in_wires = True
        draw_out_wires = True
        if comp_idx is not None and comp_idx != -1:
            draw_out_wires = False
        # Draw inputs wires
        if draw_in_wires:
            if not input_positions:
                input_positions = [(x + r + self.config.arrow_length, y)]
            assert len(input_positions) == 1
            input1 = patches.FancyArrowPatch(
                input_positions[0],
                (x + r, y),
                arrowstyle="->",
                ec="black",
                mutation_scale=20,
                linewidth=self.config.wire_width,
            )
            ax.add_patch(input1)
        # Draw output wires
        output_positions = [(x - r - self.config.arrow_length, y)]
        if draw_out_wires:
            output1 = patches.FancyArrowPatch(
                output_positions[0],
                (x - r, y),
                arrowstyle="->",
                ec="black",
                mutation_scale=20,
                linewidth=self.config.wire_width,
            )
            ax.add_patch(output1)
        return output_positions

    def _draw_scalar(self, ax: plt.Axes, diagram: ScalarDiagram) -> None:
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

    def _wire_offset(self, index: int, total: int) -> float:
        """Calculate vertical offset for wire at given index."""
        if total <= 1:
            return 0
        spacing = 0.6
        start = -(total - 1) * spacing / 2
        return start + index * spacing

    def _format_phase(self, phase: ZxPoly) -> str:
        """Format phase polynomial for display."""
        if phase.is_zero():
            return ""
        terms = []
        for d, c in sorted(phase.coeffs.items()):
            if d == 0:
                terms.append(f"{c:.2f}")
            elif d == 1:
                terms.append(f"{c:.2f}x")
            elif d == 2:
                terms.append(f"{c:.2f}x²")
            else:
                terms.append(f"{c:.2f}x^{d}")
        result = " + ".join(terms)
        if len(result) > 20:
            result = result[:17] + "..."
        return result

    def _get_spider_type(self, diagram: ProperDiagram) -> str:
        """Get spider type string from diagram instance."""
        if isinstance(diagram, QSpider):
            return "q"
        if isinstance(diagram, PSpider):
            return "p"
        # if isinstance(diagram, MacronodeSpider):
        #     return "macronode"
        # if isinstance(diagram, NonGaussianSpider):
        #     return "non_gaussian"
        if isinstance(diagram, Swap):
            return "swap"
        if isinstance(diagram, (Fourier, FourierInv, Fourier2)):
            return "fourier"
        return "default"

    def _get_input_ports_at(self, diagram: Diagram, x: float, y: float) -> list[float]:
        """Get input port positions for a diagram at given coordinates."""
        if isinstance(diagram, ProperDiagram):
            return [y + self._wire_offset(i, diagram.num_inputs) for i in range(diagram.num_inputs)]
        elif isinstance(diagram, CompositionDiagram) and diagram.diagrams:
            # For composition, inputs come from first sub-diagram
            return self._get_input_ports_at(diagram.diagrams[0], x, y)
        elif isinstance(diagram, TensorDiagram):
            # For tensor, inputs are stacked
            ports = []
            offset = 0
            for sub in diagram.diagrams:
                sub_ports = self._get_input_ports_at(sub, x, y + offset)
                ports.extend(sub_ports)
                offset += len(sub_ports) * 0.6
            return ports
        return []

    def _get_output_ports_at(self, diagram: Diagram, x: float, y: float) -> list[float]:
        """Get output port positions for a diagram at given coordinates."""
        if isinstance(diagram, ProperDiagram):
            return [y + self._wire_offset(i, diagram.num_outputs) for i in range(diagram.num_outputs)]
        elif isinstance(diagram, CompositionDiagram) and diagram.diagrams:
            # For composition, outputs come from last sub-diagram
            return self._get_output_ports_at(diagram.diagrams[-1], x, y)
        elif isinstance(diagram, TensorDiagram):
            # For tensor, outputs are stacked
            ports = []
            offset = 0
            for sub in diagram.diagrams:
                sub_ports = self._get_output_ports_at(sub, x, y + offset)
                ports.extend(sub_ports)
                offset += len(sub_ports) * 0.6
            return ports
        return []


def visualize(diagram: Diagram, title: str = "", config: Optional[VisualizerConfig] = None) -> plt.Figure:
    """Convenience function to visualize a diagram.

    Parameters:
    ----------
    diagram : Diagram
        The diagram to visualize.
    title : str
        Title for the figure.
    config : Optional[VisualizerConfig]
        Visualization configuration.

    Returns:
    -------
    plt.Figure
        Matplotlib figure.
    """
    visualizer = DiagramVisualizer(config)
    return visualizer.visualize(diagram, title)


if __name__ == "__main__":
    p = ZxPoly({1: 2, 2: 4})
    n = 1
    m = 1
    a = PSpider(n, m, p)
    b = QSpider(n, m, p)
    c = QSpider(n, m, p)
    d = Swap()
    # e = d.compose(c)
    e = a.tensor(b)
    f = e.compose(d)
    g = d.compose(f)
    h = a.compose(b)
    h = h.compose(b)
    i = a.tensor(h)
    i = i.tensor(b)
    i = i.tensor(h)

    # print(isinstance(a, ProperDiagram))
    fig = visualize(i, "tensor")
    fig.savefig("swap.png")
