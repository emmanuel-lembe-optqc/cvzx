"""Visualization for CV ZX diagrams.

This module provides visualization capabilities for:
    - ProperDiagram (spiders, gates)
    - CompositionDiagram (sequential composition)
    - TensorDiagram (parallel composition)
    - ContractedDiagram (partial trace with connections)

The visualizer uses matplotlib to draw diagrams in a way that respects
the input/output wire ordering and connection indices. `DiagramVisualizer`
itself is assembled from one mixin per diagram-type/concern, each in its own
module (`cvzx.visualization.proper`, `_composition`, `_contracted`,
`_swap_fourier`, `_overflow`) -- this module holds the shared orchestration
(`visualize()`, `vertical_shift_in_contraction`, `_draw_sub_diagram`) that
ties them together, plus `VisualizerConfig` and the public `visualize()`
convenience function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from cvzx.backends.nx.graph import to_graph
from cvzx.ir.base import CompositionDiagram, ContractedDiagram, Diagram, ProperDiagram, TensorDiagram
from cvzx.ir.gates import CompactDiagram
from cvzx.visualization.composition import _CompositionMixin
from cvzx.visualization.contracted import _ContractedMixin
from cvzx.visualization.overflow import _PhaseOverflowMixin
from cvzx.visualization.proper import _ProperDiagramMixin
from cvzx.visualization.swap_fourier import _SwapFourierMixin

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.text import Text

    from cvzx.visualization.geometry import Position


@dataclass
class VisualizerConfig:
    """Configuration for diagram visualization.

    Attributes
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
            "compact": "lightgray",
            "gaussian": "violet",
            "non_gaussian": "gold",
            "contraction_out": "gray",
            "contraction_in": "violet",
            "boundary": "white",
            "default": "white",
        }
    )

    def __post_init__(self) -> None:
        """Validate parameters and compute derived spacing values.

        Raises
        ------
        ValueError
            If vertical_factor <= 2.
        """
        if self.vertical_factor <= 2:  # ruff: ignore[magic-value-comparison]
            msg = f"vertical_factor must be > 2, got {self.vertical_factor}"
            raise ValueError(msg)

        # Derived spacing values (strict formulas)
        self.arrow_length = 2 * self.node_radius
        self.horizontal_spacing = self.node_radius * 2 + self.arrow_length * 2
        self.vertical_spacing = self.node_radius * self.vertical_factor
        self.contraction_shift = 2 * (self.vertical_spacing - self.node_radius)


class DiagramVisualizer(
    _ProperDiagramMixin,
    _CompositionMixin,
    _ContractedMixin,
    _SwapFourierMixin,
    _PhaseOverflowMixin,
):
    """Visualizer for CV ZX diagrams.

    This class handles layout and drawing of various diagram types,
    ensuring proper wire connections and index ordering. The per-diagram-type
    drawing methods themselves live on the mixins above (one per diagram
    type/concern, see this module's docstring); this class provides the
    shared orchestration and state they all rely on.
    """

    def __init__(self, config: VisualizerConfig | None = None) -> None:
        """Initialize visualizer with configuration."""
        self.config = config or VisualizerConfig()
        # Per-render vertical-phase-overflow state -- reset at the start of every
        # `visualize()` call (see there), not just here, so a `DiagramVisualizer`
        # reused across multiple renders doesn't leak state between them.
        self._overflow_counter: int = 0
        self._overflow_legend: dict[str, str] = {}
        self._phase_overflow_candidates: list[tuple[Text, float, str]] = []

    def visualize(self, diagram: Diagram, title: str = "") -> plt.Figure:
        """Visualize a diagram.

        Parameters
        ----------
        diagram : Diagram
            The diagram to visualize.
        title : str
            Title for the figure.

        Returns
        -------
        plt.Figure
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_aspect("equal")
        ax.axis("off")

        # Build the graph corresponding to diagram
        self.cvzx_graph = to_graph(diagram)
        self.graph = self.cvzx_graph.graph
        self.reg = self.cvzx_graph.registry
        self._overflow_counter = 0
        self._overflow_legend = {}
        self._phase_overflow_candidates = []

        if title:
            ax.set_title(title, fontsize=self.config.fontsize)

        if isinstance(diagram, (ProperDiagram, CompactDiagram)):
            self._draw_proper_diagram(ax, diagram)
        elif isinstance(diagram, CompositionDiagram):
            self._draw_composition(ax, diagram)
        elif isinstance(diagram, TensorDiagram):
            self._draw_tensor(ax, diagram)
        elif isinstance(diagram, ContractedDiagram):
            self._draw_contracted(ax, diagram)
        else:
            ax.text(
                0.5, 0.5, f"Unknown diagram type: {type(diagram)}", ha="center", va="center", transform=ax.transAxes
            )
        self._draw_feedforward(ax)
        # Force the axes to (re)compute their view limits from every patch
        # added while drawing (rectangles, arrows, ...). Matplotlib updates
        # `ax.dataLim` as patches are added, but does not always mark the
        # view as stale purely from `add_patch` calls -- in this version,
        # nothing here reliably requests an autoscale unless something else
        # (e.g. a `Line2D`) is added too. Previously that "something else"
        # was an incidental no-op `ax.plot()` call buried inside the
        # QSpider/PSpider/CompactDiagram drawing branch, which only ran for
        # the last sub-diagram of a composition *and* only when that last
        # sub-diagram happened to be one of those types. Whenever the last
        # sub-diagram was anything else (a `VoidDiagram`, `Swap`, Fourier
        # gate, ...) the axes were left at matplotlib's default (0, 1)x(0, 1)
        # view, so the real content -- however far from the origin -- was
        # rendered as if squeezed into (or ballooned out of) that tiny
        # window, e.g. as one giant, mostly-blank box. Calling `relim` and
        # `autoscale_view` explicitly and unconditionally here makes the
        # view limits always reflect the actual drawn content, regardless
        # of which diagram type ends up last.
        ax.relim()
        ax.autoscale_view()
        plt.tight_layout()
        self._resolve_phase_overflow(fig, ax)
        return fig

    def vertical_shift_in_contraction(  # ruff: ignore[complex-structure]
        self,
        diagram: Diagram,
        through_tensor: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument]
        is_sub_tensor: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument]
        radius: float | None = None,
    ) -> tuple[float, bool]:
        """Find the vertical shift corresponding to a diagram in a contracted diagram.

        Parameters
        ----------
        diagram : Diagram
            Input diagram.
        through_tensor : bool
            True if the diagram contains a TensorDiagram. This variable is useful
            when a diagram is a ContractedDiagram.
        is_sub_tensor : bool
            True if the diagram is a sub_diagram of a ContractDiagram. This allows
            to resize the radius to compute properly the vertical shift.
        radius : float
            radius of the diagram.

        Returns
        -------
        tuple[float, bool]
            The vertical shift, and whether the diagram goes through a tensor diagram.
        """
        if radius is None:
            radius = self.config.node_radius
        if isinstance(diagram, CompositionDiagram):
            shift = 0.0
            next_value = False
            diagram_length = len(diagram.diagrams)
            for sub_diag in diagram.diagrams:
                sub_radius = 3 * radius / (2 * diagram_length + 1) if is_sub_tensor else radius
                sub_shift, value = self.vertical_shift_in_contraction(
                    sub_diag, through_tensor, is_sub_tensor, sub_radius
                )
                shift = max(sub_shift, shift)
                next_value = next_value or value
            return shift, next_value
        if isinstance(diagram, TensorDiagram):
            shift = 0.0
            vertical_spacing = self.config.vertical_factor * radius
            for sub_diag in diagram.diagrams:
                if isinstance(sub_diag, (ProperDiagram, CompositionDiagram)):
                    shift += vertical_spacing
                elif isinstance(sub_diag, (TensorDiagram, ContractedDiagram)):
                    # NOTE: pre-existing bug fixed here — this referenced an undefined
                    # `sub_radius` (only ever assigned in the CompositionDiagram branch
                    # above, which is not on this code path) and would raise
                    # UnboundLocalError at runtime. `_draw_tensor` passes the same
                    # `radius` unchanged to every sub-diagram, so that is the correct
                    # value to use here too.
                    sub_shift, _ = self.vertical_shift_in_contraction(sub_diag, True, is_sub_tensor, radius)  # ruff: ignore[boolean-positional-value-in-call]
                    shift += sub_shift
            shift -= vertical_spacing
            return shift, True
        if isinstance(diagram, ContractedDiagram):
            shift1, through_tensor1 = self.vertical_shift_in_contraction(
                diagram.first, through_tensor, is_sub_tensor=True, radius=radius
            )
            if shift1 == 0:
                shift1 = self.config.vertical_spacing
            elif through_tensor1:
                shift1 += self.config.vertical_spacing
            shift2, through_tensor2 = self.vertical_shift_in_contraction(
                diagram.second, through_tensor, is_sub_tensor=True, radius=radius
            )
            if shift2 == 0:
                shift2 = self.config.vertical_spacing
            elif through_tensor2:
                shift2 += self.config.vertical_spacing
            through_tensor = through_tensor1 or through_tensor2
            return shift1 + shift2 + self.config.contraction_shift - self.config.vertical_spacing, through_tensor
        return 0.0, through_tensor

    def _draw_sub_diagram(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
        self,
        ax: plt.Axes,
        diagram: Diagram,
        x: float,
        y: float,
        comp_idx: int | None = None,
        sub_comp_idx: int | None = None,
        is_sub_tensor: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument]
        input_positions: list[Position] | None = None,
        radius: float | None = None,
        draw_kept_inputs: Sequence[int] | None = None,
        draw_kept_outputs: Sequence[int] | None = None,
    ) -> tuple[list[Position], list[Position], float | list[float]]:
        """Draw a sub-diagram at specified coordinates and return port positions.

        Parameters
        ----------
        ax : plt.Axes
            Matplotlib axes
        diagram : Diagram
            The sub-diagram to draw
        x: float
            First coordinate of the sub-diagram
        y: float
            Second coordinate of the sub-diagram
        comp_idx: int | None
            Index of a sub-diagram inside a composition. In a composition,
            we don't need to draw the input/output wires of all sub-diagrams.
        sub_comp_idx: int | None
            Index of a proper diagram inside a sub-composition diagram. Along with
            the comp_idx, their value determine whether to draw input and/or output
            wires on the proper diagram.
        is_sub_tensor: bool | None
            Boolean tag to indicate that a composition is a sub-diagram of a tensor
            diagram.
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
        float | list[float]
            radius of either the proper diagram, of the first diagram of a
            composition diagram or the list of radiuses of a tensor diagram.
        """
        if radius is None:
            radius = self.config.node_radius
        if isinstance(diagram, (ProperDiagram, CompactDiagram)):
            return self._draw_proper_diagram(
                ax, diagram, x, y, comp_idx, sub_comp_idx, input_positions, radius, draw_kept_inputs, draw_kept_outputs
            )
        if isinstance(diagram, CompositionDiagram):
            return self._draw_composition(
                ax,
                diagram,
                x,
                y,
                comp_idx,
                sub_comp_idx,
                is_sub_tensor,
                input_positions,
                radius,
                draw_kept_inputs,
                draw_kept_outputs,
            )
        if isinstance(diagram, TensorDiagram):
            return self._draw_tensor(
                ax,
                diagram,
                x,
                y,
                comp_idx,
                sub_comp_idx,
                is_sub_tensor,
                input_positions,
                radius,
                draw_kept_inputs,
                draw_kept_outputs,
            )
        if isinstance(diagram, ContractedDiagram):
            return self._draw_contracted(
                ax,
                diagram,
                x,
                y,
                comp_idx,
                sub_comp_idx,
                is_sub_tensor,
                input_positions,
                radius,
                draw_kept_inputs,
                draw_kept_outputs,
            )
        return [], [], radius


def visualize(diagram: Diagram, title: str = "", config: VisualizerConfig | None = None) -> plt.Figure:
    """Convenience function to visualize a diagram.

    Parameters
    ----------
    diagram : Diagram
        The diagram to visualize.
    title : str
        Title for the figure.
    config : VisualizerConfig | None
        Visualization configuration.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    visualizer = DiagramVisualizer(config)
    return visualizer.visualize(diagram, title)
