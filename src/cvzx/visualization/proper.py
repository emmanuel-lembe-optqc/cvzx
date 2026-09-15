"""Drawing a `ProperDiagram`/`CompactDiagram` leaf (spiders and gates)."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from cvzx.ir.base import (
    Fourier,
    Fourier2,
    FourierInv,
    ProperDiagram,
    PSpider,
    QSpider,
    Swap,
    VoidDiagram,
    ZxPoly,
)
from cvzx.ir.gates import (
    BeamsplitterGate,
    CompactDiagram,
    ControlledSumGate,
    ControlledZGate,
    CubicPhaseGate,
    DisplacementGate,
    PhaseRotationGate,
    SqueezingGate,
)
from cvzx.visualization.geometry import Position, is_wiring_diagram
from cvzx.visualization.overflow import LEGEND_MAX_LEN

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cvzx.visualization.protocol import Visualizer


class _ProperDiagramMixin:
    """Draws a spider/gate leaf, its wires, and its phase label."""

    def _draw_proper_diagram(  # ruff: ignore[complex-structure, too-many-branches, too-many-arguments, too-many-locals, too-many-statements, too-many-positional-arguments]
        self: Visualizer,
        ax: plt.Axes,
        diagram: ProperDiagram | CompactDiagram,
        x: float = 0,
        y: float = 0,
        comp_idx: int | None = None,
        sub_comp_idx: int | None = None,
        input_positions: list[Position] | None = None,
        radius: float | None = None,
        draw_kept_inputs: Sequence[int] | None = None,
        draw_kept_outputs: Sequence[int] | None = None,
    ) -> tuple[list[Position], list[Position], float]:
        """Draw a proper diagram (single node).

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
            radius of the proper diagram.
        """
        # NOTE: this default was previously only applied inside the QSpider/PSpider/
        # CompactDiagram branch below. Every sibling _draw_* method defaults `radius`
        # unconditionally up front; here, any diagram matching none of the branches
        # below (e.g. VoidDiagram, which isn't handled by this method) would fall
        # through and return radius=None despite the declared `-> float` contract.
        # Hoisted here for consistency; behavior for every diagram type this method
        # already draws (QSpider/PSpider/CompactDiagram/Swap/Fourier/...) is unchanged.
        if radius is None:
            radius = self.config.node_radius
        output_positions: list[Position] = []
        init_input_positions: list[Position] = []
        is_void = isinstance(diagram, VoidDiagram)
        # Determine spider type
        if isinstance(diagram, (QSpider, PSpider, CompactDiagram, VoidDiagram)):
            width = 2 * radius
            height = 2 * radius
            arrow_length = 2 * radius
            y_offset_in = [float(v) for v in np.linspace(0, height, diagram.num_inputs + 2)]
            # Remove the two edges
            y_offset_in.pop(0)
            y_offset_in.pop(-1)
            y_offset_out = [float(v) for v in np.linspace(0, height, diagram.num_outputs + 2)]
            # Remove the two edges
            y_offset_out.pop(0)
            y_offset_out.pop(-1)
            pivot = (x - radius, y - radius)
            init_input_positions = [
                (pivot[0] + width + arrow_length, pivot[1] + y_offset_in[i]) for i in range(diagram.num_inputs)
            ]

            if not is_void:
                # Spider diagram
                spider_type = self._get_spider_type(diagram)
                color = self.config.colors.get(spider_type, self.config.colors["default"])

                # Draw node
                pivot = (x - radius, y - radius)
                box = patches.Rectangle(pivot, width, height, facecolor=color)
                if is_wiring_diagram(diagram):
                    box = patches.Rectangle(pivot, width, height, facecolor="white", edgecolor=color)
                ax.add_patch(box)

                # Draw phase if present
                phase = diagram.phase if isinstance(diagram, (QSpider, PSpider)) else diagram.label  # type: ignore[union-attr]
                phase_str = self._format_phase(phase)
                legend_str = self._format_phase(phase, max_len=LEGEND_MAX_LEN)

                # Wrap text
                phase_str = textwrap.fill(phase_str, width=max(int(width), 1))
                text_obj = ax.text(
                    x,
                    y,
                    phase_str,
                    ha="center",
                    va="center",
                    fontsize=self.config.fontsize,
                    clip_on=True,
                )
                if phase_str:
                    self._register_phase_candidate(text_obj, radius, legend_str)
            # We will draw input and output wires depending of the block is part of
            # a composition diagram
            draw_in_wires = True
            draw_out_wires = True
            if (  # ruff: ignore[too-many-boolean-expressions]
                comp_idx is not None and comp_idx != -1
            ) or ((comp_idx == -1 or comp_idx is None) and (sub_comp_idx is not None and sub_comp_idx != -1)):
                draw_out_wires = False
            if draw_kept_inputs is None:
                draw_kept_inputs = range(diagram.num_inputs)
            if draw_kept_outputs is None:
                draw_kept_outputs = range(diagram.num_outputs)
            # Draw input wires (left side)
            if draw_in_wires and not is_void:
                if input_positions is None:
                    # input_positions is empty only for the first element of a composition
                    # or a single proper diagram
                    input_positions = init_input_positions
                for i in range(diagram.num_inputs):
                    if i in draw_kept_inputs:
                        if sub_comp_idx is not None:
                            input_pos = (
                                input_positions[i]
                                if (sub_comp_idx == 0 and (comp_idx is None or comp_idx == 0))
                                else input_positions[draw_kept_inputs.index(i)]
                            )
                        else:
                            input_pos = (
                                input_positions[i]
                                if (comp_idx is None or comp_idx == 0)
                                else input_positions[draw_kept_inputs.index(i)]
                            )
                        # A Hack to not draw outgoing arrows from void diagrams
                        if input_pos:
                            input_i = patches.FancyArrowPatch(
                                input_pos,
                                (pivot[0] + arrow_length, pivot[1] + y_offset_in[i]),
                                arrowstyle="->",
                                ec="black",
                                mutation_scale=20,
                                linewidth=self.config.wire_width,
                            )
                            ax.add_patch(input_i)

            # Draw output wires (right side)
            output_positions = [(pivot[0], pivot[1] + y_offset_out[i]) for i in range(diagram.num_outputs)]
            if draw_out_wires:
                for i in range(diagram.num_outputs):
                    if i in draw_kept_outputs:
                        output_i = patches.FancyArrowPatch(
                            output_positions[i],
                            (pivot[0] - arrow_length, pivot[1] + y_offset_out[i]),
                            arrowstyle="->",
                            ec="black",
                            mutation_scale=20,
                            linewidth=self.config.wire_width,
                        )
                        ax.add_patch(output_i)
                ax.plot()
            # A Hack to not draw outgoing arrows from void diagrams
            if is_void:
                output_positions = [() for i in range(diagram.num_outputs)]  # type: ignore[misc]
        elif isinstance(diagram, Swap):
            output_positions, init_input_positions, radius = self._draw_swap(
                ax, x, y, diagram, comp_idx, sub_comp_idx, input_positions, radius, draw_kept_inputs, draw_kept_outputs
            )
        elif isinstance(diagram, (Fourier, FourierInv, Fourier2)):
            output_positions, init_input_positions, radius = self._draw_fourier(
                ax, x, y, diagram, comp_idx, sub_comp_idx, input_positions, radius, draw_kept_inputs, draw_kept_outputs
            )
        self.graph.nodes[diagram.id]["pos"] = (x, y)
        self.graph.nodes[diagram.id]["radius"] = radius
        return output_positions, init_input_positions, radius

    def _format_phase(self: Visualizer, phase: ZxPoly | str, max_len: int = 20) -> str:  # ruff: ignore[too-many-branches]
        """Format phase polynomial or Compact Diagram label for display.

        Parameters
        ----------
        phase: ZxPoly | str
            Real polynomial describing a p/q spider. Or label denoting a
            CompactDiagram
        max_len: int
            Maximum length of the returned string; longer results are
            truncated with a trailing "...". Callers needing a shorter
            in-box label vs. a more generous legend entry pass different
            values here rather than truncating twice.

        Returns
        -------
        str
            String display of the phase polynomial
        """
        if isinstance(phase, str):
            return phase if len(phase) <= max_len else phase[: max_len - 3] + "..."
        if phase.is_zero:
            return ""
        terms = []
        for d, c in sorted(phase.coeffs.items()):
            if d == 0:
                if isinstance(c, float):
                    terms.append(f"{c:.2f}")
                else:
                    terms.append(f"{c}")
            elif d == 1:
                if isinstance(c, float):
                    terms.append(f"{c:.2f}x")
                else:
                    terms.append(f"{c}x")
            else:  # ruff: ignore[collapsible-else-if]
                if isinstance(c, float):
                    terms.append(f"{c:.2f}x^{d}")
                else:
                    terms.append(f"{c}x^{d}")
        result = " + ".join(terms)
        if len(result) > max_len:
            result = result[: max_len - 3] + "..."
        return result

    def _get_spider_type(self: Visualizer, diagram: ProperDiagram | CompactDiagram) -> str:  # ruff: ignore[too-many-return-statements]
        """Get spider type string from diagram instance.

        Parameters
        ----------
            diagram: ProperDiagram | CompactDiagram
                Input proper diagram.

        Returns
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
        if isinstance(diagram, CompactDiagram):
            return "compact"
        # NOTE: pre-existing bug fixed here — this called isinstance() with the
        # gate classes themselves instead of `diagram` as the first argument (an
        # invalid isinstance() call). Fixed to check `diagram` against the gate
        # classes, though this branch (and the CubicPhaseGate one below) is still
        # unreachable in practice: every one of these gate types is a CompactDiagram
        # subclass, so the `isinstance(diagram, CompactDiagram)` check above already
        # returns "compact" for them first. Left as-is since reordering those checks
        # would be a further, separate behavior change (which colors gates get)
        # beyond fixing this isinstance() call.
        if isinstance(
            diagram,
            (BeamsplitterGate, ControlledSumGate, ControlledZGate, DisplacementGate, PhaseRotationGate, SqueezingGate),
        ):
            return "gaussian"
        if isinstance(diagram, CubicPhaseGate):
            return "nongaussian"
        return "default"
