"""Vertical phase-text overflow handling: generic labels + a bounded legend.

A node whose wrapped phase text is taller than its own box gets a generic
`"D<n>"` label instead, with the real phase relocated into a legend appended
below the diagram -- see `_PhaseOverflowMixin._resolve_phase_overflow`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.text import Text

    from cvzx.visualization.protocol import Visualizer

# Bounds on the overflow legend, so it stays a small, predictable addition to
# the figure regardless of how many nodes actually overflow.
LEGEND_MAX_LEN = 45
_MAX_LEGEND_ENTRIES = 20
_LEGEND_ROWS_PER_COLUMN = 5
_LEGEND_ROW_HEIGHT_IN = 0.22
_LEGEND_TOP_MARGIN_IN = 0.15


class _PhaseOverflowMixin:
    """Vertical phase-overflow detection, generic-label substitution, and legend rendering."""

    def _register_phase_candidate(self: Visualizer, text_obj: Text, radius: float, legend_str: str) -> None:
        """Record a drawn phase `Text` as a candidate to check for vertical overflow.

        Nothing is decided here -- `ax`'s data-to-pixel mapping isn't final
        until `visualize()`'s own `relim`/`autoscale_view`/`tight_layout`
        sequence runs, so the actual overflow check happens afterward, in
        `_resolve_phase_overflow`.

        Parameters
        ----------
        text_obj : Text
            The `Text` artist `ax.text(...)` returned for this node's phase.
        radius : float
            The node's own `radius` (its box is `2 * radius` square).
        legend_str : str
            The (already length-capped) phase string to show in the legend
            if this candidate turns out to overflow.
        """
        self._phase_overflow_candidates.append((text_obj, radius, legend_str))

    def _resolve_phase_overflow(self: Visualizer, fig: plt.Figure, ax: plt.Axes) -> None:
        """Replace any vertically-overflowing phase `Text` with a generic `D<n>` label.

        Runs once, after `visualize()`'s own layout calls, so the axes'
        data-to-pixel mapping is final: forces one render pass
        (`fig.canvas.draw()`), then for each candidate registered by
        `_register_phase_candidate` compares the phase text's actual
        rendered pixel height against its own box's pixel height (both via
        real matplotlib measurement -- `Text.get_window_extent()` and an
        affine `ax.transData.transform()` -- not a guessed line-height
        heuristic). An overflowing candidate's `Text` is replaced in place
        with `f"D{n}"` (a per-render counter) and the real phase is recorded
        in `self._overflow_legend` instead; `_draw_overflow_legend` then
        renders that mapping onto the figure, if anything overflowed.

        Parameters
        ----------
        fig : plt.Figure
            The figure being finalized.
        ax : plt.Axes
            The axes the candidates were drawn onto.
        """
        if not self._phase_overflow_candidates:
            return
        fig.canvas.draw()
        for text_obj, radius, legend_str in self._phase_overflow_candidates:
            text_px_height = text_obj.get_window_extent().height
            top = ax.transData.transform((0.0, radius))
            bottom = ax.transData.transform((0.0, -radius))
            box_px_height = abs(top[1] - bottom[1])
            if text_px_height > box_px_height:
                self._overflow_counter += 1
                label = f"D{self._overflow_counter}"
                text_obj.set_text(label)
                self._overflow_legend[label] = legend_str
        if self._overflow_legend:
            self._draw_overflow_legend(fig)

    def _draw_overflow_legend(self: Visualizer, fig: plt.Figure) -> None:
        """Append a legend mapping each generic overflow label to its real phase.

        Grows `fig` vertically by a small, bounded amount (independent of
        how many nodes actually overflowed, since entries beyond
        `_MAX_LEGEND_ENTRIES` collapse into one final "... and N more" line)
        and reserves that strip via a second `tight_layout(rect=...)` call so
        the legend never clips or overlaps the diagram above it. Entries are
        laid out into multiple columns so the legend stays compact rather
        than one long vertical list.

        Parameters
        ----------
        fig : plt.Figure
            The figure to append the legend to. `self._overflow_legend` must
            be non-empty (checked by the caller, `_resolve_phase_overflow`).
        """
        entries = list(self._overflow_legend.items())
        shown, remainder = entries[:_MAX_LEGEND_ENTRIES], entries[_MAX_LEGEND_ENTRIES:]
        num_rows = (min(len(shown), _LEGEND_ROWS_PER_COLUMN) if shown else 0) + (1 if remainder else 0)
        _, row_y_frac = self._reserve_legend_strip(fig, num_rows)

        num_columns = min(4, -(-len(shown) // _LEGEND_ROWS_PER_COLUMN)) if shown else 0
        legend_fontsize = max(self.config.fontsize * 0.8, 7)
        for i, (label, phase_text) in enumerate(shown):
            column, row = divmod(i, _LEGEND_ROWS_PER_COLUMN)
            x_frac = 0.02 + column * (0.98 / max(num_columns, 1))
            fig.text(x_frac, row_y_frac(row), f"{label}: {phase_text}", ha="left", va="top", fontsize=legend_fontsize)
        if remainder:
            more_text = f"... and {len(remainder)} more"
            more_y = row_y_frac(_LEGEND_ROWS_PER_COLUMN)
            fig.text(0.02, more_y, more_text, ha="left", va="top", fontsize=legend_fontsize, style="italic")

    def _reserve_legend_strip(
        self: Visualizer, fig: plt.Figure, num_rows: int
    ) -> tuple[float, Callable[[int], float]]:
        """Grow `fig` to fit `num_rows` of legend text and reserve that space via `tight_layout`.

        Parameters
        ----------
        fig : plt.Figure
            The figure to grow.
        num_rows : int
            How many legend text rows to reserve room for.

        Returns
        -------
        tuple[float, Callable[[int], float]]
            The reserved bottom fraction of the figure (`legend_fraction`,
            for the diagram's `tight_layout(rect=...)` upper bound), and a
            `row -> y_frac` function converting a 0-indexed legend row into
            the figure-fraction y-coordinate to draw its text at.
        """
        extra_height_in = _LEGEND_TOP_MARGIN_IN + num_rows * _LEGEND_ROW_HEIGHT_IN
        width_in, height_in = (float(v) for v in fig.get_size_inches())
        total_height_in = height_in + extra_height_in
        fig.set_size_inches(width_in, total_height_in, forward=True)
        legend_fraction = extra_height_in / total_height_in
        plt.tight_layout(rect=(0, legend_fraction, 1, 1))

        def row_y_frac(row: int) -> float:
            return legend_fraction - (_LEGEND_TOP_MARGIN_IN + row * _LEGEND_ROW_HEIGHT_IN) / total_height_in

        return legend_fraction, row_y_frac
