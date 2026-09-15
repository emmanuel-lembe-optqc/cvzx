"""Structural type shared by every `DiagramVisualizer` mixin's `self`.

`DiagramVisualizer` (in `cvzx.visualization.core`) is assembled from several
mixins, each in its own module (`visualization_proper`, `_composition`,
`_contracted`, `_swap_fourier`, `_overflow`) -- see that module's docstring.
Every mixin method calls across into attributes/methods defined on *other*
mixins or on `DiagramVisualizer` itself (e.g. `self.config`, or
`self._draw_sub_diagram` from within the composition mixin). Typing `self`
directly as `DiagramVisualizer` doesn't work here: mypy requires an explicit
`self` type to be a *nominal* supertype of the class the method is defined
on, and a bare mixin (no shared base class) never satisfies that. A
`Protocol` sidesteps this entirely via structural typing instead -- every
mixin method types `self` as `Visualizer` below, and `DiagramVisualizer`
satisfies it by construction (it has every member listed here).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import matplotlib.pyplot as plt
    from matplotlib.text import Text

    from cvzx.ir.base import CompositionDiagram, ContractedDiagram, Diagram, ProperDiagram, Swap, TensorDiagram, ZxPoly
    from cvzx.ir.gates import CompactDiagram
    from cvzx.visualization.core import VisualizerConfig
    from cvzx.visualization.geometry import Position


class Visualizer(Protocol):
    """Everything a `DiagramVisualizer` mixin method may call on `self`."""

    config: VisualizerConfig
    graph: Any
    reg: Any
    cvzx_graph: Any
    _overflow_counter: int
    _overflow_legend: dict[str, str]
    _phase_overflow_candidates: list[tuple[Text, float, str]]

    def _draw_proper_diagram(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
        self,
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
    ) -> tuple[list[Position], list[Position], float]: ...

    def _draw_composition(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
        self,
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
    ) -> tuple[list[Position], list[Position], float | list[float]]: ...

    def _draw_tensor(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
        self,
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
    ) -> tuple[list[Position], list[Position], float | list[float]]: ...

    def _draw_contracted(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
        self,
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
    ) -> tuple[list[Position], list[Position], float]: ...

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
    ) -> tuple[list[Position], list[Position], float | list[float]]: ...

    def _draw_swap(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
        self,
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
    ) -> tuple[list[Position], list[Position], float]: ...

    def _draw_fourier(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
        self,
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
    ) -> tuple[list[Position], list[Position], float]: ...

    def _draw_feedforward(self, ax: plt.Axes) -> None: ...

    def vertical_shift_in_contraction(
        self,
        diagram: Diagram,
        through_tensor: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument]
        is_sub_tensor: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument]
        radius: float | None = None,
    ) -> tuple[float, bool]:
        """Find the vertical shift corresponding to a diagram in a contracted diagram."""
        ...

    def _get_spider_type(self, diagram: ProperDiagram | CompactDiagram) -> str: ...

    def _format_phase(self, phase: ZxPoly | str, max_len: int = 20) -> str: ...

    def _register_phase_candidate(self, text_obj: Text, radius: float, legend_str: str) -> None: ...

    def _resolve_phase_overflow(self, fig: plt.Figure, ax: plt.Axes) -> None: ...

    def _draw_overflow_legend(self, fig: plt.Figure) -> None: ...

    def _reserve_legend_strip(self, fig: plt.Figure, num_rows: int) -> tuple[float, Callable[[int], float]]: ...
