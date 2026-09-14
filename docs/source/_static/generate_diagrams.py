"""Regenerate the two architecture PNGs used in `docs/source/dev_guide/architecture.md`.

Run with `python docs/source/_static/generate_diagrams.py` (needs the `cvzx` dev
environment, for `matplotlib`). These are plain data-driven diagrams, not
introspected from the source tree, so whoever changes the module dependency
graph or the `CircuitRepr -> DependencyDAG` pipeline is responsible for
updating the node/edge lists below to match and re-running this script --
matching how `CHANGELOG.md` is maintained by hand alongside the code.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

_OUT_DIR = Path(__file__).parent

_BG = "#F5F7FA"
_BLUE = "#2451D6"
_TEAL = "#12806B"
_PURPLE = "#6A3FB5"
_ORANGE = "#C9701B"
_GRAY = "#5B6472"
_LAVENDER_FILL = "#EDE9FB"
_GRAY_FILL = "#ECEEF2"
_TEXT = "#1A1E26"


def _box(ax, x, y_top, w, h, title, desc, *, edgecolor=_BLUE, fill="white", title_size=13, desc_size=10.3):
    ax.add_patch(
        FancyBboxPatch(
            (x, y_top - h),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.8,
            edgecolor=edgecolor,
            facecolor=fill,
            zorder=3,
        )
    )
    pad = 0.18
    ax.text(x + pad, y_top - pad, title, ha="left", va="top", fontsize=title_size, fontweight="bold", color=edgecolor, zorder=4)
    if desc:
        ax.text(
            x + pad,
            y_top - pad - 0.36,
            desc,
            ha="left",
            va="top",
            fontsize=desc_size,
            color=_TEXT,
            linespacing=1.45,
            zorder=4,
        )
    return (x, y_top, w, h)


def _center_top(box):
    x, y_top, w, _h = box
    return (x + w / 2, y_top)


def _center_bottom(box):
    x, y_top, w, h = box
    return (x + w / 2, y_top - h)


def _elbow(ax, p_from, p_to, *, color=_GRAY, dashed=False, lw=1.6, drop=0.22):
    (x0, y0), (x1, y1) = p_from, p_to
    style = (0, (5, 3)) if dashed else "solid"
    if abs(x0 - x1) < 1e-9:
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": lw, "linestyle": style, "shrinkA": 0, "shrinkB": 0},
            zorder=2,
        )
        return
    y_mid = y0 - drop
    ax.plot([x0, x0], [y0, y_mid], color=color, lw=lw, linestyle=style, zorder=2, solid_capstyle="butt")
    ax.plot([x0, x1], [y_mid, y_mid], color=color, lw=lw, linestyle=style, zorder=2, solid_capstyle="butt")
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x1, y_mid),
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": lw, "linestyle": style, "shrinkA": 0, "shrinkB": 0},
        zorder=2,
    )


def _branch(ax, parent_box, child_boxes, *, color=_GRAY, dashed=False, drop=0.3):
    x0, y_trunk_start = _center_bottom(parent_box)
    y_trunk = y_trunk_start - drop
    ax.plot([x0, x0], [y_trunk_start, y_trunk], color=color, lw=1.6, zorder=2)
    xs = [_center_top(c)[0] for c in child_boxes]
    if len(xs) > 1:
        ax.plot([min(xs), max(xs)], [y_trunk, y_trunk], color=color, lw=1.6, zorder=2)
    style = (0, (5, 3)) if dashed else "solid"
    for c in child_boxes:
        xc, yc = _center_top(c)
        ax.plot([xc, xc], [y_trunk, y_trunk], color=color, lw=1.6, zorder=2)
        ax.annotate(
            "",
            xy=(xc, yc),
            xytext=(xc, y_trunk),
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.6, "linestyle": style, "shrinkA": 0, "shrinkB": 0},
            zorder=2,
        )


def _side_route(ax, parent_box, child_box, lane_x, *, color=_ORANGE, label=""):
    x0, y0 = parent_box[0] + parent_box[2], parent_box[1] - parent_box[3] / 2
    x1, y1 = child_box[0] + child_box[2], child_box[1] - child_box[3] / 2
    style = (0, (4, 3))
    ax.plot([x0, lane_x], [y0, y0], color=color, lw=1.5, linestyle=style, zorder=2)
    ax.plot([lane_x, lane_x], [y0, y1], color=color, lw=1.5, linestyle=style, zorder=2)
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(lane_x, y1),
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.5, "linestyle": style, "shrinkA": 0, "shrinkB": 0},
        zorder=2,
    )
    if label:
        ax.text(lane_x + 0.08, (y0 + y1) / 2, label, ha="left", va="center", fontsize=8.3, color=color, style="italic", rotation=90, zorder=4)


def _row_boxes(ax, level_y, names, *, x0=0.3, x1=12.0, h=1.0, **kw):
    n = len(names)
    gap = 0.4
    w = (x1 - x0 - gap * (n - 1)) / n
    boxes = {}
    for i, (node_id, title, desc, edgecolor) in enumerate(names):
        x = x0 + i * (w + gap)
        boxes[node_id] = _box(ax, x, level_y, w, h, title, desc, edgecolor=edgecolor, **kw)
    return boxes


def generate_module_map() -> None:
    """Regenerate `module_map.png`.

    Layout is by import-depth (longest path from a no-cvzx-import module),
    matching the real, current `from cvzx.<x> import ...` graph across
    `src/cvzx/*.py` -- verified by grepping every module's top-level and
    deferred `cvzx.*` imports directly, not guessed. As in the previous
    version of this diagram, only each node's most architecturally
    relevant direct dependency/dependencies are drawn (a spanning tree,
    not the full transitive edge set) to stay readable; a few edges that
    skip several levels (dashed, routed through the right-hand lane) are
    real deferred/lazy imports in the source, not eager ones -- noted
    where that's the reason for the skip.
    """
    fig, ax = plt.subplots(figsize=(16.5, 18), dpi=140)
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)
    ax.set_xlim(0, 14.9)
    ax.set_ylim(-16.6, 1.5)
    ax.axis("off")

    ax.text(7, 1.25, "cvzx module dependency graph", ha="center", va="top", fontsize=21, fontweight="bold", color=_TEXT)
    ax.text(
        7,
        0.75,
        "each arrow points from a module to the module(s) built directly on top of it (a spanning tree, not every import edge)",
        ha="center",
        va="top",
        fontsize=11,
        color=_GRAY,
        style="italic",
    )

    row_h = 1.95
    box_h = 1.25
    y = {lvl: 0.15 - lvl * row_h for lvl in range(8)}

    l0 = _row_boxes(
        ax,
        y[0],
        [
            ("exceptions", "exceptions.py", "CvzxError hierarchy: ParameterError,\nDiagramError, RewriteError, BackendError.", _PURPLE),
            ("config", "config.py", "Backend enum + auto-detecting\nDEFAULT_BACKEND.", _PURPLE),
            ("logging_config", "logging_config.py", "Opt-in setup_file_logging() for the\npipeline's loggers.", _TEAL),
        ],
        h=box_h,
    )
    ax.text(0.3, y[0] - box_h - 0.22, "no internal cvzx imports", fontsize=9.3, color=_PURPLE, style="italic")

    l1 = _row_boxes(
        ax,
        y[1],
        [
            ("base_gates", "base_gates.py", "Diagram hierarchy, ZxPoly, the Parametrized mixin.", _BLUE),
            ("backend", "backend.py", "get_backend_modules() -> (Backend, graph\nmodule, rules module).", _BLUE),
        ],
        h=box_h,
    )
    _elbow(ax, _center_bottom(l0["exceptions"]), _center_top(l1["base_gates"]), color=_PURPLE)
    _elbow(ax, _center_bottom(l0["config"]), _center_top(l1["backend"]), color=_PURPLE)

    l2 = _row_boxes(
        ax,
        y[2],
        [
            ("gates", "gates.py", "CompactDiagram + every gate class.", _BLUE),
            ("completion", "completion.py", "complete_diagram()/complete_boundaries()\nclose open ports.", _ORANGE),
        ],
        h=box_h,
    )
    _elbow(ax, _center_bottom(l1["base_gates"]), _center_top(l2["gates"]))
    _elbow(ax, _center_bottom(l1["backend"]), _center_top(l2["completion"]))

    l3 = _row_boxes(
        ax,
        y[3],
        [
            ("utils", "utils.py", "expand_two_mode_gates() and\nother shared helpers.", _BLUE),
            ("nx_graph", "nx_graph.py", "Diagram <-> networkx.DiGraph,\nGateRegister.", _BLUE),
            ("rx_graph", "rx_graph.py", "Diagram <-> rustworkx.PyDiGraph,\nGateRegister.", _BLUE),
        ],
        h=box_h,
    )
    _branch(ax, l2["gates"], [l3["utils"], l3["nx_graph"], l3["rx_graph"]])

    l4 = _row_boxes(
        ax,
        y[4],
        [
            ("nx_rewrite_rules", "nx_rewrite_rules.py", "Rewrite rules over\na networkx graph.", _BLUE),
            ("rx_rewrite_rules", "rx_rewrite_rules.py", "Rewrite rules over\na rustworkx graph.", _BLUE),
            ("normalize_diagram", "normalize_diagram.py", "Canonicalization pass\n(type-1/type-2 stages).", _BLUE),
            ("visualize_base_gates", "visualize_base_gates.py", "matplotlib rendering\nof a Diagram.", _TEAL),
        ],
        h=box_h,
        title_size=11.3,
        desc_size=9.1,
    )
    _branch(ax, l3["nx_graph"], [l4["nx_rewrite_rules"], l4["normalize_diagram"], l4["visualize_base_gates"]])
    _elbow(ax, _center_bottom(l3["rx_graph"]), _center_top(l4["rx_rewrite_rules"]))
    ax.text(
        _center_top(l4["nx_rewrite_rules"])[0],
        y[4] + 0.22,
        "(+ utils.py)",
        fontsize=7.8,
        color=_GRAY,
        style="italic",
        ha="center",
        va="bottom",
        zorder=4,
    )
    ax.text(
        _center_top(l4["rx_rewrite_rules"])[0],
        y[4] + 0.22,
        "(+ utils.py)",
        fontsize=7.8,
        color=_GRAY,
        style="italic",
        ha="center",
        va="bottom",
        zorder=4,
    )

    l5 = _row_boxes(
        ax,
        y[5],
        [
            ("optimize", "optimize.py", "Backend-dispatched fixed-point\nrewrite loop.", _BLUE),
            ("circuit_to_diagram", "circuit_to_diagram.py", "mqc3 CircuitRepr -> Diagram.", _ORANGE),
            ("diagram_to_circuit", "diagram_to_circuit.py", "Diagram -> mqc3 CircuitRepr.", _ORANGE),
        ],
        h=box_h,
    )
    _branch(ax, l4["normalize_diagram"], [l5["optimize"], l5["circuit_to_diagram"], l5["diagram_to_circuit"]])

    l6 = _row_boxes(
        ax,
        y[6],
        [("dag_extraction", "dag_extraction.py", "Direct CVZXGraph forward sweep -> mqc3 DependencyDAG.", _ORANGE)],
        h=box_h,
    )
    _elbow(ax, _center_bottom(l5["diagram_to_circuit"]), _center_top(l6["dag_extraction"]))

    l7 = _row_boxes(
        ax,
        y[7],
        [("lowering", "lowering.py", 'LoweringBackend plugin registry ("mqc3", "cvzx-direct").', _ORANGE)],
        h=box_h,
    )
    _elbow(ax, _center_bottom(l6["dag_extraction"]), _center_top(l7["lowering"]), dashed=True)

    _side_route(ax, l1["backend"], l5["optimize"], 12.55, label="backend.py")
    _side_route(ax, l2["completion"], l6["dag_extraction"], 13.15, label="completion.py")
    _side_route(ax, l5["diagram_to_circuit"], l7["lowering"], 13.75, label="deferred import")

    ax.text(
        7,
        y[7] - box_h - 0.4,
        "Dashed arrows are deferred (function-local) imports in the real source, not eager top-level ones --\n"
        "lowering.py only imports diagram_to_circuit.py / dag_extraction.py inside to_dependency_dag(), so registering\n"
        "a LoweringBackend never pulls in mqc3 (or the rest of the bridge) until it's actually used.",
        ha="center",
        va="top",
        fontsize=9.6,
        color=_GRAY,
        style="italic",
    )

    fig.tight_layout()
    fig.savefig(_OUT_DIR / "module_map.png", facecolor=_BG)
    plt.close(fig)


def generate_pipeline_overview() -> None:
    """Regenerate `pipeline_overview.png`.

    `CircuitRepr -> DependencyDAG`, ending where `cvzx.lowering`'s real,
    current API actually ends (`graph_to_dependency_dag`) -- the previous
    version of this diagram continued on into a `graph_to_machinery_repr`/
    `MachineryRepr` step that was never actually implemented in
    `lowering.py`; this version stops at `DependencyDAG` and shows the
    remaining mqc3-internal steps (`GraphEmbedder`, `MachineryRepr`) as an
    explicitly out-of-package continuation instead.
    """
    fig, ax = plt.subplots(figsize=(16, 23), dpi=140)
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)
    ax.set_xlim(0, 15)
    ax.set_ylim(-26.9, 1.7)
    ax.axis("off")

    ax.text(7.5, 1.35, "Compiler pipeline overview", ha="center", va="top", fontsize=22, fontweight="bold", color=_TEXT)
    ax.text(
        7.5,
        0.78,
        'CircuitRepr -> DependencyDAG, through optimize() and the cvzx.lowering plugin registry -- every box is the real function/class\n'
        "performing that step; the greyed-out tail is mqc3's own downstream machinery, not part of cvzx",
        ha="center",
        va="top",
        fontsize=10.6,
        color=_GRAY,
        style="italic",
    )

    circuit_repr = _box(ax, 5.4, -0.15, 4.2, 0.75, "CircuitRepr", "", edgecolor=_PURPLE, fill=_LAVENDER_FILL, title_size=15)
    ax.text(7.5, -1.05, "mqc3.circuit.CircuitRepr", ha="center", fontsize=9, color=_GRAY, style="italic")

    from_repr = _box(
        ax,
        3.6,
        -1.55,
        7.8,
        1.05,
        "from_circuit_repr(circuit)",
        "cvzx.circuit_to_diagram -- walks the circuit in time order (_naive_translate),\nproducing a compact-form Diagram, then canonicalizes it with normalize_diagram().",
    )
    _elbow(ax, _center_bottom(circuit_repr), _center_top(from_repr))

    diagram0 = _box(ax, 6.0, -3.05, 3.0, 0.65, "Diagram", "", edgecolor=_GRAY, fill=_GRAY_FILL, title_size=13.5)
    _elbow(ax, _center_bottom(from_repr), _center_top(diagram0))

    opt_x, opt_y, opt_w, opt_h = 1.0, -3.95, 13.0, 5.8
    ax.add_patch(
        FancyBboxPatch(
            (opt_x, opt_y - opt_h),
            opt_w,
            opt_h,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            linewidth=1.9,
            edgecolor=_BLUE,
            facecolor="white",
            zorder=3,
        )
    )
    ax.text(opt_x + 0.25, opt_y - 0.22, "optimize(diagram, *, backend=None, assume_infinite_squeezing=False)", fontsize=13.5, fontweight="bold", color=_BLUE, zorder=4)
    ax.text(
        opt_x + 0.25,
        opt_y - 0.58,
        "repeats the steps below each round until one makes no further change (capped by _MAX_SIMPLIFY_PASSES);\n"
        "backend selects networkx or rustworkx via cvzx.backend.get_backend_modules() (default: auto-detected DEFAULT_BACKEND)",
        fontsize=9.6,
        color=_TEXT,
        zorder=4,
    )
    ax.add_patch(
        FancyBboxPatch((opt_x + opt_w - 3.0, opt_y - 0.62), 2.7, 0.5, boxstyle="round,pad=0.02,rounding_size=0.08", linewidth=1.3, edgecolor=_BLUE, facecolor=_LAVENDER_FILL, zorder=4)
    )
    ax.text(opt_x + opt_w - 1.65, opt_y - 0.37, "cvzx.optimize", ha="center", va="center", fontsize=9.6, fontweight="bold", color=_BLUE, zorder=5)

    _box(ax, opt_x + 0.4, opt_y - 1.15, 4.3, 0.75, "normalize_diagram(diagram)", "", edgecolor=_BLUE)
    _box(ax, opt_x + 0.4, opt_y - 2.15, 4.3, 0.75, "graph_mod.to_graph(diagram)", "", edgecolor=_BLUE)
    _box(ax, opt_x + 5.1, opt_y - 1.15, 3.4, 0.75, "Diagram (canonical)", "", edgecolor=_GRAY, fill=_GRAY_FILL, title_size=11.5)
    _box(ax, opt_x + 5.1, opt_y - 2.15, 3.4, 0.75, "graph (nx/rx)", "", edgecolor=_GRAY, fill=_GRAY_FILL, title_size=11.5)
    ax.annotate("", xy=(opt_x + 5.1, opt_y - 1.55), xytext=(opt_x + 4.7, opt_y - 1.55), arrowprops={"arrowstyle": "-|>", "color": _BLUE, "lw": 1.5})
    ax.annotate("", xy=(opt_x + 5.1, opt_y - 2.55), xytext=(opt_x + 4.7, opt_y - 2.55), arrowprops={"arrowstyle": "-|>", "color": _BLUE, "lw": 1.5})
    ax.text(opt_x + 8.7, opt_y - 2.15, "networkx.DiGraph or rustworkx.PyDiGraph,\nvia cvzx.nx_graph / cvzx.rx_graph", fontsize=8.6, color=_GRAY, style="italic", va="top")

    rewrite_b = _box(
        ax,
        opt_x + 0.4,
        opt_y - 3.75,
        9.9,
        1.15,
        "rewrite rules to a fixed point, then graph_mod.to_diagram()",
        "IdentityRule -> FusionRule -> ChainReductionRule -> FourierNormalizationRule -> TerminalAbsorptionRule [-> CopyRule]",
        edgecolor=_BLUE,
    )
    _elbow(ax, (opt_x + 0.4 + 4.3 / 2, opt_y - 2.15 - 0.75), (opt_x + 0.4 + 4.3 / 2, opt_y - 3.75), color=_BLUE)

    rewrite_bottom = opt_y - 3.75 - 1.15
    ax.text(
        opt_x + 0.25,
        rewrite_bottom - 0.32,
        "-- fed back into normalize_diagram() for another round; stops when a round changes nothing (or the pass cap is hit).\n"
        "remove_void_and_identity_nodes() then runs once, on the final graph only.",
        fontsize=8.9,
        color=_GRAY,
        style="italic",
    )

    _elbow(ax, _center_bottom(diagram0), (opt_x + opt_w / 2, opt_y), color=_GRAY)

    result_b = _box(ax, 5.2, opt_y - opt_h - 0.55, 4.6, 0.7, "OptimizeResult(graph, diagram)", "", edgecolor=_GRAY, fill=_GRAY_FILL, title_size=12)
    _elbow(ax, (opt_x + opt_w / 2, opt_y - opt_h), _center_top(result_b), color=_GRAY)

    y_branch = opt_y - opt_h - 1.9
    vis_b = _box(ax, 0.8, y_branch, 5.6, 1.05, "visualize(diagram, title=...)", "cvzx.visualize_base_gates -- renders the diagram\nas a matplotlib figure for interactive inspection.", edgecolor=_TEAL)
    complete_b = _box(
        ax,
        7.0,
        y_branch,
        7.0,
        1.3,
        "complete_boundaries(diagram, *, backend=None, ...)",
        "cvzx.completion -- closes every open input port with a fresh ideal state and\nevery open output port with a fresh symbolic measurement effect (QSpider/PSpider).",
        edgecolor=_ORANGE,
    )
    ax.text(3.6, y_branch + 0.35, "at the same level:", ha="center", fontsize=9.3, color=_GRAY, style="italic")
    _branch(ax, result_b, [vis_b, complete_b], drop=0.4)

    fig_b = _box(ax, 1.6, y_branch - 1.65, 4.0, 0.65, "matplotlib.figure.Figure", "", edgecolor=_GRAY, fill=_GRAY_FILL, title_size=11)
    _elbow(ax, _center_bottom(vis_b), _center_top(fig_b), dashed=True)
    ax.text(3.6, y_branch - 2.55, "leaf -- not consumed further", fontsize=8.6, color=_ORANGE, style="italic", ha="center")

    diagram_closed = _box(ax, 8.5, y_branch - 1.65, 4.0, 0.7, "Diagram (closed)", "num_inputs = num_outputs = 0", edgecolor=_GRAY, fill=_GRAY_FILL, title_size=11)
    _elbow(ax, _center_bottom(complete_b), _center_top(diagram_closed))

    lower_x, lower_y, lower_w, lower_h = 5.6, y_branch - 2.6, 8.7, 6.7
    ax.add_patch(
        FancyBboxPatch((lower_x, lower_y - lower_h), lower_w, lower_h, boxstyle="round,pad=0.02,rounding_size=0.1", linewidth=1.9, edgecolor=_PURPLE, facecolor="white", zorder=1.5)
    )
    ax.text(lower_x + 0.25, lower_y - 0.22, 'graph_to_dependency_dag(diagram, backend="mqc3" | "cvzx-direct")', fontsize=12.3, fontweight="bold", color=_PURPLE, zorder=4)
    ax.text(
        lower_x + 0.25,
        lower_y - 0.58,
        'cvzx.lowering -- dispatches via the LoweringBackend registry (default backend: "mqc3")',
        fontsize=9.6,
        color=_TEXT,
        zorder=4,
    )
    _elbow(ax, _center_bottom(diagram_closed), (lower_x + lower_w / 2, lower_y), color=_PURPLE)

    sub_top = lower_y - 1.0
    sub_h = 3.15
    mqc3_ref = _box(ax, lower_x + 0.35, sub_top, 3.9, sub_h, 'Mqc3ReferenceBackend ("mqc3")', "", edgecolor=_BLUE, title_size=10.6)
    ax.text(lower_x + 0.55, sub_top - 0.62, "to_circuit_repr(diagram)", fontsize=9.6, fontweight="bold", color=_BLUE)
    ax.text(lower_x + 0.55, sub_top - 0.90, "cvzx.diagram_to_circuit", fontsize=8.2, color=_GRAY, style="italic")
    ax.annotate("", xy=(lower_x + 2.3, sub_top - 1.35), xytext=(lower_x + 2.3, sub_top - 1.0), arrowprops={"arrowstyle": "-|>", "color": _BLUE, "lw": 1.4})
    ax.text(lower_x + 0.55, sub_top - 1.55, "DependencyDAG(circuit)", fontsize=9.6, fontweight="bold", color=_BLUE)
    ax.text(lower_x + 0.55, sub_top - 1.83, "mqc3.graph.embed.dep_dag, from the CircuitRepr", fontsize=8.0, color=_GRAY, style="italic")

    cvzx_direct = _box(ax, lower_x + 4.4, sub_top, 3.9, sub_h, 'CvzxDirectBackend ("cvzx-direct")', "", edgecolor=_BLUE, title_size=10.6)
    ax.text(lower_x + 4.6, sub_top - 0.62, "extract_dependency_dag(diagram)", fontsize=9.6, fontweight="bold", color=_BLUE)
    ax.text(lower_x + 4.6, sub_top - 0.90, "cvzx.dag_extraction", fontsize=8.2, color=_GRAY, style="italic")
    ax.text(
        lower_x + 4.6,
        sub_top - 1.40,
        "dual-backend forward sweep over the diagram's\nown CVZXGraph wire/classical edges -- reuses\ndiagram_to_circuit's per-leaf translators directly.",
        fontsize=8.0,
        color=_TEXT,
        va="top",
        linespacing=1.5,
    )

    _branch(ax, (lower_x + lower_w / 2 - 0.01, lower_y - 0.68, 0.02, 0.0), [mqc3_ref, cvzx_direct], color=_PURPLE, drop=0.18)

    dep_dag = _box(ax, lower_x + 1.75, sub_top - sub_h - 0.55, 5.2, 0.75, "DependencyDAG", "", edgecolor=_PURPLE, fill=_LAVENDER_FILL, title_size=14)
    _branch(ax, mqc3_ref, [dep_dag], color=_BLUE, drop=0.35)
    _branch(ax, cvzx_direct, [dep_dag], color=_BLUE, drop=0.35)

    lower_bottom = lower_y - lower_h
    ax.text(
        lower_x + lower_w / 2,
        lower_bottom + 0.5,
        "Both backends verified to produce an isomorphic DependencyDAG for the same input\n(same nodes/ops/modes/feedforward edges), across both graph backends.",
        ha="center",
        fontsize=8.8,
        color=_GRAY,
        style="italic",
    )

    tail_x, tail_w = lower_x + 0.6, lower_w - 1.2
    embed_b = _box(ax, tail_x, lower_bottom - 0.5, tail_w, 0.75, "GraphEmbedder.embed(dep_dag)", "mqc3.graph.embed -- beamsearch.py / greedy.py", edgecolor=_GRAY, fill=_GRAY_FILL, title_size=10.8)
    _elbow(ax, (lower_x + lower_w / 2, lower_bottom), _center_top(embed_b), color=_GRAY, dashed=True)

    graph_repr_b = _box(ax, tail_x, _center_bottom(embed_b)[1] - 0.35, tail_w, 0.6, "GraphRepr", "", edgecolor=_GRAY, fill=_GRAY_FILL, title_size=11)
    _elbow(ax, _center_bottom(embed_b), _center_top(graph_repr_b), color=_GRAY, dashed=True)

    machinery_b = _box(ax, tail_x, _center_bottom(graph_repr_b)[1] - 0.35, tail_w, 0.75, "MachineryRepr", "mqc3.machinery -- QPU-ready output", edgecolor=_GRAY, fill=_GRAY_FILL, title_size=11)
    _elbow(ax, _center_bottom(graph_repr_b), _center_top(machinery_b), color=_GRAY, dashed=True)

    tail_bottom = _center_bottom(machinery_b)[1]
    ax.text(
        tail_x + tail_w / 2,
        tail_bottom - 0.4,
        "downstream of cvzx -- mqc3's own embedding + machinery pipeline, not implemented in this package",
        ha="center",
        fontsize=9,
        color=_GRAY,
        style="italic",
    )

    ax.text(
        7.5,
        tail_bottom - 1.15,
        "The two representations either side of optimize() (Diagram <-> graph) are converted via graph_mod.to_graph/to_diagram, where\n"
        "graph_mod is whichever of cvzx.nx_graph / cvzx.rx_graph optimize()'s backend= selected. cvzx.lowering is the analogous plugin\n"
        "point one step further downstream: a QPU wanting a different DependencyDAG construction strategy registers its own\n"
        "LoweringBackend (cvzx-direct is one example, skipping the CircuitRepr round-trip) without touching get_backend or any other backend.",
        ha="center",
        va="top",
        fontsize=9.6,
        color=_GRAY,
    )

    fig.tight_layout()
    fig.savefig(_OUT_DIR / "pipeline_overview.png", facecolor=_BG)
    plt.close(fig)


if __name__ == "__main__":
    generate_module_map()
    generate_pipeline_overview()
    print(f"Wrote {_OUT_DIR / 'module_map.png'} and {_OUT_DIR / 'pipeline_overview.png'}")
