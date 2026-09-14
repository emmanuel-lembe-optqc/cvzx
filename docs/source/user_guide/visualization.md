# Visualizing diagrams

`cvzx.utils.visualization_base_gates.visualize(diagram, title="", config=None) -> matplotlib.figure.Figure`
draws any `Diagram` — spiders, gates (drawn compact unless expanded), tensors, compositions,
and contractions (including a contraction nested inside another one):

```python
from cvzx.ir.gates import DisplacementGate, PhaseRotationGate
from cvzx.utils.visualization_base_gates import visualize

D = DisplacementGate(alpha=1.0 + 0.5j)
R = PhaseRotationGate(theta=0.7)
comp = D.compose(R, connectivity={0: 0})

fig = visualize(comp, title="D ∘ R with connectivity (0→0)")
fig.savefig("D_R.png", dpi=150, bbox_inches="tight")
```

## Layout configuration

Pass a `cvzx.utils.visualization_base_gates.VisualizerConfig` to `visualize()` to control node size,
vertical spacing, wire width, font size, and the color used per spider/gate category
(`q`, `p`, `macronode`, `compact`, `gaussian`, `non_gaussian`, ...):

```python
from cvzx.utils.visualization_base_gates import VisualizerConfig, visualize

config = VisualizerConfig(node_radius=8, vertical_factor=2.5, fontsize=12)
fig = visualize(comp, config=config)
```

`vertical_factor` must be greater than 2 (`VisualizerConfig.__post_init__` raises
`ValueError` otherwise) — it and `node_radius` are what the derived spacing values
(`horizontal_spacing`, `vertical_spacing`, `contraction_shift`, ...) are computed from.

For repeated use, `cvzx.utils.visualization_base_gates.DiagramVisualizer(config)` holds a config and
exposes the same `.visualize(diagram, title)` method `visualize()` delegates to.

## Before/after comparisons

`cvzx.utils.visualization_base_gates.visualize_before_after(diagram_before, diagram_after, test_name,
rule_name)` renders both diagrams as a single side-by-side figure and saves it to
`test_images_{rule_name}/{test_name}.png` — handy when developing or debugging a rewrite
rule, to see at a glance what one application of the rule actually changed:

```python
from cvzx.backends.nx.rules import IdentityRule, apply_rule_to_diagram
from cvzx.utils.visualization_base_gates import visualize_before_after

before = comp  # a diagram containing an identity spider
after = apply_rule_to_diagram(IdentityRule(), before)
visualize_before_after(before, after, test_name="my_case", rule_name="identity_rule")
```
