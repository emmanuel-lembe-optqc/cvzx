# Quickstart

## Building a diagram

A `Diagram` is built out of leaves (spiders and gates) combined with
`CompositionDiagram` (sequential) and `TensorDiagram` (parallel):

```python
from sympy import pi

from cvzx.ir.base import CompositionDiagram
from cvzx.ir.gates import PhaseRotationGate

theta1, theta2, theta3 = pi / 6, pi / 5, pi / 7
comp = CompositionDiagram([
    PhaseRotationGate(theta1),
    PhaseRotationGate(theta2),
    PhaseRotationGate(theta3),
])
```

`CompositionDiagram([d1, d2, d3])` composes causally left to right — `d1`'s outputs feed
`d2`'s inputs, and so on (see {doc}`../theory` for the exact correspondence with the paper's
$D_3 \circ D_2 \circ D_1$ notation). Every gate class also exposes `.compose(other)` and
`.tensor(other)` methods if you prefer building a circuit that way instead of nested list
literals.

Every `Diagram` has `.num_inputs`, `.num_outputs`, and a unique `.id` (assigned at
construction time, used by the rewrite rules and the graph conversion to refer back to a
specific leaf):

```python
>>> comp.num_inputs, comp.num_outputs
(1, 1)
```

## Converting to and from the graph representation

`cvzx.backends.nx.graph.to_graph`/`to_diagram` convert losslessly between a `Diagram` tree and its
`networkx.DiGraph` form (the representation the rewrite rules actually operate on — see
{doc}`../dev_guide/rewrite_engine`):

```python
from cvzx.backends.nx.graph import to_diagram, to_graph

graph = to_graph(comp)
round_tripped = to_diagram(graph)
assert round_tripped == comp
```

You will rarely need to call `to_graph`/`to_diagram` directly — `cvzx.passes.optimize.optimize`
(see {doc}`optimization`) does the conversion for you — but `graph` is handy for inspecting a
diagram's graph structure directly, e.g. via its `GateRegister`:

```python
registry = graph.registry
```

`GateRegister` is what every `RewriteRule.match()` uses to find candidate nodes quickly
without re-scanning the whole graph — see {doc}`rewrite_rules`.

## Simplifying and visualizing

Putting it together — build a circuit, simplify it, and look at the result:

```python
from cvzx.passes.optimize import optimize
from cvzx.visualization.core import visualize

graph, diagram = optimize(comp)
fig = visualize(diagram, title="R(θ1)·R(θ2)·R(θ3)")
fig.savefig("rotation_chain.png")
```

`optimize()` returns a `(graph, diagram)` named tuple — see {doc}`optimization` for what
each one is for and why you get both.
