"""Small, diagram-type-agnostic geometry/predicate helpers for the visualizer.

Standalone functions used across several of the per-diagram-type drawing
mixins in `cvzx.visualization.core*` -- none of them need `self`.
"""

import random as r

import numpy as np

from cvzx.ir.base import Diagram, PSpider, QSpider

# A single wire/node anchor point in the drawing, as (x, y) figure coordinates.
type Position = tuple[float, float]


def require[T](value: T | None, message: str) -> T:
    """Narrow an Optional value to its non-None type, or raise.

    Used where an invariant elsewhere in this module guarantees a value is not
    None (e.g. two parameters are always provided together, or a loop always
    runs at least once), but that guarantee is not visible to the type checker
    from local control flow alone.

    Parameters
    ----------
    value: T | None
        The value expected to be non-None at this point.
    message: str
        Error message if the invariant does not hold.

    Returns
    -------
    T
        The narrowed, non-None value.

    Raises
    ------
    RuntimeError
        If value is None, meaning the invariant this call relies on was violated.
    """
    if value is None:
        raise RuntimeError(message)
    return value


def reorder_positions(connectivity: dict[int, int], positions: list[Position]) -> list[Position]:
    """Reorder input positions of a diagram inside a composition diagram according to the connectivity.

    Parameters
    ----------
    connectivity : dict[int, int]
        Dictionary indicating how the output position of a sub_diagram
        of a composition is connected to the the next one.
    positions : list[Position]
        List of positions of the sub_diagram which must be reordered
        according to the connectivity

    Returns
    -------
    list[Position]
        Reorder positions
    """
    # We must reverse the mapping because we want to link inputs
    # sub_diagram[i] and sub_diagram[i+1]
    connectivity_inv = {value: key for key, value in connectivity.items()}
    positions.reverse()
    res = [positions[connectivity_inv[i]] for i in range(len(positions))]
    res.reverse()
    return res


def normalize_radiuses(radiuses: list[float]) -> list[float]:
    """Normalize list of radiuses.

    The list of radiuses corresponds to sub-diagrams of a tensor diagram
    which is part of a contracted diagram. The normalization allows to
    avoid line superpositions when connecting outputs/inputs of D1 to
    inputs/outputs of D2 of the contracted diagram.

    Parameters
    ----------
    radiuses: list[float]
        List of input radiuses.

    Returns
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
    output = [float(v) for v in np.linspace(min_val, max_val, len(radiuses) + 1)]
    output.pop(0)
    return output


def is_wiring_diagram(diagram: Diagram) -> bool:
    """Check if a diagram is a wiring (identity) diagram.

    Returns
    -------
    bool
    """
    return (
        isinstance(diagram, (QSpider, PSpider))
        and not diagram.phase.coeffs
        and diagram.num_inputs == diagram.num_outputs
    )
