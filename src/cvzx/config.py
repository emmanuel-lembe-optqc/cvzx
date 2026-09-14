"""Backend selection for `CVZXGraph`-based diagram processing.

`cvzx` maintains two parallel graph backends (`nx_graph`/`nx_rewrite_rules`
built on `networkx`, `rx_graph`/`rx_rewrite_rules` built on `rustworkx`).
`Backend` names the choice; `DEFAULT_BACKEND` is what callers get when they
don't pick one explicitly.
"""

import importlib.util
from enum import StrEnum


class Backend(StrEnum):
    """Which graph library backs a `CVZXGraph`."""

    NETWORKX = "networkx"
    RUSTWORKX = "rustworkx"


DEFAULT_BACKEND: Backend = Backend.RUSTWORKX if importlib.util.find_spec("rustworkx") is not None else Backend.NETWORKX
