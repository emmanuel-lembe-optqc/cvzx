"""Runtime dispatch between the `networkx` and `rustworkx` `CVZXGraph` backends.

`nx_graph`/`nx_rewrite_rules` and `rx_graph`/`rx_rewrite_rules` are
maintained as parallel modules exposing the same names (`CVZXGraph`,
`to_graph`, `to_diagram`, and every `RewriteRule` subclass). This module
resolves a `Backend` selector to that pair of modules so a single call
site (e.g. `optimize()`) can stay backend-agnostic.
"""

from types import ModuleType

from cvzx.config import DEFAULT_BACKEND, Backend


def get_backend_modules(backend: Backend | str | None = None) -> tuple[Backend, ModuleType, ModuleType]:
    """Resolve a backend selector to its graph-conversion and rewrite-rule modules.

    Parameters
    ----------
    backend : Backend | str | None
        Which backend to use. `None` (the default) uses `DEFAULT_BACKEND`.

    Returns
    -------
    tuple[Backend, ModuleType, ModuleType]
        The resolved `Backend`, its `*_graph` module (`CVZXGraph`,
        `to_graph`, `to_diagram`), and its `*_rewrite_rules` module (the
        `RewriteRule` subclasses).

    Raises
    ------
    ValueError
        If `backend` is not a valid `Backend` value.
    """
    if backend is None:
        chosen = DEFAULT_BACKEND
    else:
        try:
            chosen = Backend(backend)
        except ValueError as exc:
            msg = f"Unknown backend {backend!r}; expected one of {[b.value for b in Backend]}."
            raise ValueError(msg) from exc

    if chosen == Backend.RUSTWORKX:
        import cvzx.rx_graph as graph_mod  # ruff: ignore[import-outside-top-level]
        import cvzx.rx_rewrite_rules as rules_mod  # ruff: ignore[import-outside-top-level]
    else:
        import cvzx.nx_graph as graph_mod  # type: ignore[no-redef]  # ruff: ignore[import-outside-top-level]
        import cvzx.nx_rewrite_rules as rules_mod  # type: ignore[no-redef]  # ruff: ignore[import-outside-top-level]

    return chosen, graph_mod, rules_mod
