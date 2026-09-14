"""Manual/debug helper for comparing a diagram before and after a rewrite rule."""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from PIL import Image

from cvzx.visualization.core import visualize

if TYPE_CHECKING:
    from cvzx.ir.base import Diagram


def visualize_before_after(diagram_before: Diagram, diagram_after: Diagram, test_name: str, rule_name: str) -> None:
    """Visualize diagrams before and after applying a rewriting rule.

    Parameters
    ----------
    diagram_before : Diagram
        Diagram before applying the identity rule.
    diagram_after : Diagram
        Diagram after applying the identity rule.
    test_name : str
        Name of the test.
    rule_name : str
        Name of the rewriting rule used.
    """
    VIS_OUTPUT_DIR = Path(f"test_images_{rule_name}")  # ruff: ignore[non-lowercase-variable-in-function]
    VIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Create and render individual figures
    fig_before = visualize(diagram_before, title=f"Before: {test_name}")
    fig_after = visualize(diagram_after, title=f"After: {test_name}")

    # Convert to images
    buf_before = io.BytesIO()
    fig_before.savefig(buf_before, format="png", dpi=100, bbox_inches="tight")
    buf_before.seek(0)
    img_before = Image.open(buf_before)

    buf_after = io.BytesIO()
    fig_after.savefig(buf_after, format="png", dpi=100, bbox_inches="tight")
    buf_after.seek(0)
    img_after = Image.open(buf_after)

    plt.close(fig_before)
    plt.close(fig_after)

    # Create combined figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Display images
    ax1.imshow(img_before)
    ax1.axis("off")
    ax1.set_title(f"Before {rule_name}: {test_name}", fontsize=14)
    ax2.imshow(img_after)
    ax2.axis("off")
    ax2.set_title(f"After {rule_name}: {test_name}", fontsize=14)

    plt.tight_layout()
    filepath = VIS_OUTPUT_DIR / f"{test_name}.png"
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
