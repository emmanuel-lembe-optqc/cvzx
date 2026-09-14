"""Run the optimization-quality benchmark suite and print a report.

Usage
-----
    python -m benchmarks.run_benchmarks [--format text|json]

For each fixture in `benchmarks.fixtures.FIXTURES`, runs `optimize()` under
both `assume_infinite_squeezing` settings and reports `cvzx.utils.metrics`
before/after/ratio/reduction for every `DiagramMetrics` field, plus an
arithmetic-mean-reduction summary per setting across all fixtures.

Dev-invoked only -- not wired into CI (see the dev guide, "Benchmarking
optimization quality", for why). Metrics are always computed from
`OptimizeResult.diagram`, never `.graph` (see that class's docstring).
"""

# ruff: file-ignore[print] -- printing the report is this script's entire job.

from __future__ import annotations

import argparse
import dataclasses
import json
import sys

from benchmarks.fixtures import FIXTURES
from cvzx.passes.optimize import optimize
from cvzx.utils.metrics import DiagramMetrics, MetricsComparison, compare_metrics

_METRIC_FIELDS = [f.name for f in dataclasses.fields(DiagramMetrics)]
_ASSUME_INFINITE_SQUEEZING_SETTINGS = (False, True)


def _fmt(value: float | None) -> str:
    """Format a metric value or ratio/reduction for the text table.

    Returns
    -------
    str
    """
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _print_text_table(fixture_name: str, *, assume_infinite_squeezing: bool, comparison: MetricsComparison) -> None:
    """Print one fixture/setting's before/after/ratio/reduction table."""
    print(f"\n=== {fixture_name} (assume_infinite_squeezing={assume_infinite_squeezing}) ===")
    header = f"{'metric':<22}{'before':>8}{'after':>8}{'ratio':>8}{'reduction':>11}"
    print(header)
    for field in _METRIC_FIELDS:
        before = getattr(comparison.before, field)
        after = getattr(comparison.after, field)
        ratio = comparison.ratio(field)
        reduction = comparison.reduction(field)
        print(f"{field:<22}{_fmt(before):>8}{_fmt(after):>8}{_fmt(ratio):>8}{_fmt(reduction):>11}")


def _print_summary(reductions_by_setting: dict[bool, dict[str, list[float]]]) -> None:
    """Print the arithmetic-mean-reduction summary, one row per (setting, metric)."""
    print("\n=== Summary: arithmetic mean reduction across fixtures ===")
    header = f"{'assume_infinite_squeezing':<28}{'metric':<22}{'mean_reduction':>14}"
    print(header)
    for assume_infinite_squeezing in _ASSUME_INFINITE_SQUEEZING_SETTINGS:
        for field in _METRIC_FIELDS:
            values = reductions_by_setting[assume_infinite_squeezing][field]
            mean = sum(values) / len(values) if values else None
            print(f"{assume_infinite_squeezing!s:<28}{field:<22}{_fmt(mean):>14}")


def run(*, output_format: str) -> None:
    """Run every fixture under both settings and print the report.

    Parameters
    ----------
    output_format : str
        `"text"` for the human-readable tables, `"json"` for a machine-
        readable list of per-fixture/setting records.
    """
    records: list[dict] = []
    reductions_by_setting: dict[bool, dict[str, list[float]]] = {
        setting: {field: [] for field in _METRIC_FIELDS} for setting in _ASSUME_INFINITE_SQUEEZING_SETTINGS
    }

    for fixture_name, factory in FIXTURES.items():
        for assume_infinite_squeezing in _ASSUME_INFINITE_SQUEEZING_SETTINGS:
            diagram = factory()
            result = optimize(diagram, assume_infinite_squeezing=assume_infinite_squeezing)
            comparison = compare_metrics(diagram, result.diagram)

            if output_format == "text":
                _print_text_table(
                    fixture_name, assume_infinite_squeezing=assume_infinite_squeezing, comparison=comparison
                )
            else:
                records.append({
                    "fixture": fixture_name,
                    "assume_infinite_squeezing": assume_infinite_squeezing,
                    "before": dataclasses.asdict(comparison.before),
                    "after": dataclasses.asdict(comparison.after),
                    "reduction": {field: comparison.reduction(field) for field in _METRIC_FIELDS},
                })

            for field in _METRIC_FIELDS:
                reduction = comparison.reduction(field)
                if reduction is not None:
                    reductions_by_setting[assume_infinite_squeezing][field].append(reduction)

    if output_format == "text":
        _print_summary(reductions_by_setting)
    else:
        summary = {
            str(assume_infinite_squeezing): {
                field: (sum(values) / len(values) if values else None)
                for field, values in reductions_by_setting[assume_infinite_squeezing].items()
            }
            for assume_infinite_squeezing in _ASSUME_INFINITE_SQUEEZING_SETTINGS
        }
        json.dump({"fixtures": records, "summary": summary}, sys.stdout, indent=2)
        sys.stdout.write("\n")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format (default: text).")
    args = parser.parse_args()
    run(output_format=args.format)


if __name__ == "__main__":
    main()
