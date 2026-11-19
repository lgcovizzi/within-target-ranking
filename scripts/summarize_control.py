#!/usr/bin/env python3
"""Summarize the negative control: confidence on a pair that does not bind.

The run is the CJM112 Fv against the IL-17F homodimer, 100 seeds and five diffusion samples.
It has no experimental complex and no DockQ score; the question is how often the predictor is
confident about an interface that should not exist. This script reproduces the table in the
article and the one-sided Clopper--Pearson bound.

    RECORD=execution-record python scripts/summarize_control.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import os

RUN = "20261112T103735Z_aa0ecb94"
THRESHOLD = 0.9398
SIGNALS = ("complex_iplddt", "complex_plddt", "iptm", "ptm", "confidence_score")


def main() -> int:
    record = Path(os.environ.get("RECORD", "execution-record"))
    run = record / RUN
    if not run.is_dir():
        raise SystemExit(f"run not found: {run}")
    values: dict[str, list[float]] = {signal: [] for signal in SIGNALS}
    for path in sorted(run.rglob("confidence_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for signal in SIGNALS:
            value = data.get(signal)
            if value is not None:
                values[signal].append(float(value))

    n = len(values["complex_iplddt"])
    print(f"\nnegative control {RUN}: {n} models")
    print(f"{'measure':17} {'minimum':>8} {'median':>8} {'maximum':>8}")
    for signal in SIGNALS:
        series = values[signal]
        print(f"{signal:17} {min(series):8.4f} {statistics.median(series):8.4f} "
              f"{max(series):8.4f}")

    above = sum(1 for value in values["complex_iplddt"] if value > THRESHOLD)
    upper = 1 - 0.05 ** (1 / n) if n else float("nan")
    print(f"\n{above} of {n} models above {THRESHOLD}; "
          f"one-sided 95% Clopper--Pearson upper bound {100 * upper:.2f}%")
    print("This is the independent control. It is not pooled with the development-set errors: "
          "the threshold was chosen from those errors, so their zero above it is in-sample.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
