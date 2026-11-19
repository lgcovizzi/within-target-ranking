#!/usr/bin/env python3
"""Seed-clustered bootstrap intervals for the within-target AUCs.

The five diffusion samples of one seed are not independent observations, so an interval that
resamples models would be too narrow. This script resamples **seeds** with replacement, keeps
every model of the sampled seeds, and recomputes the AUC by pair counting. It reads the same
distinct-model set as `analyze_ranking.py`.

    RECORD=execution-record python scripts/bootstrap_auc.py

It prints the 2.5th and 97.5th percentiles of 1,000 resamples for each signal and each target
in which both classes occur.
"""

from __future__ import annotations

import random

import analyze_ranking as ranking

RESAMPLES = 1_000
SEED = 20261118
TARGETS = ("1vfb", "9sg0", "9gvn")


def main() -> int:
    rows, _ = ranking.load()
    by_target: dict[str, dict[int, list[dict]]] = {}
    for row in rows:
        by_target.setdefault(row["target"], {}).setdefault(row["seed"], []).append(row)
    random.seed(SEED)
    for target in TARGETS:
        seeds = list(by_target[target])
        positives = sum(1 for row in rows if row["target"] == target and row["correct"])
        print(f"\n{target}: {len(seeds)} seeds, {positives} correct poses")
        for signal in ranking.SIGNALS:
            values = []
            for _ in range(RESAMPLES):
                sample = [random.choice(seeds) for _ in seeds]
                pos, neg = [], []
                for seed in sample:
                    for row in by_target[target][seed]:
                        (pos if row["correct"] else neg).append(row[signal])
                value = ranking.auc(pos, neg)
                if value is not None:
                    values.append(value)
            values.sort()
            low = values[int(0.025 * len(values))]
            high = values[int(0.975 * len(values))]
            print(f"  {signal:17} {low:.3f}--{high:.3f}  ({len(values)} resamples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
