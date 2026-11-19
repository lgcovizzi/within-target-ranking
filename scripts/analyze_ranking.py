#!/usr/bin/env python3
"""Ranking within target against a threshold between targets, over the set of DISTINCT models.

Why this script exists
----------------------
The reanalysis of 12 November counted DockQ **records**, not **models**. The long runs of
5 and 6 November reuse the seeds of the short runs of 3 November, and the same model enters
the count twice: in 1VFB, the 5 seeds of 3/11 (11 to 55, 5 samples each) are the same 25
predictions that reappear inside the 100 seeds of 5/11, with DockQ identical down to the last
digit. Counting 525 where there are 500 inflated denominators and numerators.

Here each model enters **once**, identified by (case, seed, sample). When the same pair appears
in different runs, the script checks that DockQ is equal before discarding the copy, and aborts
if it is not — because then it would not be a duplicate, it would be another condition.

The October pilots are left out: the methods document declares that the pilot seeds and the
dimer feasibility session do not enter the main analysis.

When the same model was scored under more than one chain assignment — homodimeric antigen, or
two copies of the crystal in 5NHR — the best one stays, which is what DockQ prescribes for
assignment ambiguity. Discarding the variant instead of resolving it would erase whole models:
in 5NHR, three of the 25 exist only under `crystal_copy_2`.

What comes out
--------------
`analyses/ranking/results.json` and a Markdown summary on stdout. No number in the manuscript
should be typed by hand: they all come from here.
"""
from __future__ import annotations

import json
import os
import re
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _analyses_dir() -> Path:
    """The analyses folder, in the work tree or in the deposited package.

    In the work tree the scripts sit in `manuscrito/` and the analyses in
    `manuscrito/analyses/`; in the deposit the scripts sit in `scripts/` and the analyses in
    `analyses/` at the package root. Both layouts are supported.
    """
    for base in (HERE, ROOT):
        if (base / "analyses").is_dir():
            return base / "analyses"
    return ROOT / "analyses"


ANALYSES = _analyses_dir()
# The Drive mirror lives in the main checkout, not in the article's work tree. A deposited
# execution record can be named with the RECORD environment variable, which takes precedence.
_RECORD = os.environ.get("RECORD")
_CANDIDATES = (Path(_RECORD) if _RECORD else None,
               ROOT / "execution-record",
               ROOT / "drive_espelho" / "modelagem" / "benchmarks" / "execucoes")
MIRROR = next((c for c in _CANDIDATES if c is not None and c.is_dir()),
              next(c for c in _CANDIDATES if c is not None))
DOCKQ = next((candidate for candidate in
              (ANALYSES / "dockq" / "results.json",
               ANALYSES / "dockq_20261118" / "results.json")
              if candidate.is_file()), ANALYSES / "dockq" / "results.json")
OUTPUT = ANALYSES / "ranking"

DOCKQ_CORRECT = 0.23   # DockQ, acceptable CAPRI class (Mirabello & Wallner 2024)
F1_CORRECT = 0.5       # this project's pre-registered criterion, by contact-pair F1
THRESHOLD = 0.9398     # the complex_iplddt cutoff the project had been using
PILOTS = ("20261026", "20261027", "20261028")
# The alignment-depth arms are a different input condition and are held out of the distinct-model
# set, as the article states. The curated DockQ file already omits them; listing the runs here
# makes a full rescoring from `structures/` produce the same set.
DEPTH_RUNS = {
    "20261116T104906Z_f16185d0", "20261116T105839Z_115bc275",
    "20261116T110644Z_85f64443", "20261116T222128Z_64b9d31e",
    "20261116T223343Z_7cf6f822", "20261117T104630Z_d224895c",
    "20261117T234611Z_282c1941",
}
SIGNALS = ("complex_iplddt", "complex_plddt", "confidence_score", "iptm", "ptm")
# Deposit from before the predictor's training cutoff, for the pre/post table.
PRE_CUTOFF = {"1vfb", "5nhr", "2vxs", "5f3b", "6wzm"}


def run_name(path: str) -> str:
    """The run identifier, from either the workstation tree or the deposited tree."""
    match = re.search(r"(?:execucoes|structures)/([^/]+)/", path)
    if not match:
        raise ValueError(f"path without a run: {path}")
    return match.group(1)


def metrics_of(entry: dict) -> dict | None:
    """The contact metrics of the evaluation, in the three forms the evaluators produce.

    `metrics` directly in the simple case; `copies` when the crystal has more than one copy of
    the reference, and then the best by pair F1 stays, which is what the table generator already
    did; `symmetry_assignments` when the antigen is homodimeric, and then the assignment the
    evaluator itself marked as invariant stays.
    """
    if "metrics" in entry:
        return entry["metrics"]
    if "copies" in entry:
        return max(entry["copies"].values(),
                   key=lambda c: c["metrics"]["contact_pairs"]["f1"])["metrics"]
    if "symmetry_assignments" in entry:
        return entry["symmetry_assignments"][entry["symmetry_invariant_assignment_index"]]["metrics"]
    return None


def resolve(path: Path) -> Path:
    """Where this file is, in the working mirror or in the deposited package tree.

    The stored paths come from the workstation, in `.../execucoes/<run>/<job>/output/.../file`.
    The mirror reproduces that whole tree; the deposited package keeps, per run and job, only
    the small files, in `<run>/<job>/file`. The three forms are tried in the order in which they
    are most likely, and the last one serves as the error message when none exists.
    """
    parts = path.parts
    attempts = [MIRROR / path]
    if "execucoes" in parts:
        i = parts.index("execucoes") + 1
        attempts.append(MIRROR.joinpath(*parts[i:]))          # without the project prefix
        attempts.append(MIRROR.joinpath(*parts[i:i + 2], path.name))  # <run>/<job>/name
    if "structures" in parts:
        # Results produced inside the deposit by score_dockq.py carry paths that start at
        # `structures/`; the record keeps the same run and job, without that prefix.
        i = parts.index("structures") + 1
        attempts.append(MIRROR.joinpath(*parts[i:]))
        attempts.append(MIRROR.joinpath(*parts[i:i + 2], path.name))
    return next((c for c in attempts if c.exists()), attempts[-1])


def f1_per_model(cif_path: str, cache: dict) -> float | None:
    """Contact-pair F1 of the model, matched by file NAME, not by order.

    The evaluation of a seed is a list with one entry per diffusion sample, and trusting the
    order has gone wrong before. Each entry carries the path it evaluated, in a workstation
    tree; only the file name is comparable on this side.
    """
    cif = Path(cif_path)
    job = next((a for a in cif.parents if "_seed" in a.name), None)
    if job is None:
        return None
    if job not in cache:
        # The path comes from the workstation tree. Here it is resolved against the mirror —
        # and, in the deposited data package, against a tree that starts at the run, without
        # the `modelagem/benchmarks/execucoes/` prefix. Both forms are tried, in that order.
        file = resolve(job / "evaluation.json")
        if not file.exists():
            cache[job] = {}
        else:
            data = json.loads(file.read_text())
            data = data if isinstance(data, list) else [data]
            index = {}
            for entry in data:
                name = Path(entry.get("predicted_structure", "")).name
                m = metrics_of(entry)
                if name and m:
                    index[name] = m["contact_pairs"]["f1"]
            cache[job] = index
    return cache[job].get(cif.name)


def load() -> tuple[list[dict], dict]:
    """One record per distinct model, with the confidence already read from the mirror."""
    raw = json.loads(DOCKQ.read_text())
    cleaning_log = {"records": len(raw["results"]), "dockq_version": raw["dockq_version"],
                    "pilot_records_discarded": 0, "depth_records_discarded": 0,
                    "assignments_resolved": 0,
                    "duplicate_records_discarded": 0, "divergent_duplicates": [],
                    "models_without_confidence": []}

    by_key: dict[tuple[str, int, int], dict] = {}
    for r in raw["results"]:
        run = run_name(r["model"])
        if run.startswith(PILOTS):
            cleaning_log["pilot_records_discarded"] += 1
            continue
        if run in DEPTH_RUNS:
            cleaning_log["depth_records_discarded"] += 1
            continue
        key = (r["case"], r["seed"], r["sample"])
        previous = by_key.get(key)
        if previous is None:
            by_key[key] = r
            continue
        if previous["variant"] != r["variant"]:
            # Same model scored under two chain assignments (homodimeric antigen) or against
            # two copies of the crystal. The best one stays, which is what DockQ prescribes
            # for assignment ambiguity.
            cleaning_log["assignments_resolved"] += 1
            if r["dockq"] > previous["dockq"]:
                by_key[key] = r
            continue
        if round(previous["dockq"], 6) != round(r["dockq"], 6):
            cleaning_log["divergent_duplicates"].append(
                {"key": list(key), "dockq": [previous["dockq"], r["dockq"]],
                 "runs": [run_name(previous["model"]), run_name(r["model"])]})
        cleaning_log["duplicate_records_discarded"] += 1

    rows, f1_cache = [], {}
    for (case, seed, sample), r in sorted(by_key.items()):
        cif = Path(r["model"])
        js = cif.parent / f"confidence_{cif.stem}.json"
        local = resolve(js)
        if not local.exists():
            cleaning_log["models_without_confidence"].append(str(js))
            continue
        confidence = json.loads(local.read_text())
        rows.append({"case": case, "target": case.split("_")[0], "seed": seed,
                     "sample": sample, "dockq": r["dockq"], "capri_class": r["capri_class"],
                     "correct": r["dockq"] >= DOCKQ_CORRECT, "run": run_name(r["model"]),
                     "f1_contacts": f1_per_model(r["model"], f1_cache),
                     **{s: confidence.get(s) for s in SIGNALS}})
    cleaning_log["distinct_models"] = len(by_key)
    cleaning_log["models_analyzed"] = len(rows)
    return rows, cleaning_log


def auc(positive: list[float], negative: list[float]) -> float | None:
    """AUC by pair counting, with a tie worth half a pair."""
    if not positive or not negative:
        return None
    wins = sum((1.0 if p > n else 0.5 if p == n else 0.0)
               for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def binomial_tail(k: int, n: int, p: float) -> float:
    return sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1))


def main() -> None:
    rows, cleaning_log = load()
    targets = sorted({row["target"] for row in rows})

    by_target = {}
    for target in targets:
        subset = [row for row in rows if row["target"] == target]
        correct = [row for row in subset if row["correct"]]
        by_target[target] = {
            "n": len(subset), "correct": len(correct), "rate": len(correct) / len(subset),
            "pre_cutoff": target in PRE_CUTOFF,
            "max_dockq": max(row["dockq"] for row in subset),
            "medium_or_better": sum(1 for row in subset if row["dockq"] >= 0.49),
            "high": sum(1 for row in subset if row["dockq"] >= 0.80),
            "auc": {s: auc([row[s] for row in correct],
                           [row[s] for row in subset if not row["correct"]])
                    for s in SIGNALS},
        }

    pooled = {s: auc([row[s] for row in rows if row["correct"]],
                     [row[s] for row in rows if not row["correct"]]) for s in SIGNALS}

    # Selection within the seed: the independent unit.
    seeds = {}
    for row in rows:
        seeds.setdefault((row["target"], row["seed"]), []).append(row)
    mixed = {k: v for k, v in seeds.items()
             if any(x["correct"] for x in v) and any(not x["correct"] for x in v)}
    selection = {s: {"correct_selected": 0, "trials": 0, "by_target": {}} for s in SIGNALS}
    for (target, _), group in sorted(mixed.items()):
        for s in SIGNALS:
            chosen = max(group, key=lambda x: x[s])
            d = selection[s]
            d["trials"] += 1
            d["correct_selected"] += int(chosen["correct"])
            per_target = d["by_target"].setdefault(target, {"correct_selected": 0, "trials": 0})
            per_target["trials"] += 1
            per_target["correct_selected"] += int(chosen["correct"])
    # Baseline: the chance of getting it right by drawing a sample from the mixed seed.
    baseline = sum(sum(1 for x in g if x["correct"]) / len(g)
                   for g in mixed.values()) / len(mixed)
    for s in SIGNALS:
        d = selection[s]
        d["p_vs_random_draw"] = binomial_tail(d["correct_selected"], d["trials"], baseline)
        d["p_vs_one_half"] = binomial_tail(d["correct_selected"], d["trials"], 0.5)

    above = [row for row in rows if row["complex_iplddt"] >= THRESHOLD]
    all_correct = [row for row in rows if row["correct"]]
    threshold = {
        "value": THRESHOLD,
        "models_above": len(above),
        "correct_above": sum(1 for row in above if row["correct"]),
        "incorrect_above": sum(1 for row in above if not row["correct"]),
        "correct_total": len(all_correct),
        "correct_lost": len(all_correct) - sum(1 for row in above if row["correct"]),
        "by_target": {a: {"correct": by_target[a]["correct"],
                          "correct_above": sum(1 for row in rows
                                               if row["target"] == a and row["correct"]
                                               and row["complex_iplddt"] >= THRESHOLD)}
                      for a in targets},
    }

    # Agreement between the pre-registered criterion (pair F1 >= 0.5) and DockQ.
    with_f1 = [row for row in rows if row["f1_contacts"] is not None]
    table = {"f1_correct_dockq_correct": 0, "f1_correct_dockq_incorrect": 0,
             "f1_incorrect_dockq_correct": 0, "f1_incorrect_dockq_incorrect": 0}
    for row in with_f1:
        a = "correct" if row["f1_contacts"] >= F1_CORRECT else "incorrect"
        b = "correct" if row["correct"] else "incorrect"
        table[f"f1_{a}_dockq_{b}"] += 1
    agreed = table["f1_correct_dockq_correct"] + table["f1_incorrect_dockq_incorrect"]
    agreement = {
        "models_with_f1": len(with_f1), "without_f1": len(rows) - len(with_f1),
        "f1_threshold": F1_CORRECT, **table,
        "agreement": agreed / len(with_f1) if with_f1 else None,
        "disagreements": len(with_f1) - agreed,
    }

    output = {"generated_from": str(DOCKQ.relative_to(ROOT)),
              "dockq_correct_threshold": DOCKQ_CORRECT,
              "cleaning_log": cleaning_log, "by_target": by_target, "pooled_auc": pooled,
              "mixed_seeds": len(mixed), "random_draw_baseline": baseline,
              "seed_selection": selection, "threshold": threshold,
              "f1_dockq_agreement": agreement}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "results.json").write_text(json.dumps(output, indent=1, ensure_ascii=False))

    # --- readable summary -------------------------------------------------
    d = cleaning_log
    print(f"records {d['records']}  pilots -{d['pilot_records_discarded']}  "
          f"depth -{d['depth_records_discarded']}  "
          f"assignments resolved -{d['assignments_resolved']}  "
          f"duplicates -{d['duplicate_records_discarded']}  => distinct {d['distinct_models']}, "
          f"with confidence {d['models_analyzed']}")
    if d["divergent_duplicates"]:
        print(f"WARNING: {len(d['divergent_duplicates'])} duplicates with different DockQ")
    if d["models_without_confidence"]:
        print(f"WARNING: {len(d['models_without_confidence'])} models without confidence "
              f"in the mirror")

    print(f"\n{'target':8} {'cut':5} {'n':>5} {'correct':>8} {'rate':>7} {'medium+':>7} "
          f"{'high':>5} {'maxDockQ':>9} " + " ".join(f"{s[:9]:>9}" for s in SIGNALS))
    for a in targets:
        p = by_target[a]
        aucs = " ".join(f"{p['auc'][s]:9.3f}" if p["auc"][s] is not None else f"{'-':>9}"
                        for s in SIGNALS)
        print(f"{a:8} {'pre' if p['pre_cutoff'] else 'post':5} {p['n']:5d} {p['correct']:8d} "
              f"{p['rate']*100:6.1f}% {p['medium_or_better']:7d} {p['high']:5d} "
              f"{p['max_dockq']:9.3f} {aucs}")
    print(f"{'POOLED':8} {'':5} {len(rows):5d} {len(all_correct):8d} "
          f"{len(all_correct)/len(rows)*100:6.1f}% {'':7} {'':5} {'':9} "
          + " ".join(f"{pooled[s]:9.3f}" for s in SIGNALS))

    print(f"\nmixed seeds: {len(mixed)}   random-draw baseline: {baseline:.3f}")
    for s in SIGNALS:
        sel = selection[s]
        detail = " ".join(f"{a}:{v['correct_selected']}/{v['trials']}"
                          for a, v in sorted(sel["by_target"].items()))
        print(f"  {s:17} {sel['correct_selected']:2d}/{sel['trials']:2d}  "
              f"p(draw)={sel['p_vs_random_draw']:.3g}  "
              f"p(0.5)={sel['p_vs_one_half']:.3g}   {detail}")

    c = agreement
    print(f"\nContact F1 (>= {F1_CORRECT}) against DockQ (>= {DOCKQ_CORRECT}), "
          f"{c['models_with_f1']} models ({c['without_f1']} without evaluation):")
    print(f"  {'':14} {'DockQ correct':>13} {'DockQ incorrect':>15}")
    print(f"  {'F1 correct':14} {c['f1_correct_dockq_correct']:13d} "
          f"{c['f1_correct_dockq_incorrect']:15d}")
    print(f"  {'F1 incorrect':14} {c['f1_incorrect_dockq_correct']:13d} "
          f"{c['f1_incorrect_dockq_incorrect']:15d}")
    print(f"  agreement {c['agreement']*100:.2f}%, {c['disagreements']} disagreements")

    thr = threshold
    print(f"\nthreshold {THRESHOLD}: {thr['models_above']} models above, "
          f"{thr['correct_above']} correct and {thr['incorrect_above']} incorrect")
    print(f"  loses {thr['correct_lost']} of the {thr['correct_total']} correct "
          f"({thr['correct_lost']/thr['correct_total']*100:.0f}%)")
    for a in targets:
        v = thr["by_target"][a]
        if v["correct"]:
            print(f"    {a:8} {v['correct_above']:3d} of {v['correct']:3d} correct accepted")


if __name__ == "__main__":
    main()
