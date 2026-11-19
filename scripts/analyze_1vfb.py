#!/usr/bin/env python3
"""E1 analysis: the 1VFB control with one hundred seeds and five diffusion samples per seed.

Same rule as the tables and figures: no number is copied by hand. The script reads the run from
the Drive mirror and writes manuscrito/analyses/1vfb/summary.json, the confidence-versus-
correctness figure and PROVENANCE.json with the SHA-256 of every file read.

    python manuscrito/analyze_1vfb.py
    RECORD=/path/to/deposit/execution-record python manuscrito/analyze_1vfb.py
    python manuscrito/analyze_1vfb.py --run 20261105T115759Z_e0e7ca12

Unlike make_article_tables.py, this script does not require run_plan.json: the run of 5 November
reached the Drive without it, with 92 of the 100 seeds. The expected seeds are those of the
command recorded in the E1 plan (11, 22, ..., 1100); the missing ones are listed in the summary,
and no conclusion is marked final while any is missing.

The formal test, with permutation and Holm correction, remains the one in
modelagem/benchmarks/testar_discriminadores.py. This script describes: success rate, where the
correct models sit in each confidence signal, AUC, reproducibility of the repeated seeds and the
scale of complex_plddt in the other controls and in B48C. The complex_plddt thresholds appear as
a description of what was observed, not as a criterion: they were looked at after the data, and
a criterion only counts if it is fixed before a new run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RELATIVE_SCRIPT = Path(__file__).resolve().relative_to(ROOT).as_posix()

os.environ.setdefault("SOURCE_DATE_EPOCH", "1793836800")  # reproducible PDF

import matplotlib  # noqa: E402

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

F1_CORRECT = 0.5  # the project's pre-registered criterion: contact-pair F1 at or above this


def f1_contact_pairs(evaluation: dict) -> float | None:
    """Contact-pair F1 of a model evaluation, in the three forms the evaluators produce.

    `metrics` directly in the simple case; `copies` when the crystal has more than one copy of
    the reference, and then the best by contact-pair F1 is kept; `symmetry_assignments` when the
    antigen is homodimeric, and then the assignment the evaluator itself marked as invariant.

    This is a copy of the project's helper, kept here so that the deposited code runs on its own.
    """
    metrics = evaluation.get("metrics")
    if metrics and "contact_pairs" in metrics:
        return metrics["contact_pairs"]["f1"]
    if "copies" in evaluation:
        return max(c["metrics"]["contact_pairs"]["f1"] for c in evaluation["copies"].values())
    if "symmetry_assignments" in evaluation:
        index = evaluation.get("symmetry_invariant_assignment_index")
        if index is None:
            return max(a["metrics"]["contact_pairs"]["f1"]
                       for a in evaluation["symmetry_assignments"])
        return evaluation["symmetry_assignments"][index]["metrics"]["contact_pairs"]["f1"]
    return None

# The mirror lives in the main checkout, not in the article's work tree. The deposited execution
# record can be named with the RECORD environment variable, which takes precedence.
_RECORD = os.environ.get("RECORD")
_CANDIDATES = (Path(_RECORD) if _RECORD else None,
               ROOT / "execution-record",
               ROOT / "drive_espelho")
MIRROR = next((c for c in _CANDIDATES if c is not None and c.is_dir()),
              next(c for c in _CANDIDATES if c is not None))
if (MIRROR / "execution-record").is_dir():
    MIRROR = MIRROR / "execution-record"   # the deposit root, not the record itself

OUTPUT = ROOT / "analyses" / "1vfb"
CASE = "1vfb_fv_lysozyme"
RUN_E1 = "20261105T115759Z_e0e7ca12"
EXPECTED_SEEDS = list(range(11, 1101, 11))  # sequence 11 11 1100, from the E1 command
REPEATED_SEEDS = (11, 22, 33, 44, 55)  # already run on 3 November, same options
PREVIOUS_1VFB_RUN = "20261103T114054Z_647515fd"
SCALE_RUNS = {
    "1VFB (3/11)": ("benchmarks/execucoes/20261103T114054Z_647515fd", "1vfb_fv_lysozyme"),
    "5NHR": ("benchmarks/execucoes/20261103T114745Z_8d23fb8c", "5nhr_fv_acvr2b"),
    "9SG0": ("benchmarks/execucoes/20261103T170542Z_f5c44104", "9sg0_fv_il17a_dimer"),
    "CJM112–IL-17F (negative)": ("benchmarks/execucoes/20261103T171722Z_f7ddbca4",
                                 "cjm112_fv_il17f_negative"),
    "B48C, Fv + murine monomer": ("execucoes/20261103T115422Z_05745f95", "mouse_fv_monomer"),
    "B48C, Fv + murine dimer": ("execucoes/20261103T115422Z_05745f95", "mouse_fv_dimer"),
    "B48C, Fv + human monomer": ("execucoes/20261103T115422Z_05745f95", "human_fv_monomer"),
    "B48C, Fv + human dimer": ("execucoes/20261103T115422Z_05745f95", "human_fv_dimer"),
}
SIGNALS = ("iptm", "ptm", "confidence_score", "complex_plddt", "complex_iplddt",
           "complex_pde", "complex_ipde")
DESCRIPTIVE_THRESHOLDS = (0.93, 0.935, 0.94, 0.945)
PERMUTATIONS = 10_000  # same number as testar_discriminadores.py
PERMUTATION_SEED = 20261104  # same seed, so the report is reproducible

_read: dict[str, str] = {}
_curated: dict[str, Path] | None = None


def sha256(path: Path) -> str:
    """SHA-256 of a file, in blocks, as the deposit manifest computes it."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _curated_runs() -> dict[str, Path]:
    """Run id -> `execucao` folder of the curated record, for runs kept only there.

    The deposited record is not a single mirror tree: the benchmark runs sit directly under
    `execution-record/<run>/`, while the B48C session is kept only under
    `curation/b48c/<session>/execucao/`. The curation provenance names the original run, so the
    folder can be found without hard-coding the session.
    """
    global _curated
    if _curated is not None:
        return _curated
    _curated = {}
    for root in (MIRROR / "curation", MIRROR.parent / "curation"):
        if not root.is_dir():
            continue
        for provenance in sorted(root.rglob("PROVENANCE.json")):
            try:
                origin = json.loads(provenance.read_text(encoding="utf-8"))["source"]["run"]
            except (KeyError, ValueError):
                continue
            execution = provenance.parent / "execution"
            if execution.is_dir():
                _curated.setdefault(origin, execution)
    return _curated


def _attempts(path: Path) -> list[Path]:
    """Places where a logical path can physically be, most likely first.

    The script builds every path under `MIRROR/modelagem`, which is how the Drive mirror stores
    it and how the provenance keys name it. The deposited execution record drops the
    `modelagem/` prefix, flattens the confidence JSONs into the job folder, and keeps B48C under
    `curation/`; each form is tried in turn.
    """
    try:
        parts = path.relative_to(MIRROR).parts
    except ValueError:
        return [path]
    attempts = [path]
    if not parts or parts[0] != "modelagem":
        return attempts
    rest = parts[1:]
    attempts.append(MIRROR.joinpath(*rest))
    if rest[:2] == ("benchmarks", "execucoes"):
        offset = 2
    elif rest[:1] == ("execucoes",):
        offset = 1
    else:
        return attempts
    if len(rest) <= offset:
        return attempts
    run = rest[offset]
    curated = _curated_runs().get(run)
    if len(rest) == offset + 1:            # the run folder itself
        attempts.append(MIRROR / run)
        if curated is not None:
            attempts.append(curated)
    elif len(rest) == offset + 2:          # one job folder
        attempts.append(MIRROR.joinpath(*rest[offset:offset + 2]))
        if curated is not None:
            attempts.append(curated / rest[offset + 1])
    else:                                  # a file inside the job
        if curated is not None:
            attempts.append(curated.joinpath(*rest[offset + 1:]))
        attempts.append(MIRROR.joinpath(*rest[offset:offset + 2], path.name))
    return attempts


def resolve_file(path: Path) -> Path:
    """Physical file behind a logical path, or the logical path itself when it is absent."""
    attempts = _attempts(path)
    return next((c for c in attempts if c.is_file()), attempts[0])


def resolve_dir(path: Path) -> Path:
    """Physical folder behind a logical path, or the logical path itself when it is absent."""
    attempts = _attempts(path)
    return next((c for c in attempts if c.is_dir()), attempts[0])


def read_json(path: Path):
    _read[path.relative_to(MIRROR).as_posix()] = sha256(resolve_file(path))
    return json.loads(resolve_file(path).read_text(encoding="utf-8"))


def confidences(job: Path, case: str) -> dict[int, dict]:
    base = job / "output" / f"boltz_results_{case}" / "predictions" / case
    folder = resolve_dir(base) if resolve_dir(base).is_dir() else resolve_dir(job)
    output = {}
    for file in sorted(folder.glob(f"confidence_{case}_model_*.json")):
        model = int(re.search(r"_model_(\d+)\.json$", file.name).group(1))
        output[model] = read_json(base / file.name)
    return output


def auc(correct: np.ndarray, incorrect: np.ndarray) -> float | None:
    """P(signal of a correct model > signal of an incorrect one), ties counting half."""
    if not len(correct) or not len(incorrect):
        return None
    greater = (correct[:, None] > incorrect[None, :]).sum()
    tie = (correct[:, None] == incorrect[None, :]).sum()
    return float((greater + 0.5 * tie) / (len(correct) * len(incorrect)))


def collect_e1(run: Path) -> tuple[list[dict], list[int], list[int]]:
    rows, present, incomplete = [], [], []
    for seed in EXPECTED_SEEDS:
        job = run / f"{CASE}_seed{seed}"
        evaluation = job / "evaluation.json"
        if not resolve_file(evaluation).is_file():
            continue
        present.append(seed)
        if not resolve_file(job / "status.json").is_file():
            incomplete.append(seed)
        confidence = confidences(job, CASE)
        for model, ev in enumerate(read_json(evaluation)):
            pose = ev["metrics"]["pose"]
            row = {"seed": seed, "model": model, "f1_contacts": f1_contact_pairs(ev),
                   "antigen_rotation_degrees": pose["antigen_residual_rotation_deg"],
                   "antigen_rmsd_angstrom": pose["antigen_rmsd_in_fv_frame_A"],
                   "rmsd_fv_angstrom": pose["fv_rmsd_A"],
                   "structure_sha256": ev["predicted_structure_sha256"]}
            row.update({s: confidence[model][s] for s in SIGNALS})
            row["correct"] = row["f1_contacts"] >= F1_CORRECT
            rows.append(row)
    return rows, present, incomplete


def describe_signals(rows: list[dict]) -> list[dict]:
    labels = np.array([l["correct"] for l in rows])
    output = []
    for signal in SIGNALS:
        values = np.array([l[signal] for l in rows])
        order = (-values).argsort(kind="stable")
        position = np.empty(len(values), dtype=int)
        position[order] = np.arange(1, len(values) + 1)
        top = {str(n): int(labels[order[:n]].sum()) for n in (1, 5, 10, 25, 50)}
        output.append({
            "signal": signal,
            "auc_higher_means_correct": auc(values[labels], values[~labels]),
            "correct_positions_highest_first": sorted(int(p) for p in position[labels]),
            "correct_among_top_n": top,
            "median_correct": float(np.median(values[labels])) if labels.any() else None,
            "median_incorrect": float(np.median(values[~labels])),
        })
    return output


def seed_level_test(rows: list[dict]) -> dict:
    """Separation measured once per seed, where the observations are independent.

    The five diffusion samples of a seed are not independent observations, and the permutation
    test over the 460 models assumes that they are. Here each seed enters once, represented by
    its highest value of the signal, and the label is whether that seed produced any correct
    model. The permutation test then runs over the 92 seeds.

    The separation using only sample 0 of each seed is also reported; it is independent by
    construction and does not depend on choosing the best sample.
    """
    groups: dict[int, list[dict]] = {}
    for l in rows:
        groups.setdefault(l["seed"], []).append(l)
    rng = np.random.default_rng(PERMUTATION_SEED)
    output = {"seeds": len(groups), "signals": []}
    for signal in SIGNALS:
        values = np.array([max(g, key=lambda l: l[signal])[signal] for g in groups.values()])
        labels = np.array([any(l["correct"] for l in g) for g in groups.values()])
        if not labels.any() or labels.all():
            continue
        observed = auc(values[labels], values[~labels])
        deviation = abs(observed - 0.5)
        k = int(labels.sum())
        indices = np.arange(len(values))
        extremes = 0
        for _ in range(PERMUTATIONS):
            mask = np.zeros(len(values), dtype=bool)
            mask[rng.choice(indices, size=k, replace=False)] = True
            if abs(auc(values[mask], values[~mask]) - 0.5) >= deviation:
                extremes += 1
        sample0 = [g[0] for g in groups.values() if g]
        a0 = np.array([l[signal] for l in sample0 if l["correct"]])
        i0 = np.array([l[signal] for l in sample0 if not l["correct"]])
        output["signals"].append({
            "signal": signal,
            "seeds_with_correct": k,
            "auc_per_seed": observed,
            "p_permutation_per_seed": (extremes + 1) / (PERMUTATIONS + 1),
            "auc_sample_0_only": auc(a0, i0),
            "correct_in_sample_0": int(len(a0)),
        })
    output["signals"].sort(key=lambda r: -abs(r["auc_per_seed"] - 0.5))
    output["note"] = (
        "The smallest achievable p with 10 000 permutations is 1/10 001. Holm correction over "
        "the seven signals keeps any p at that floor significant.")
    return output


def by_seed(rows: list[dict]) -> dict:
    groups: dict[int, list[dict]] = {}
    for l in rows:
        groups.setdefault(l["seed"], []).append(l)
    count: dict[str, int] = {}
    for g in groups.values():
        k = str(sum(l["correct"] for l in g))
        count[k] = count.get(k, 0) + 1
    rate = float(np.mean([l["correct"] for l in rows]))
    expected = len(groups) * (1 - (1 - rate) ** 5 - 5 * rate * (1 - rate) ** 4)
    with_correct = [g for g in groups.values() if any(l["correct"] for l in g)]

    def best(g, signal):
        return max(g, key=lambda l: l[signal])["correct"]

    return {
        "seeds_by_number_correct": dict(sorted(count.items())),
        "seeds_with_two_or_more_expected_if_samples_independent": expected,
        "sample_0_correct": sum(g[0]["correct"] for g in groups.values()),
        "seed_highest_iptm_correct": sum(best(g, "iptm") for g in groups.values()),
        "seed_highest_complex_plddt_correct": sum(best(g, "complex_plddt") for g in groups.values()),
        "seeds_with_correct": len(with_correct),
        "among_them_highest_complex_plddt_correct": sum(best(g, "complex_plddt") for g in with_correct),
        "among_them_highest_iptm_correct": sum(best(g, "iptm") for g in with_correct),
    }


def reproducibility(rows: list[dict]) -> dict:
    previous = MIRROR / "modelagem" / "benchmarks" / "execucoes" / PREVIOUS_1VFB_RUN
    compared, identical = 0, 0
    for seed in REPEATED_SEEDS:
        file = previous / f"{CASE}_seed{seed}" / "evaluation.json"
        if not resolve_file(file).is_file():
            continue
        before = [ev["predicted_structure_sha256"] for ev in read_json(file)]
        now = [l["structure_sha256"] for l in rows if l["seed"] == seed]
        for a, b in zip(before, now):
            compared += 1
            identical += a == b
    return {"previous_run": PREVIOUS_1VFB_RUN, "seeds": list(REPEATED_SEEDS),
            "structures_compared": compared, "identical_structures_sha256": identical}


def plddt_scale() -> list[dict]:
    """Range of the local confidence in each system, for the transfer condition.

    The pre-registered criterion requires a signal to reject the predictions of a control where
    the method is wrong. Here the range of both local-confidence signals is measured in every
    system already run, to compare with the range of the correct 1VFB models.
    """
    output = []
    for label, (folder, case) in SCALE_RUNS.items():
        run = MIRROR / "modelagem" / folder
        values: dict[str, list[float]] = {"complex_plddt": [], "complex_iplddt": []}
        correct = []
        physical = resolve_dir(run)
        for job_name in sorted(p.name for p in physical.glob(f"{case}_seed*")):
            job = run / job_name
            evaluation = job / "evaluation.json"
            evaluations = read_json(evaluation) if resolve_file(evaluation).is_file() else None
            for model, c in sorted(confidences(job, case).items()):
                for signal in values:
                    values[signal].append(c[signal])
                f1 = f1_contact_pairs(evaluations[model]) if evaluations else None
                correct.append(None if f1 is None else f1 >= F1_CORRECT)
        if not correct:
            continue
        known = [a for a in correct if a is not None]
        row = {"system": label, "run": run.name, "models": len(correct),
               "correct": sum(known) if known else None}
        for signal, v in values.items():
            row[f"{signal}_min"] = min(v)
            row[f"{signal}_median"] = float(np.median(v))
            row[f"{signal}_max"] = max(v)
        output.append(row)
    return output


def figure(rows: list[dict], destination: Path) -> None:
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), constrained_layout=True)
    correct = np.array([l["correct"] for l in rows])
    f1 = np.array([l["f1_contacts"] for l in rows])
    for axis, signal, label in ((axes[0], "iptm", "ipTM"), (axes[1], "complex_plddt", "complex pLDDT")):
        x = np.array([l[signal] for l in rows])
        axis.scatter(x[~correct], f1[~correct], s=7, marker="o", facecolors="none",
                     edgecolors="0.45", linewidths=0.5, label=f"incorrect (n = {int((~correct).sum())})")
        axis.scatter(x[correct], f1[correct], s=16, marker="^", color="black",
                     label=f"correct (n = {int(correct.sum())})")
        axis.axhline(F1_CORRECT, color="0.6", linestyle="--", linewidth=0.6)
        axis.set_xlabel(label)
        axis.set_ylabel("F1 of contact pairs")
        axis.set_ylim(-0.03, 1.0)
    axes[0].legend(frameon=False, loc="center right")
    fig.savefig(destination, metadata={"CreationDate": None})
    plt.close(fig)


def summarize(run_id: str) -> dict:
    run = MIRROR / "modelagem" / "benchmarks" / "execucoes" / run_id
    rows, present, incomplete = collect_e1(run)
    missing = [s for s in EXPECTED_SEEDS if s not in present]
    run_files = [n for n in ("run_plan.json", "summary.json") if resolve_file(run / n).is_file()]
    complete = not missing and not incomplete and len(run_files) == 2
    correct = [l for l in rows if l["correct"]]
    new = [l for l in rows if l["seed"] not in REPEATED_SEEDS]
    correct_fields = ("seed", "model", "f1_contacts", "antigen_rotation_degrees",
                      "antigen_rmsd_angstrom", "iptm", "complex_plddt")
    return {
        "run": run_id,
        "complete": complete,
        "warning": None if complete else (
            "Incomplete run on the Drive: the numbers are preliminary and change once the "
            "missing seeds arrive."),
        "expected_seeds": len(EXPECTED_SEEDS),
        "present_seeds": len(present),
        "missing_seeds": missing,
        "seeds_without_status_json": incomplete,
        "run_files_present": run_files,
        "models": len(rows),
        "correct": len(correct),
        "success_rate": len(correct) / len(rows),
        "f1_correct_threshold": F1_CORRECT,
        "f1_bands": {
            "zero": sum(l["f1_contacts"] == 0 for l in rows),
            "up_to_0_15": sum(0 < l["f1_contacts"] <= 0.15 for l in rows),
            "between_0_15_and_0_5": sum(0.15 < l["f1_contacts"] < F1_CORRECT for l in rows),
            "correct": len(correct),
        },
        "correct_detail": sorted(({k: l[k] for k in correct_fields} for l in correct),
                                 key=lambda d: -d["complex_plddt"]),
        "rmsd_fv_angstrom": {"min": min(l["rmsd_fv_angstrom"] for l in rows),
                             "max": max(l["rmsd_fv_angstrom"] for l in rows)},
        "signals": describe_signals(rows),
        "signals_new_seeds_only": {
            "note": "excludes seeds 11 to 55, already run on 3 November, when complex_plddt "
                    "was first noticed",
            "models": len(new), "correct": sum(l["correct"] for l in new),
            "auc": {s: auc(np.array([l[s] for l in new if l["correct"]]),
                           np.array([l[s] for l in new if not l["correct"]])) for s in SIGNALS},
        },
        "complex_plddt_descriptive_thresholds": {
            "note": "looked at after the data; description, not criterion",
            "thresholds": [{"threshold": t,
                            "models_above": sum(l["complex_plddt"] >= t for l in rows),
                            "correct_above": sum(l["correct"] and l["complex_plddt"] >= t for l in rows)}
                           for t in DESCRIPTIVE_THRESHOLDS]},
        "separation_without_overlap": [
            {"signal": s,
             "lowest_correct_value": min(l[s] for l in rows if l["correct"]),
             "highest_incorrect_value": max(l[s] for l in rows if not l["correct"]),
             "incorrect_above_lowest_correct": sum(
                 l[s] >= min(x[s] for x in rows if x["correct"]) for l in rows if not l["correct"])}
            for s in ("complex_plddt", "complex_iplddt")],
        "global_ranking_by_complex_iplddt": [
            {"n": n, "correct": sum(l["correct"] for l in
                                    sorted(rows, key=lambda l: -l["complex_iplddt"])[:n])}
            for n in (13, 15, 20, 25, 50)],
        "seed_level_test": seed_level_test(rows),
        "by_seed": by_seed(rows),
        "repeated_seed_reproducibility": reproducibility(rows),
        "complex_plddt_scale_other_systems": plddt_scale(),
        "interpretation_limit": (
            "A single control with both classes. The separation by complex_plddt is a candidate "
            "for a criterion, not a criterion: it needs a new run with the rule fixed beforehand, "
            "and verification on controls without a correct model. All correct models are the "
            "native pose, so the count of distinct poses of the rule of 4 November is one by "
            "construction."),
        "_rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=RUN_E1, help="E1 run folder in the mirror")
    args = parser.parse_args()
    if not resolve_dir(MIRROR / "modelagem" / "benchmarks" / "execucoes" / args.run).is_dir():
        print(f"ERROR: run {args.run} missing from the mirror; sync it", file=sys.stderr)
        return 1
    summary = summarize(args.run)
    rows = summary.pop("_rows")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure(rows, OUTPUT / "confidence_vs_correctness.pdf")
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    provenance = {"generated_by": RELATIVE_SCRIPT, "run": args.run,
                  "inputs_sha256": dict(sorted(_read.items()))}
    (OUTPUT / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=1, ensure_ascii=False), encoding="utf-8")
    for name in ("summary.json", "confidence_vs_correctness.pdf", "PROVENANCE.json"):
        print(f"  {(OUTPUT / name).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
