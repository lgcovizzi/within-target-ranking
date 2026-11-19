#!/usr/bin/env python3
"""Score the already predicted models with DockQ, against the experimental structure.

DockQ is the metric the evaluation literature uses, with 0.23 for a correct interface
and 0.49 for medium quality. Without it, none of our numbers is comparable to a
published benchmark. This script predicts nothing: it reads the `.cif` files that
already exist and scores them.

    python scripts/score_dockq.py --list
    python scripts/score_dockq.py --case 13ha_fv_tnfrsf8 --limit 5
    python scripts/score_dockq.py            # every system

**Separate environment, mandatory.** DockQ 2.1.3 requires `numpy<2` and the rest of
the project requires `numpy>=2`. Installing DockQ in the project environment breaks
SciPy. Create an environment just for it:

    python -m venv .venv_dockq
    .venv_dockq/bin/pip install DockQ          # Windows: .venv_dockq\\Scripts\\pip install DockQ
    .venv_dockq/bin/python scripts/score_dockq.py

That is why this file **imports nothing from the project**: only DockQ and the standard
library.

## Which interface is scored

Only the **antigen–Fv** interface. The heavy and light chains are fused into a single
chain before scoring, and the interface between them is discarded. This is what CASP16
does on antibody–antigen targets, because the antibody's internal orientation is not
what one wants to measure. Antigen protomers are fused as well, so that a quaternary
epitope counts as a single interface.

## What it does not do

It does not cut the Fab of the reference down to the Fv: DockQ aligns the sequences and
uses the residues in common, so the constant domain that exists only in the deposit
falls outside the alignment. It does not replace the contact-pair F1, which each
control's evaluators still compute; it runs alongside it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT_DEFAULT = Path(__file__).resolve().parent.parent / "structures"
REFERENCE_DEFAULT = Path(__file__).resolve().parent.parent / "reference-structures"
OUTPUT_DEFAULT = Path(__file__).resolve().parent.parent / "analyses" / "dockq"

ANTIGEN = "R"    # antigen chain in the reduced structure
FV = "F"         # Fv chain in the reduced structure

# DockQ thresholds from Basu and Wallner (2016), as CAPRI-Q reproduces them.
THRESHOLD_ACCEPTABLE = 0.23
THRESHOLD_MEDIUM = 0.49
THRESHOLD_HIGH = 0.80

# Chain roles in each system's experimental structure. They came from the constants of
# the evaluators themselves: AUTH_CHAINS of benchmark_1vfb, COPIES of benchmark_5nhr,
# AUTH of benchmark_9sg0 and the `auth` field of the case table in benchmark_e2.
# Repeated here, not imported, because this script runs in another environment.
SYSTEMS: dict[str, dict] = {
    "1vfb_fv_lysozyme": {
        "reference": "1vfb.cif",
        "antigen": ["C"], "fv": ["B", "A"],          # B = heavy, A = light
    },
    "5nhr_fv_acvr2b": {
        "reference": "5nhr.cif",
        "antigen": ["C"], "fv": ["H", "L"],
        "copies": {"crystal_copy_2": {"antigen": ["D"], "fv": ["A", "B"]}},
    },
    "9sg0_fv_il17a_dimer": {
        "reference": "9sg0-assembly1.cif",
        "antigen": ["A", "A-2"], "fv": ["H", "L"],
    },
    "2vxs_fv_il17a_dimer": {
        "reference": "2vxs-assembly1.cif",
        "antigen": ["C", "D"], "fv": ["J", "N"],
    },
    "5f3b_fv_myostatin_dimer": {
        "reference": "5f3b-assembly1.cif",
        "antigen": ["C", "D"], "fv": ["A", "B"],
    },
    "6wzm_fv_il8_dimer": {
        "reference": "6wzm-assembly1.cif",
        "antigen": ["E", "F"], "fv": ["A", "B"],
    },
    "9gvn_fv_il5_dimer": {
        "reference": "9gvn-assembly1.cif",
        "antigen": ["A", "B"], "fv": ["C", "D"],
    },
    "35ui_fv_tfpi": {
        "reference": "35ui.cif",
        "antigen": ["T"], "fv": ["H", "L"],
    },
    "13ha_fv_tnfrsf8": {
        "reference": "13ha.cif",
        "antigen": ["A"], "fv": ["B", "C"],
    },
}

# In the predicted models the convention is fixed: antigen in A (and B, if a dimer),
# heavy in H and light in L. Checked in the runs of 3 to 6 November 2026.
MODEL_FV = ("H", "L")


def capri_class(dockq: float) -> str:
    """Quality class by DockQ range, as in Basu and Wallner (2016)."""
    if dockq >= THRESHOLD_HIGH:
        return "high"
    if dockq >= THRESHOLD_MEDIUM:
        return "medium"
    if dockq >= THRESHOLD_ACCEPTABLE:
        return "acceptable"
    return "incorrect"


def reduce(path: Path, antigen: list[str], fv: list[str], destination: Path) -> dict:
    """Write a PDB with two chains: fused antigen and fused Fv.

    Fuses them so that the antigen–Fv interface is a single pair, and so that the
    internal heavy–light interface is not scored. Residues are renumbered sequentially
    because the fusion can repeat numbering across the source chains.
    """
    from Bio.PDB import MMCIFParser, PDBIO, PDBParser
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model
    from Bio.PDB.Chain import Chain

    suffix = path.suffix.lower()
    parser = PDBParser(QUIET=True) if suffix == ".pdb" else MMCIFParser(QUIET=True)
    source = parser.get_structure("x", str(path))[0]
    present = {ch.id for ch in source}

    new = Structure("r")
    model = Model(0)
    new.add(model)
    counts = {}
    for destination_id, names in ((ANTIGEN, antigen), (FV, fv)):
        chain = Chain(destination_id)
        n = 0
        for name in names:
            if name not in present:
                raise KeyError(f"chain {name!r} missing in {path.name}; present: {sorted(present)}")
            for residue in source[name]:
                if residue.id[0] != " ":          # drop water and heteroatoms
                    continue
                copy = residue.copy()
                n += 1
                copy.id = (" ", n, " ")
                chain.add(copy)
        if n == 0:
            raise ValueError(f"no polymer residue in {names} of {path.name}")
        counts[destination_id] = n
        model.add(chain)

    io = PDBIO()
    io.set_structure(new)
    io.save(str(destination))
    return counts


def score(model: Path, reference: Path, antigen: list[str], fv: list[str],
          antigen_model: list[str] | None = None,
          fv_model: list[str] | None = None) -> dict:
    """DockQ of a model against a reference, on the antigen–Fv interface only.

    `antigen_model` fixes the order in which the model's protomers are fused. The order
    matters because the fusion renumbers the residues sequentially: joining A+B against
    a reference whose first protomer corresponds to B aligns the two halves swapped and
    sinks the DockQ. The caller tries both orders.
    """
    from DockQ.DockQ import load_PDB, run_on_all_native_interfaces

    if antigen_model is None:
        antigen_model = ["A", "B"] if len(antigen) > 1 else ["A"]
    if fv_model is None:
        fv_model = list(MODEL_FV)
    with tempfile.TemporaryDirectory() as tmp:
        model_pdb = Path(tmp) / "model.pdb"
        native_pdb = Path(tmp) / "native.pdb"
        model_counts = reduce(model, antigen_model, list(fv_model), model_pdb)
        reference_counts = reduce(reference, antigen, fv, native_pdb)
        result, _ = run_on_all_native_interfaces(
            load_PDB(str(model_pdb)), load_PDB(str(native_pdb)),
            chain_map={ANTIGEN: ANTIGEN, FV: FV})
    if not result:
        raise ValueError("DockQ found no antigen–Fv interface in the reference")
    (_, d), = result.items()
    return {
        "dockq": round(float(d["DockQ"]), 4),
        "capri_class": capri_class(float(d["DockQ"])),
        "fnat": round(float(d["fnat"]), 4),
        "fnonnat": round(float(d["fnonnat"]), 4),
        "f1_dockq": round(float(d["F1"]), 4),
        "irmsd": round(float(d["iRMSD"]), 3),
        "lrmsd": round(float(d["LRMSD"]), 3),
        "clashes": int(float(d.get("clashes", 0))),
        "native_contacts": int(float(d["nat_total"])),
        "recovered_contacts": int(float(d["nat_correct"])),
        "model_residues": model_counts,
        "reference_residues": reference_counts,
    }


def find_models(root: Path, case: str) -> list[Path]:
    """Every predicted `.cif` of a case, in stable seed and sample order."""
    found = sorted(root.glob(f"*/{case}_seed*/**/*_model_*.cif"))
    return [path for path in found if "output_anterior_" not in str(path)]


def model_label(path: Path, case: str) -> dict:
    """Seed and sample number, read from the path."""
    seed = sample = None
    for part in path.parts:
        if part.startswith(f"{case}_seed"):
            seed = int(part.rsplit("seed", 1)[1])
    name = path.stem
    if "_model_" in name:
        sample = int(name.rsplit("_model_", 1)[1])
    return {"seed": seed, "sample": sample}


def run(root: Path, reference: Path, cases: list[str], limit: int | None) -> dict:
    start = time.time()
    records, failures = [], []
    for case in cases:
        spec = SYSTEMS[case]
        ref = reference / spec["reference"]
        if not ref.is_file():
            failures.append({"case": case, "error": f"missing reference: {ref}"})
            continue
        models = find_models(root, case)
        if limit:
            models = models[:limit]
        print(f"{case}: {len(models)} models", file=sys.stderr, flush=True)
        base = {"main": {"antigen": spec["antigen"], "fv": spec["fv"]}}
        base.update(spec.get("copies", {}))
        # Identical protomers: the two permutations of the antigen in the model, as in
        # the symmetry rule that the project's evaluators already apply.
        variants = {}
        for name, v in base.items():
            if len(v["antigen"]) > 1:
                variants[f"{name}:AB"] = {**v, "antigen_model": ["A", "B"]}
                variants[f"{name}:BA"] = {**v, "antigen_model": ["B", "A"]}
            else:
                variants[name] = {**v, "antigen_model": ["A"]}
        for i, model in enumerate(models, 1):
            row = {"case": case, "model": str(model), **model_label(model, case)}
            best = None
            for name, v in variants.items():
                try:
                    result = score(model, ref, v["antigen"], v["fv"], v["antigen_model"])
                except Exception as exc:                    # noqa: BLE001
                    failures.append({"case": case, "model": str(model), "variant": name,
                                     "error": f"{type(exc).__name__}: {exc}"})
                    continue
                result["variant"] = name
                if best is None or result["dockq"] > best["dockq"]:
                    best = result
            if best is None:
                continue
            row.update(best)
            records.append(row)
            if i % 25 == 0:
                print(f"  {i}/{len(models)}", file=sys.stderr, flush=True)
    return {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dockq_version": dockq_version(), "root": str(root),
            "seconds": round(time.time() - start, 1),
            "thresholds": {"acceptable": THRESHOLD_ACCEPTABLE, "media": THRESHOLD_MEDIUM,
                           "high": THRESHOLD_HIGH},
            "n": len(records), "results": records, "failures": failures}


def dockq_version() -> str:
    try:
        import importlib.metadata as md
        return md.version("DockQ")
    except Exception:                                        # noqa: BLE001
        return "unknown"


def summarize(data: dict) -> list[str]:
    """One line per system: how many models in each CAPRI class."""
    by_case = {}
    for r in data["results"]:
        c = by_case.setdefault(r["case"], {"n": 0, "high": 0, "medium": 0, "acceptable": 0,
                                           "incorrect": 0, "total": 0.0})
        c["n"] += 1
        c[r["capri_class"]] += 1
        c["total"] += r["dockq"]
    lines = [f"{'SYSTEM':<26}{'n':>4}{'DockQ avg':>11}{'correct':>9}{'high':>6}{'medium':>7}{'accept.':>8}"]
    for case, c in sorted(by_case.items()):
        correct = c["high"] + c["medium"] + c["acceptable"]
        lines.append(f"{case:<26}{c['n']:>4}{c['total']/c['n']:>11.3f}"
                     f"{100*correct/c['n']:>8.1f}%{c['high']:>6}{c['medium']:>7}{c['acceptable']:>8}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT,
                        help="folder of the runs (default: modelagem/benchmarks/execucoes)")
    parser.add_argument("--reference", type=Path, default=REFERENCE_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT,
                        help="output folder, or a .json file for a flat results file")
    parser.add_argument("--case", action="append", choices=sorted(SYSTEMS),
                        help="may repeat; without it, every system")
    parser.add_argument("--limit", type=int, default=None,
                        help="score at most N models per case; for a quick check")
    parser.add_argument("--list", action="store_true",
                        help="only count the models found, without scoring")
    args = parser.parse_args(argv)

    cases = args.case or sorted(SYSTEMS)
    if args.list:
        for case in cases:
            ref = args.reference / SYSTEMS[case]["reference"]
            print(f"{case:<26}{len(find_models(args.root, case)):>5} models   "
                  f"reference {'ok' if ref.is_file() else 'MISSING'}")
        return 0

    data = run(args.root, args.reference, cases, args.limit)
    if args.output.suffix.lower() == ".json":
        # Direct mode: the package README writes to analyses/dockq/results.json, which is the
        # file analyze_ranking.py reads. No timestamped folder to rename afterwards.
        destination = args.output.parent
        destination.mkdir(parents=True, exist_ok=True)
        results_path, summary_path = args.output, destination / "summary.txt"
    else:
        destination = args.output / f"dockq_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        destination.mkdir(parents=True, exist_ok=True)
        results_path, summary_path = destination / "results.json", destination / "summary.txt"
    results_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = summarize(data)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n{data['n']} models scored in {data['seconds']} s; "
          f"{len(data['failures'])} failures. Written to {results_path}")
    return 1 if data["failures"] and not data["n"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
