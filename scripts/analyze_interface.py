#!/usr/bin/env python3
"""Split `complex_iplddt` into the antigen-Fv interface and the antibody heavy-light interface.

    python scripts/analyze_interface.py \
        --run structures/20261105T115759Z_e0e7ca12 \
        --antigen A --antibody H L \
        --dockq analyses/dockq/results.json \
        --output analyses/interface/models_1vfb.json

The predictor's `complex_iplddt` is **not** the mean pLDDT of the interface: it is the mean over
all tokens, with weight 10 for interface tokens, 1 for the others and 20 for ligands (read from
the predictor code, in `src/boltz/model/modules/confidencev2.py`). A token is "interface" when
it lies within 8 Å of a token of **another chain**, in the predicted structure.

In an antibody-antigen system the Fv enters as two chains, so the **heavy-light** contacts count
as interface by the same criterion. This script measures how much of the weighted interface is
the antibody's internal interface — easy, conserved and almost always correct — instead of the
antigen-Fv interface, which is what the article wants to evaluate.

For each model it computes, from the `.cif`:

- `reproduced_iplddt`: the predictor formula, to check against `complex_iplddt` in the JSON.
  If the two do not match, nothing else here is valid, and the script says so.
- `iplddt_antigen_only`: the same formula, but upweighting **only** the antigen-Fv interface
  tokens.
- `interface_mean` and `antigen_interface_mean`: simple means, without weights.
- the token counts of each kind of interface.

It does not depend on gemmi: it reads `_atom_site` from the mmCIF directly. It needs numpy (the
DockQ environment will do). It writes nothing outside `--output`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

INTERFACE_WEIGHT = 10.0        # interface_weight in the predictor code
NON_INTERFACE_WEIGHT = 1.0     # non_interface_weight
CONTACT_CUTOFF = 8.0           # is_contact = (d < 8)


def read_tokens(cif: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (chain per token, coordinate of the central atom, pLDDT per token).

    A token is a residue, and the central atom is the CA — what the predictor uses for standard
    residues. The pLDDT comes from column B of the mmCIF, where the predictor stores it, on the
    0–100 scale.
    """
    chains: list[str] = []
    centers: list[tuple[float, float, float]] = []
    plddts: list[float] = []
    with cif.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            fields = line.split()
            if fields[3] != "CA":          # label_atom_id
                continue
            chains.append(fields[15])      # auth_asym_id
            centers.append((float(fields[10]), float(fields[11]), float(fields[12])))
            plddts.append(float(fields[17]) / 100.0)   # B_iso_or_equiv, 0–100 → 0–1
    return chains, np.array(centers), np.array(plddts)


def masks(chains: list[str], centers: np.ndarray, antigen: set[str],
          antibody: set[str]) -> dict[str, np.ndarray]:
    """Which tokens are interface, and with whom."""
    d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    contact = d < CONTACT_CUTOFF
    cad = np.array(chains)
    other_chain = cad[:, None] != cad[None, :]

    is_antigen = np.isin(cad, list(antigen))
    is_antibody = np.isin(cad, list(antibody))

    # Interface as the predictor defines it: contact with any token of another chain.
    interface = (contact & other_chain).any(axis=1)
    # Antigen-Fv interface: the pair crosses the antigen/antibody boundary.
    cross_pair = (is_antigen[:, None] & is_antibody[None, :]) | (is_antibody[:, None] & is_antigen[None, :])
    antigen_interface = (contact & cross_pair).any(axis=1)
    # Antibody's internal interface: another chain, but both are antibody.
    internal_pair = is_antibody[:, None] & is_antibody[None, :] & other_chain
    internal_interface = (contact & internal_pair).any(axis=1)
    return {"interface": interface, "antigen_interface": antigen_interface,
            "internal_interface": internal_interface,
            "internal_only": internal_interface & ~antigen_interface}


def weighted(plddt: np.ndarray, overweight: np.ndarray) -> float:
    weight = np.where(overweight, INTERFACE_WEIGHT, NON_INTERFACE_WEIGHT)
    return float((plddt * weight).sum() / weight.sum())


def measure(cif: Path, antigen: set[str], antibody: set[str]) -> dict:
    chains, centers, plddt = read_tokens(cif)
    m = masks(chains, centers, antigen, antibody)
    n = len(chains)
    n_iface = int(m["interface"].sum())
    return {
        "tokens": n,
        "interface_tokens": n_iface,
        "antigen_interface_tokens": int(m["antigen_interface"].sum()),
        "internal_only_tokens": int(m["internal_only"].sum()),
        "internal_fraction_of_interface": (float(m["internal_only"].sum()) / n_iface) if n_iface else None,
        "internal_weight_of_score": _weight_share(m["internal_only"], n),
        "antigen_weight_of_score": _weight_share(m["antigen_interface"], n),
        "reproduced_complex_plddt": float(plddt.mean()),
        "reproduced_iplddt": weighted(plddt, m["interface"]),
        "iplddt_antigen_only": weighted(plddt, m["antigen_interface"]),
        "interface_mean": _mean(plddt, m["interface"]),
        "antigen_interface_mean": _mean(plddt, m["antigen_interface"]),
        "internal_interface_mean": _mean(plddt, m["internal_only"]),
    }


def _weight_share(mask: np.ndarray, n: int) -> float:
    """How much of the score denominator these tokens account for, if upweighted to 10."""
    k = int(mask.sum())
    return (INTERFACE_WEIGHT * k) / (INTERFACE_WEIGHT * k + NON_INTERFACE_WEIGHT * (n - k)) if n else 0.0


def _mean(plddt: np.ndarray, mask: np.ndarray) -> float | None:
    return float(plddt[mask].mean()) if mask.any() else None


def reported_confidence(cif: Path) -> dict | None:
    """The confidence JSON the predictor wrote for this model.

    In the deposit, the structures and the execution record are separate archives: the model is
    under `structures/<run>/<job>/` and its confidence file under
    `execution-record/<run>/<job>/`. `RECORD` points to the latter, and the fallback below keeps
    the script usable both inside the deposit and in the workstation mirror, where the file sits
    beside the model.
    """
    target = cif.parent / f"confidence_{cif.stem}.json"
    if target.is_file():
        return json.loads(target.read_text(encoding="utf-8"))
    candidates = sorted(cif.parent.glob("confidence_*.json"))
    suffix = cif.stem.split("_model_")[-1]
    found = next((c for c in candidates if c.stem.endswith(f"_model_{suffix}")), None)
    if found is not None:
        return json.loads(found.read_text(encoding="utf-8"))
    record = os.environ.get("RECORD")
    if record:
        job = next((p for p in cif.parents if "_seed" in p.name), None)
        run = next((p for p in cif.parents if p.parent.name == "structures"), None)
        if job is not None and run is not None:
            fallback = Path(record) / run.name / job.name / f"confidence_{cif.stem}.json"
            if fallback.is_file():
                return json.loads(fallback.read_text(encoding="utf-8"))
    return None


def auc(positives: list[float], negatives: list[float]) -> float | None:
    """AUC by pair counting, which is the definition and needs no dependency."""
    if not positives or not negatives:
        return None
    pairs = [(1 if p > n else 0.5 if p == n else 0) for p in positives for n in negatives]
    return sum(pairs) / len(pairs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, action="append",
                        help="a run folder; may repeat")
    parser.add_argument("--antigen", action="append", required=True, help="antigen chain")
    parser.add_argument("--antibody", nargs="+", required=True, help="antibody chains")
    parser.add_argument("--dockq", type=Path, default=None,
                        help="results.json from score_dockq.py, to separate correct from incorrect")
    parser.add_argument("--output", type=Path, default=None, help="where to write the JSON")
    args = parser.parse_args()

    antigen, antibody = set(args.antigen), set(args.antibody)
    dockq = {}
    if args.dockq:
        raw = json.loads(args.dockq.read_text(encoding="utf-8"))
        items = raw["results"] if isinstance(raw, dict) and "results" in raw else raw
        for item in items:
            # The key needs the case: the file name repeats across seeds, and (seed, sample)
            # alone still collides across targets — a results.json with the nine targets has
            # (11, 0) nine times, and the last one read would overwrite the other eight.
            key = (item["case"], int(item["seed"]), int(item["sample"]))
            if item.get("variant", "main").startswith("main") or key not in dockq:
                dockq[key] = max(dockq.get(key, -1.0), item["dockq"])

    rows = []
    for run in args.run:
        for cif in sorted(run.rglob("*_model_*.cif")):
            measured = measure(cif, antigen, antibody)
            reported = reported_confidence(cif)
            measured["file"] = cif.name
            measured["seed"] = next((p.name for p in cif.parents if "_seed" in p.name), None)
            measured["run"] = run.name
            if reported:
                measured["reported_complex_iplddt"] = reported.get("complex_iplddt")
                measured["reported_complex_plddt"] = reported.get("complex_plddt")
                measured["reported_iptm"] = reported.get("iptm")
            case = cif.stem.split("_model_")[0]
            key = (case, _seed_of(measured["seed"]), int(cif.stem.split("_model_")[-1]))
            measured["case"] = case
            if key in dockq:
                measured["dockq"] = dockq[key]
            rows.append(measured)

    if not rows:
        print("No model found.")
        return 1

    # Check: the reproduced formula must agree with what the predictor reported.
    pairs = [(l["reproduced_iplddt"], l["reported_complex_iplddt"]) for l in rows
             if l.get("reported_complex_iplddt") is not None]
    max_error = max((abs(a - b) for a, b in pairs), default=None)
    print(f"{len(rows)} models; formula check on {len(pairs)} of them: "
          f"maximum error {max_error:.4f}" if max_error is not None else "no JSON to check against")
    if max_error is not None and max_error > 0.02:
        print("  WARNING: the reproduced formula does not match the reported one; the rest is not reliable.")

    fr = [l["internal_fraction_of_interface"] for l in rows if l["internal_fraction_of_interface"] is not None]
    pi = [l["internal_weight_of_score"] for l in rows]
    pa = [l["antigen_weight_of_score"] for l in rows]
    print(f"\nFraction of the interface that is heavy-light only: median {_median(fr):.3f} "
          f"(from {min(fr):.3f} to {max(fr):.3f})")
    print(f"Share of the score that comes from that internal interface: median {_median(pi):.3f}")
    print(f"Share of the score that comes from the antigen-Fv interface: median {_median(pa):.3f}")

    if dockq:
        correct = [l for l in rows if l.get("dockq", 0) >= 0.23]
        incorrect = [l for l in rows if l.get("dockq", 1) < 0.23]
        print(f"\nCorrect (DockQ ≥ 0.23): {len(correct)}; incorrect: {len(incorrect)}")
        print(f"{'measure':<28} {'AUC':>6}  {'correct':>18}  {'incorrect':>18}")
        for key in ("reported_complex_iplddt", "reproduced_iplddt", "iplddt_antigen_only",
                    "interface_mean", "antigen_interface_mean", "internal_interface_mean",
                    "reproduced_complex_plddt", "reported_iptm"):
            p = [l[key] for l in correct if l.get(key) is not None]
            n = [l[key] for l in incorrect if l.get(key) is not None]
            value = auc(p, n)
            if value is None:
                continue
            print(f"{key:<28} {value:>6.3f}  {_range(p):>18}  {_range(n):>18}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"\nWritten to {args.output}")
    return 0


def _seed_of(folder: str | None) -> int:
    """"…_seed11" → 11."""
    return int(folder.rsplit("_seed", 1)[-1]) if folder and "_seed" in folder else -1


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def _range(values: list[float]) -> str:
    return f"{min(values):.4f}–{max(values):.4f}" if values else "—"


if __name__ == "__main__":
    raise SystemExit(main())
