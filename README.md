# Analysis code and derived results for the within-target ranking study

Analysis code and derived data for a study of what the confidence scores of a co-folding
predictor do, and do not, establish for antibody–antigen interfaces.

The question is narrow on purpose. A confidence score can be used in two ways that make
different demands on it: to **rank** the models generated for one target, and to **decide**,
against a fixed threshold, whether a predicted interface for a new target can be trusted.
Evidence for the first is routinely cited in support of the second. We measured both, on nine
antibody–antigen complexes with experimental structures and a negative control built from a
published non-cross-reactive pair, and they come apart.

## What is here

| | |
|---|---|
| `scripts/` | the analysis code that produces every number in the article |
| `analyses/` | the derived results those scripts produce: DockQ per model, the set of distinct models with its statistics, and the decomposition of the interface-weighted score |

| script | what it does |
|---|---|
| `scripts/score_dockq.py` | scores every predicted structure against its experimental reference |
| `scripts/analyze_ranking.py` | builds the distinct-model set and computes the statistics of ranking and threshold |
| `scripts/analyze_interface.py` | decomposes the interface-weighted score into antigen–Fv and heavy–light contributions |
| `scripts/summarize_control.py` | summarizes the negative control and its false-positive bound |
| `scripts/bootstrap_auc.py` | seed-clustered bootstrap intervals for the within-target AUCs |
| `scripts/analyze_1vfb.py` | analyses the hundred-seed 1VFB run on its own |

**The raw data are not in this repository.** Predicted structures and the per-run execution
record are deposited separately, with a citable DOI, because they are too large for a git
repository and because a data archive is the right place for them. The deposit comes in three
parts, so that recomputing the statistics does not require downloading the structures:
`structures.zip` (3,426 predicted structures from 24 runs), `execution-record.zip` (the full
execution record and the experimental reference structures) and `code-and-analyses.zip`, with a
SHA-256 manifest covering all of them. The deposit is at
[10.5281/zenodo.24355999](https://doi.org/10.5281/zenodo.24355999) (version 2; concept DOI
[10.5281/zenodo.24348102](https://doi.org/10.5281/zenodo.24348102)).

## Reproducing the numbers

Download and unpack the deposit, then point the scripts at it:

```bash
RECORD=/path/to/deposit/execution-record python scripts/analyze_ranking.py
```

That recomputes the distinct-model set, the areas under the ROC curve per target and pooled, the
seed-level selection counts, what the global threshold accepts and loses, and the agreement
between the two accuracy criteria. It reads only local files.

To rescore the structures from scratch, rather than trusting `analyses/dockq/results.json`:

```bash
python scripts/score_dockq.py --root /path/to/deposit/structures \
                              --reference /path/to/deposit/reference-structures
```

DockQ 2.1.3 requires `numpy<2` and must live in its own environment; the header of that script
explains why and how. Everything else runs on the standard library, except
`analyze_interface.py`, which needs `numpy`.

## Conventions the code depends on

Which chain assignment is kept when an antigen is homodimeric, why repeated seeds are counted
once, and why the baseline for seed-level selection is not one half are decisions recorded in
the module docstrings of the scripts that make them.

## Licence

Code under the MIT licence (`LICENSE`). Derived data under CC BY 4.0. The predictor itself,
Boltz-2, is distributed by its own authors and is not included here.
