# DockQ scoring of the article set, 18 November 2026 version

`results.json` is the union of two scoring runs, both with
`modelagem/benchmarks/pontuar_dockq.py`, DockQ 2.1.3 and the same interface convention
(antigen–Fv only, chains H and L fused, antigen protomers fused):

1. the official scoring of 11 November, `dockq_20261111T111747Z`, with **2,248 models**,
   which is the content of the sibling folder [`dockq_oficial/`](../dockq_oficial/);
2. the **500 models** of run `20261117T105747Z_c4539a23` — experiment P, 35UI at one
   hundred seeds — scored on this machine from the Drive mirror on 18 November, 0 failures.

Total: **2,748 records, 0 failures**. After the cleaning done by `analyze_ranking.py` (23
pilot records from October and 125 duplicates, 25 of them the seeds 11 to 55 of 35UI that
the new run repeats from the 6 November round), **2,600 distinct models** remain.

**What was left out, on purpose:** the 25 models of 1VFB at 256 MSA rows
(`20261117T104630Z_d224895c`), scored in the same round. They are another input condition
and would collide, on seeds 11 to 55, with the default-depth models. The depth runs are
analysed separately, in [resultado_35ui_cem_sementes.md](../../resultado_35ui_cem_sementes.md).

`analyze_ranking.py` reads this folder, and no longer `dockq_oficial/`, which stays intact
as the record of the 11 November scoring.
