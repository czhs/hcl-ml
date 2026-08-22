# Reference results

The numbers quoted in `TECHNICAL_NOTE.md` and the README, exactly as produced
by the original runs (RTX 4090, driver 535.309.01, torch 2.5.1+cu121,
transformers 5.15.1; seeds 5–7 of the multi-seed sweep ran on an RTX 3060).

| file | what |
|---|---|
| `test_results.json` | the three challenge conditions + supplementary `free`, training seed 0, shuffle seed 123 |
| `seed_summary.json` | val/test macro-F1 for training seeds 0–7, every variant; shuffled control with shuffle seed 123+s |
| `all_runs.json` | best-val / best-epoch / parameter count / per-epoch curves for all 63 runs, including every ablation (`python -m hcl.collect_runs`) |
| `mech_probe.json` | layer-wise probes / attention / logit-lens on `simplex_s1` (`python -m hcl.mech_probe --run simplex_s1`) |
| `continuity.json` | softness / PCA / discreteness / smoothness of the code w on `simplex_s0` |
| `splits.json` | the subject-wise split (split seed 42; val subjects 17, 25, 26, 30) |
| `checkpoints/<variant>_s0/` | trained weights (trainable parts only — the frozen LM is never saved), normalisation statistics, per-epoch metrics, for `direct`, `simplex`, `free`, `bottleneck`; plus `simplex_s1` (the best-val simplex seed, used by the mech probe) |

Evaluate the shipped checkpoints on the test split without training anything:

```bash
python -m hcl.eval_test --results reference/checkpoints --variants direct simplex free bottleneck \
       --out results/reference_eval.json
```

This must reproduce `test_results.json` exactly (direct 0.9618 · simplex 0.9555 ·
shuffled 0.1763 · free 0.9149): evaluation is deterministic given the weights.

Run names: `<variant>[-<ablation tags>]_s<seed>`, e.g. `simplex-geometric_s1`,
`simplex-learnA-randinit_s2`, `simplex-geometric16_s0`, `simplex-randLM_s0`,
`simplex-noprompt_s0`, `simplex-unrelatedprompt_s0`, `simplex-geometric1-gate_s0`.
The learnable-corner K-sweep (K ≥ 8) ran on a cloud box whose checkpoints were
not retained; its best-val numbers are listed in the README from the run logs.
