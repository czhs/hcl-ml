### Single run (training seed 0, shuffle seed 123)

| Condition | Macro-F1 (test) | Macro-F1 (val) | Trainable params |
|---|---:|---:|---:|
| Direct sensor classifier | **0.9618** | 0.977 | 161,286 |
| Context-embedding model (simplex) | **0.9555** | 0.9741 | 167,052 |
| Context model, shuffled embeddings | **0.1763** | — | — |
| *supplementary:* unconstrained 960-d projector | **0.9149** | 0.9615 | 412,230 |

### Mean ± std over training seeds [0, 1, 2, 3, 4, 5, 6, 7]

| Condition | Macro-F1 (test) | Macro-F1 (val) | per-seed test |
|---|---:|---:|---|
| Direct sensor classifier | **0.9402 ± 0.0154** | 0.9762 ± 0.0049 | 0.962 0.917 0.927 0.939 0.955 0.936 0.931 0.954 |
| Context-embedding model (simplex) | **0.8937 ± 0.1325** | 0.9203 ± 0.1333 | 0.956 0.937 0.923 0.962 0.568 0.951 0.923 0.931 |
| Context model, shuffled embeddings | **0.1652 ± 0.0083** | — | 0.176 0.166 0.165 0.170 0.147 0.170 0.163 0.164 |
| *supplementary:* unconstrained 960-d projector | **0.9138 ± 0.0121** | 0.9700 ± 0.0063 | 0.915 0.929 0.897 0.914 0.914 0.933 0.906 0.904 |
| *supplementary:* simplex code without the LM | **0.6313 ± 0.1146** | 0.6610 ± 0.1221 | 0.537 0.714 0.690 0.729 0.723 0.501 0.451 0.704 |
| — simplex, 7/8 converged seeds (val ≥ 0.8) | **0.9403 ± 0.0159** | | |

### All runs (best validation macro-F1; ablations are val-only)

| run tag | seeds | val macro-F1 | per-seed |
|---|---:|---:|---|
| `bottleneck` | 8 | 0.6610 ± 0.1221 | 0.5582 0.7374 0.7431 0.7519 0.7570 0.5508 0.4485 0.7411 |
| `direct` | 8 | 0.9762 ± 0.0049 | 0.9770 0.9799 0.9819 0.9770 0.9690 0.9806 0.9694 0.9745 |
| `free` | 8 | 0.9700 ± 0.0063 | 0.9615 0.9785 0.9728 0.9611 0.9665 0.9718 0.9730 0.9746 |
| `simplex` | 8 | 0.9203 ± 0.1333 | 0.9741 0.9806 0.9626 0.9734 0.5911 0.9559 0.9710 0.9535 |
| `simplex-geometric` | 3 | 0.9696 ± 0.0038 | 0.9710 0.9725 0.9653 |
| `simplex-geometric1` | 1 | 0.0525 | 0.0525 |
| `simplex-geometric1-gate` | 1 | 0.9361 | 0.9361 |
| `simplex-geometric12` | 1 | 0.9508 | 0.9508 |
| `simplex-geometric16` | 1 | 0.9782 | 0.9782 |
| `simplex-geometric2` | 1 | 0.6940 | 0.6940 |
| `simplex-geometric32` | 1 | 0.9704 | 0.9704 |
| `simplex-geometric4` | 1 | 0.9589 | 0.9589 |
| `simplex-geometric64` | 1 | 0.9545 | 0.9545 |
| `simplex-geometric8` | 1 | 0.9743 | 0.9743 |
| `simplex-learnA` | 3 | 0.9675 ± 0.0040 | 0.9644 0.9661 0.9720 |
| `simplex-learnA-randinit` | 3 | 0.9536 ± 0.0194 | 0.9499 0.9364 0.9746 |
| `simplex-noprompt` | 3 | 0.8418 ± 0.2265 | 0.9779 0.5804 0.9672 |
| `simplex-randLM` | 3 | 0.6172 ± 0.0252 | 0.6334 0.6301 0.5882 |
| `simplex-unrelatedprompt` | 3 | 0.9609 ± 0.0142 | 0.9718 0.9660 0.9449 |

