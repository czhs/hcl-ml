# Technical note — Sensor Context Encoder Challenge

Chris Shi · 2026-08-21 · code and reference results: <https://github.com/czhs/HCL-ML> · writeup: <https://czhs.github.io/HCL-ML/>

**Summary.** I built the required pair — a direct sensor classifier and a context-embedding model sharing its encoder — plus one design choice that makes the interface interpretable by construction: the context embedding is constrained to a *simplex* whose corners are six real token embeddings. On the held-out subjects both models reach macro-F1 0.94, the shuffled-embedding check collapses to chance, and the trainable parameter count is 1.7 % of the budget. The interface works; what the language model contributes, the ablations show, is a well-conditioned channel rather than understanding.

## Sensor encoder and projector design

**Encoder** (identical in every condition, so the comparison isolates the interface rather than the feature extractor): a 1-D CNN over the raw 128×9 window — Conv1d 9→64 (k5, stride 2), 64→128 (k5, stride 2), 128→128 (k3), each BatchNorm + GELU, then global mean- and max-pooling over time concatenated into a 256-d feature and a Linear + GELU projection; 160,742 parameters. Convolution suits the signal because the label is invariant to *where* in the window a step lands; mean-pooling carries the gravity/posture statistics that separate the static classes, max-pooling the transient impacts that separate the gait classes. Inputs are z-scored per channel with training-split statistics.

**Projector.** A single Linear 256→6 followed by a softmax gives barycentric weights *w* on the 5-simplex; the context embedding is *e* = *w*·A, where A holds six frozen rows of the LM's own embedding table. The corners come from a search over the 21,185 single word-like tokens in SmolLM2's vocabulary: grow a synonym island around each activity in centred embedding space, then pick one token per island minimising the worst pairwise cosine while keeping each a recognisable synonym. The result — *walking · climb · descending · seated · upright · lying* — has a maximum pairwise cosine of 0.24, against 0.82 for the original label phrases (which share the token ` walking`). *e* is a convex combination of genuine token embeddings, so it lies on the LM's input manifold by construction, and the encoder's entire interface to the LM is six numbers.

**Language-model path.** *e* replaces `<SENSOR>` in the mandated prompt's input-embedding sequence (position 28 of 32); sensor values are never rendered as text. The LM is SmolLM2-360M-Instruct in bf16, fully frozen including its embedding table; gradients flow through it into the projector and encoder. A trainable Linear 960→6 head reads the final hidden state after `Activity:`. The direct classifier is the identical trunk with a Linear 256→6 head.

## Training setup and trainable parameter count

AdamW, learning rate 1e-3 with cosine decay, weight decay 1e-4, batch 128, 30 epochs, cross-entropy loss. The official subject-wise test split was never touched during development; model selection used a subject-wise validation split — subjects 17, 25, 26 and 30, carved from the 21 training subjects with seed 42 — giving 5,800 / 1,552 / 2,947 windows for train / val / test. For each training seed the best-validation checkpoint is evaluated exactly once on test. A seed-0 run of all conditions takes ≈ 5 min on one RTX 4090.

| Model | Trainable parameters |
|---|---:|
| Direct classifier (trunk + head) | 161,286 |
| Context model (trunk + simplex projector + LM head) | **167,052** |

Both ≈ 1.7 % of the 10 M budget; the LM's 361.8 M parameters are frozen. (The supplementary unconstrained-projector ablation used 412,230.)

## Results and interpretation

Macro-F1 on the test split, single run (training seed 0, shuffle seed 123) and mean ± std over 8 training seeds (0–7; shuffle seed 123 + s). The shuffled condition permutes *complete* projected embeddings between test examples with no retraining.

| Condition | seed 0 | 8 seeds |
|---|---:|---:|
| Direct sensor classifier | **0.9618** | **0.9402 ± 0.0154** |
| Context-embedding model (simplex) | **0.9555** | **0.8937 ± 0.1325** — 7/8 converged seeds: 0.9403 ± 0.0159 |
| Context model with shuffled embeddings | **0.1763** | **0.1652 ± 0.0083** |
| *supplementary:* unconstrained 960-d projector | 0.9149 | 0.9138 ± 0.0121 |
| *supplementary:* same 6-d code without the LM | — | 0.6313 ± 0.1146 |

The frozen-LM interface is free: on its converged seeds the context model and the direct classifier are statistically indistinguishable on unseen subjects. One simplex seed in eight settled in a bad basin (val 0.59, test 0.57) — loud on validation, so a val gate catches it, but a real ~1-in-8 instability at these hyperparameters. The shuffled control falls to chance (1/6 ≈ 0.167): the prediction depends on the matching sensor embedding, not on prompt priors. Errors land where the sensor physics predicts — sitting vs standing dominates the confusion for both models (a waist-mounted phone sees nearly the same gravity vector in both postures) — not where the label tokens are most similar, so the context path is not bottlenecked by label semantics.

Ablations (validation macro-F1, 3 seeds unless noted) locate what the LM contributes. An unconstrained 960-d projector at the same output radius scores lower on test (0.914) and trains unstably, so the on-manifold constraint does real work. Removing the LM but keeping the six-number bottleneck collapses to 0.66 ± 0.12 (8 seeds): *w* saturates to one-hot and gradients through the softmax die — the frozen LM keeps the code soft and trainable. A randomly initialised frozen transformer of the same shape (pretrained embedding table kept) drops to 0.617 ± 0.025: pretraining is worth ≈ 35 points. Yet deleting the prompt (0.978 / 0.967 on converged seeds) or swapping it for an unrelated recipe (0.961 ± 0.014) changes nothing, and six maximally separated *meaning-free* anchors (*the · antioxid · agitated · salaries · whe · toggle*) tie the synonym anchors (0.970 ± 0.004 vs 0.967 ± 0.010). The pretrained network matters as a well-conditioned, information-preserving channel — not as a reader of the prompt or of the anchor words.

## Known limitations

- **The anchors are not semantically used.** Meaning-free anchors tie, and the encoder's frame is a derangement (walking windows sit by the *descending* corner). Separation matters; meaning does not.
- **The code is a coordinate, not a classification.** argmax(*w*) matches the label only ~31 % of the time; the LM and head decode the hull's interior jointly, so *w* must not be read as class probabilities.
- **Capacity saturates early.** ~4 of the 5 degrees of freedom are used; a sweep over the number of corners plateaus from K ≈ 4 to 64, and a single anchor with a sigmoid strength gate already reaches 0.936 — one scalar through the LM.
- **The transformer functions as a wire.** A layer-wise probe finds the activity fully decodable at the readout position right after layer 1 (0.97) and flat for 31 more layers — transported, not computed — and the logit lens shows the vocabulary space ignoring the sensor until the last few layers. The wire is nonetheless load-bearing (no-LM bottleneck 0.66).
- **Stability and invariance.** ~1-in-8 seed collapse (untested stabilisers: temperature, entropy bonus, projector warm-up); the code also carries *who* is moving — a probe on *w* identifies the unseen validation subject at up to 2× chance — a style channel behind part of the val→test gap and a soft biometric if embeddings leave the device.
- One dataset and sensor placement; validation is 4 subjects, so val−test gaps of 2–4 points are expected cross-subject noise; the LM path costs a 360 M forward per window against a 161 k CNN.

## Recommendation

**Develop the context-embedding approach further — mechanistically, not by chasing accuracy.** The premise holds: a sensor can write into a frozen language model as one continuous vector and be read out at parity with a direct classifier, with a clean sensor-dependence check and a projector whose six corners are auditable tokens. The simplex parameterisation is the interface pattern I would reuse for further modalities: on-manifold by construction, stable where the unconstrained projector is not, six interpretable numbers wide. But every ablation says the model is used as a structured non-linear function, not as something that understands the signal, so the next step is not a better encoder but a direct test of understanding — train the model to *describe* its input (self-explanation in the style of Li et al., arXiv:2511.08579), instrumented with the layer-wise probes that proved decisive here. For activity recognition alone, the direct classifier remains the better engineering choice: equal accuracy at a fraction of the inference cost.
