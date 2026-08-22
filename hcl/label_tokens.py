#!/usr/bin/env python3
"""Single-token anchor identifiers for the six activities in SmolLM2's vocabulary.

Method
  1. Universe = the word-like vocab tokens (" [a-z]{3,}", the form that appears
     mid-prompt) embedded with the tied embedding table, centred on the vocab
     mean and L2-normalised.
  2. Per class, a synonym island: hand-picked seed words that survive as single
     tokens -> centroid -> nearest word tokens above a similarity floor
     (islands are disjoint; contested tokens go to the closer centroid).
  3. Profiles — one token per island:
       canonical   the token closest to its island centroid (the plain synonym)
       balanced    the most separated sextet that keeps (almost) all of the
                   canonical profile's synonym strength: randomised hill-climb
                   minimising the (max, mean) pairwise cosine among the six
                   picks subject to sum of cosine-to-own-centroid >= 4.10
                   (canonical: 4.125)                           <- used by the model
       orthogonal  unconstrained hill-climb over candidates with cosine to own
                   centroid >= 0.42: 4x more separation, weaker meaning

Writes configs/label_tokens.json (deterministic: SEED=0). Pass --check to
verify that the committed file regenerates instead of overwriting it.
The maximally-separated "geometric" ladders are added by hcl.geometric_anchors.
"""
import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .paths import CLASSES, LABEL_TOKENS, LM_ID

SEED = 0
FLOOR = 0.35            # island membership: min cosine to island centroid
MAX_ISLAND = 25
ANCHOR = 0.42            # candidate floor: min cosine to own island centroid
BALANCED_MIN_SUM = 4.10  # balanced profile: sum over classes of cosine to own centroid
RESTARTS = 300

SEEDS = {
    "WALKING": ["walking", "walk", "strolling", "stroll", "pacing", "hiking",
                "marching", "trekking", "wandering", "strides"],
    "WALKING_UPSTAIRS": ["upstairs", "ascending", "ascent", "climbing", "climb",
                         "rising", "upward", "uphill"],
    "WALKING_DOWNSTAIRS": ["downstairs", "descending", "descent", "downward",
                           "downhill", "dropping", "sinking"],
    "SITTING": ["sitting", "seated", "sedentary", "perched", "crouching"],
    "STANDING": ["standing", "upright", "stationary", "erect", "still", "vertical"],
    "LAYING": ["laying", "lying", "reclining", "recumbent", "prone", "supine",
               "horizontal", "resting", "sleeping", "asleep"],
}


def embeddings():
    tok = AutoTokenizer.from_pretrained(LM_ID)
    model = AutoModelForCausalLM.from_pretrained(LM_ID, dtype=torch.float32)
    E = model.get_input_embeddings().weight.detach().numpy()
    Ec = E - E.mean(0)
    N = Ec / np.linalg.norm(Ec, axis=1, keepdims=True)
    words = {i: s for i in range(len(E)) if re.fullmatch(r" [a-z]{3,}", (s := tok.decode([i])))}
    return tok, N, words


def build_islands(tok, N, words):
    wid = np.fromiter(words, dtype=np.int64)

    def single_id(w):
        ids = tok(" " + w, add_special_tokens=False).input_ids
        return ids[0] if len(ids) == 1 else None

    cents = {}
    for cls, seeds in SEEDS.items():
        sids = [t for w in seeds if (t := single_id(w)) is not None]
        c = N[sids].mean(0)
        cents[cls] = c / np.linalg.norm(c)
    simmat = np.stack([N[wid] @ cents[c] for c in CLASSES])           # (6, |universe|)
    owner = simmat.argmax(0)
    islands, member_sim = {}, {}
    for k, cls in enumerate(CLASSES):
        mine = np.where((owner == k) & (simmat[k] >= FLOOR))[0]
        order = mine[np.argsort(-simmat[k][mine])][:MAX_ISLAND]
        islands[cls] = [int(wid[j]) for j in order]
        member_sim[cls] = {int(wid[j]): float(simmat[k][j]) for j in order}
    return islands, member_sim


def pair_score(N, sel):
    V = N[[sel[c] for c in CLASSES]]
    off = (V @ V.T)[np.triu_indices(len(CLASSES), 1)]
    return float(off.max()), float(off.mean())


def hill_climb(N, islands, member_sim, anchor=ANCHOR, min_sum=0.0, seed=SEED, restarts=RESTARTS):
    """Minimise (max, mean) pairwise cosine over one token per island, with every
    candidate at least `anchor` from its centroid and, optionally, the total
    centroid similarity at least `min_sum` (the canonical picks are always feasible)."""
    cand = {c: [t for t in islands[c] if member_sim[c][t] >= anchor] or islands[c][:5] for c in CLASSES}
    asum = lambda sel: sum(member_sim[c][sel[c]] for c in CLASSES)
    rng = random.Random(seed)
    best, best_s = None, (2.0, 2.0)
    for _ in range(restarts):
        sel = {c: rng.choice(cand[c]) for c in CLASSES}
        if asum(sel) < min_sum:
            sel = {c: islands[c][0] for c in CLASSES}
        improved = True
        while improved:
            improved = False
            for c in CLASSES:
                cur = pair_score(N, sel)
                for t in cand[c]:
                    trial = {**sel, c: t}
                    if asum(trial) >= min_sum and pair_score(N, trial) < cur:
                        sel, cur, improved = trial, pair_score(N, trial), True
        if pair_score(N, sel) < best_s:
            best, best_s = dict(sel), pair_score(N, sel)
    return best


def profile_record(N, words, member_sim, sel):
    V = N[[sel[c] for c in CLASSES]]
    Mx = np.round(V @ V.T, 3)
    off = Mx[np.triu_indices(6, 1)]
    return {"tokens": {c: {"id": int(sel[c]), "token": words[sel[c]],
                           "anchor": round(member_sim[c][sel[c]], 3)} for c in CLASSES},
            "max_pair": float(off.max()), "mean_pair": float(round(off.mean(), 3)),
            "anchor_sum": float(round(sum(member_sim[c][sel[c]] for c in CLASSES), 3)),
            "matrix": Mx.tolist()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare with the committed file, do not write")
    ap.add_argument("--out", default=str(LABEL_TOKENS))
    args = ap.parse_args()

    tok, N, words = embeddings()
    print(f"word-like universe: {len(words)} tokens")
    islands, member_sim = build_islands(tok, N, words)
    for cls in CLASSES:
        print(f"  island {cls}: " + ", ".join(f"{words[t]!r}:{member_sim[cls][t]:.2f}" for t in islands[cls][:8]))

    sels = {"canonical": {c: islands[c][0] for c in CLASSES},
            "balanced": hill_climb(N, islands, member_sim, min_sum=BALANCED_MIN_SUM),
            "orthogonal": hill_climb(N, islands, member_sim)}
    profiles = {name: profile_record(N, words, member_sim, sel) for name, sel in sels.items()}
    for name, p in profiles.items():
        print(f"{name:10s} max pair {p['max_pair']:.3f}  anchor sum {p['anchor_sum']:.2f}  "
              + " ".join(p["tokens"][c]["token"].strip() for c in CLASSES))

    out = Path(args.out)
    existing = json.loads(out.read_text()) if out.exists() else {}
    if args.check:
        ok = True
        for name, p in profiles.items():
            have = existing.get("profiles", {}).get(name, {}).get("tokens", {})
            mine = {c: p["tokens"][c]["id"] for c in CLASSES}
            theirs = {c: have.get(c, {}).get("id") for c in CLASSES}
            status = "OK" if mine == theirs else f"MISMATCH (file has {[have.get(c, {}).get('token') for c in CLASSES]})"
            ok &= mine == theirs
            print(f"check {name:10s} {status}")
        raise SystemExit(0 if ok else 1)

    orth = sels["orthogonal"]
    record = {
        "chosen": {c: {"id": int(orth[c]), "token": words[orth[c]], "anchor_sim": member_sim[c][orth[c]]} for c in CLASSES},
        "score": dict(zip(("max_pair", "mean_pair"), pair_score(N, orth))),
        "matrix": {"classes": CLASSES, "values": profiles["orthogonal"]["matrix"]},
        "islands": {c: [{"id": t, "token": words[t], "sim": member_sim[c][t]} for t in islands[c]] for c in CLASSES},
        "baseline_original_labels": {"up_vs_down": 0.817, "max_pair": 0.817, "note": "orig labels, centered"},
        "params": {"seed": SEED, "floor": FLOOR, "anchor": ANCHOR, "balanced_min_sum": BALANCED_MIN_SUM,
                   "universe": int(len(words)), "restarts": RESTARTS},
        "profiles": {**existing.get("profiles", {}), **profiles},   # keep geometric ladders
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
