#!/usr/bin/env python3
"""Select K MAXIMALLY-SEPARATED word tokens as simplex corners — geometry only,
semantics ignored. Companion to the balanced (synonym) anchors: tests whether
anchor meaning matters at all, or only separation.

Greedy max-min (farthest-point) on centred, L2-normalised embeddings over the
same word-like universe (" [a-z]{3,}") used for the synonym islands. The greedy
order is nested, so its first K elements form the K-corner ladder used by the
K-sweep. Adds profiles "geometric" (K=6, class-keyed) and "geometric{K}" for
K in 1,2,4,8,12,16,32,64 to configs/label_tokens.json.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .paths import CLASSES, LABEL_TOKENS, LM_ID

KMAX = 64
LADDER = [1, 2, 4, 8, 12, 16, 32, 64]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=str(LABEL_TOKENS))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(LM_ID)
    m = AutoModelForCausalLM.from_pretrained(LM_ID, dtype=torch.float32)
    E = m.get_input_embeddings().weight.detach().numpy()
    Ec = E - E.mean(0)
    N = (Ec / np.linalg.norm(Ec, axis=1, keepdims=True)).astype(np.float32)
    ids = np.array([i for i in range(len(E)) if re.fullmatch(r" [a-z]{3,}", tok.decode([i]))])
    V = N[ids]
    print(f"universe {len(ids)} word tokens")

    S = V @ V.T                                   # (21k, 21k) cosine, ~1.8 GB
    np.fill_diagonal(S, 1.0)
    i, j = np.unravel_index(np.argmin(S), S.shape)  # farthest pair
    sel = [int(i), int(j)]
    maxsim = np.maximum(S[i], S[j])
    while len(sel) < KMAX:                        # add the token whose WORST similarity to the set is lowest
        maxsim[sel] = 2.0
        k = int(np.argmin(maxsim))
        sel.append(k)
        maxsim = np.maximum(maxsim, S[k])
    chosen = [int(ids[k]) for k in sel]
    names = [tok.decode([t]) for t in chosen]

    out = Path(args.out)
    cfg = json.loads(out.read_text())
    M6 = np.round(N[chosen[:6]] @ N[chosen[:6]].T, 3)
    off6 = M6[np.triu_indices(6, 1)]
    print("first 6:", names[:6], f"max pair {off6.max():.3f}")
    new = {"geometric": {
        "tokens": {c: {"id": t, "token": n, "anchor": None} for c, t, n in zip(CLASSES, chosen[:6], names[:6])},
        "max_pair": float(off6.max()), "mean_pair": float(off6.mean()), "matrix": M6.tolist(),
        "note": "greedy farthest-point, meaning ignored; class order is arbitrary"}}
    for K in LADDER:
        sub = N[chosen[:K]] @ N[chosen[:K]].T
        off = sub[np.triu_indices(K, 1)] if K > 1 else np.array([0.0])
        new[f"geometric{K}"] = {"tokens": [{"id": t, "token": n} for t, n in zip(chosen[:K], names[:K])],
                                "max_pair": float(off.max()), "mean_pair": float(off.mean()),
                                "note": f"first {K} of the nested max-min greedy order"}
        print(f"geometric{K}: max pair {off.max():.3f}")

    if args.check:
        have = cfg.get("profiles", {}).get("geometric64", {}).get("tokens", [])
        ok = [t["id"] for t in have] == chosen
        print("check geometric ladder:", "OK" if ok else "MISMATCH")
        raise SystemExit(0 if ok else 1)
    cfg.setdefault("profiles", {}).update(new)
    out.write_text(json.dumps(cfg, indent=1))
    print("wrote geometric profiles ->", out)


if __name__ == "__main__":
    main()
