#!/usr/bin/env python3
"""Layer-wise mechanistic probe of a trained simplex model (val split).

Through the frozen LM (eager attention, so attention weights are returned):
  1. Linear probes per layer at the FINAL position: activity (6-way) and
     subject (4-way) cross-validated accuracy — where information appears.
  2. Attention mass from the final position to the sensor-embedding position,
     per layer (mean over heads + max head) — the routing picture.
  3. Logit lens per layer at the final position: probability mass on the six
     anchor tokens vs the prompt's own label words.

  python -m hcl.mech_probe --run simplex_s1     (reference/mech_probe.json used simplex_s1)
Writes <results>/mech_probe.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from transformers import AutoTokenizer

from . import model as M
from .data import load_split_raw
from .paths import LM_ID, RESULTS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--run", default="simplex_s0")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    results = Path(args.results).expanduser()
    device = args.device

    model, stats, _, _ = M.load_run(results / args.run, device)
    model.lm.config._attn_implementation = "eager"
    Xr, y, subj = load_split_raw("val")
    X = torch.from_numpy((Xr - stats[0]) / stats[1])

    Tb = model.emb_before.shape[0]          # sensor position index
    n_layers = model.lm.config.num_hidden_layers
    hid, attn_e = [], []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            x = X[i:i + 256].to(device)
            e, _ = model.context_embedding(model.encoder(x))
            B = x.shape[0]
            seq = torch.cat([model.emb_before.unsqueeze(0).expand(B, -1, -1),
                             e.unsqueeze(1).to(model.emb_before.dtype),
                             model.emb_after.unsqueeze(0).expand(B, -1, -1)], dim=1)
            out = model.lm(inputs_embeds=seq, use_cache=False, output_hidden_states=True, output_attentions=True)
            hid.append(torch.stack([h[:, -1].float().cpu() for h in out.hidden_states]))   # (L+1, B, d)
            attn_e.append(torch.stack([a[:, :, -1, Tb].float().cpu() for a in out.attentions]))  # (L, B, H)
    H = torch.cat(hid, dim=1).numpy()
    A = torch.cat(attn_e, dim=1).numpy()
    print(f"{H.shape[1]} windows, {n_layers} layers, sensor pos {Tb}")

    act_acc, subj_acc = [], []
    for L in range(H.shape[0]):
        act = cross_val_score(LogisticRegression(max_iter=2000), H[L], y, cv=3).mean()
        sub = cross_val_score(LogisticRegression(max_iter=2000), H[L], subj, cv=3).mean()
        act_acc.append(round(float(act), 4))
        subj_acc.append(round(float(sub), 4))
        print(f"L{L:2d} activity {act:.3f} subject {sub:.3f}")
    attn_mean = A.mean(axis=(1, 2)).round(4).tolist()
    attn_maxh = A.mean(axis=1).max(axis=1).round(4).tolist()

    # logit lens at the final position (note: hidden_states[-1] is already post-norm)
    W_U = model.lm.get_output_embeddings().weight.detach().float().to(device)
    aids = [e_["id"] for e_ in M.anchor_entries(model.anchors_profile)]
    tok = AutoTokenizer.from_pretrained(LM_ID)
    pids = sorted({tid for w_ in (" walking", " sitting", " standing", " laying")
                   for tid in tok(w_, add_special_tokens=False).input_ids})
    norm = model.lm.model.norm
    anchor_mass, prompt_mass = [], []
    with torch.no_grad():
        for L in range(H.shape[0]):
            h = norm(torch.tensor(H[L]).to(device).to(torch.bfloat16)).float()
            p = F.softmax(h @ W_U.T, dim=-1)
            anchor_mass.append(round(float(p[:, aids].sum(-1).mean()), 4))
            prompt_mass.append(round(float(p[:, pids].sum(-1).mean()), 4))
    print("logit-lens anchor mass:", anchor_mass)
    print("logit-lens prompt-word mass:", prompt_mass)

    (results / "mech_probe.json").write_text(json.dumps({
        "activity_probe_by_layer": act_acc, "subject_probe_by_layer": subj_acc,
        "attn_to_e_mean": attn_mean, "attn_to_e_maxhead": attn_maxh,
        "logitlens_anchor_mass": anchor_mass, "logitlens_promptword_mass": prompt_mass,
        "sensor_pos": int(Tb), "ckpt": args.run}, indent=1))
    print("wrote", results / "mech_probe.json")


if __name__ == "__main__":
    main()
