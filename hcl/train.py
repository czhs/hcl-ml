#!/usr/bin/env python3
"""Train one condition on the subject-wise splits; select on val; never touch test.

Writes <results>/<tag>_s<seed>/{metrics.jsonl, summary.json, best.pt, norm_stats.npz}
where <tag> encodes the variant and any ablation switches (see tag_for()).

Examples
  python -m hcl.train --variant direct  --seed 0
  python -m hcl.train --variant simplex --seed 0
  python -m hcl.train --variant simplex --anchors geometric --seed 1
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from . import model as M
from .data import load_split, save_norm_stats
from .paths import RESULTS


def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def tag_for(args):
    tag = args.variant if args.anchors == "balanced" else f"{args.variant}-{args.anchors}"
    if args.learn_anchors:
        tag += "-learnA"
    if args.anchor_init == "random":
        tag += "-randinit"
    if args.n_anchors:
        tag += f"-K{args.n_anchors}"
    if args.gate:
        tag += "-gate"
    if args.lm_init == "random":
        tag += "-randLM"
    if args.no_prompt:
        tag += "-noprompt"
    if args.prompt_style != "task":
        tag += f"-{args.prompt_style}prompt"
    return tag


@torch.no_grad()
def evaluate(model, X, y, bs, device):
    model.eval()
    preds, ents, wargs = [], [], []
    for i in range(0, len(X), bs):
        logits, extras = model(X[i:i + bs].to(device))
        preds.append(logits.argmax(-1).cpu())
        if "entropy" in extras:
            ents.append(extras["entropy"].cpu())
            wargs.append(extras["w"].argmax(-1).cpu())
    preds = torch.cat(preds)
    f1 = f1_score(y.numpy(), preds.numpy(), average="macro")
    diag = {}
    if ents:   # simplex diagnostics: is w a soft code or a hidden classifier?
        diag = {"w_entropy_mean": float(torch.cat(ents).mean()),
                "w_argmax_agrees_label": float((torch.cat(wargs) == y).float().mean())}
    return f1, preds, diag


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, choices=["direct", "simplex", "free", "bottleneck"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--anchors", default="balanced", help="profile in configs/label_tokens.json")
    ap.add_argument("--learn-anchors", action="store_true")
    ap.add_argument("--anchor-init", default="tokens", choices=["tokens", "random"])
    ap.add_argument("--n-anchors", type=int, default=0)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--lm-init", default="pretrained", choices=["pretrained", "random"])
    ap.add_argument("--no-prompt", action="store_true")
    ap.add_argument("--prompt-style", default="task", choices=["task", "unrelated"])
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--outdir", default=None, help="override <results>/<tag>_s<seed>")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--nondeterministic", action="store_true", help="allow cuDNN autotuning")
    args = ap.parse_args()

    set_seed(args.seed, deterministic=not args.nondeterministic)
    device = args.device
    tag = tag_for(args)
    out = Path(args.outdir).expanduser() if args.outdir else Path(args.results).expanduser() / f"{tag}_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    Xtr, ytr, stats = load_split("train")
    Xva, yva, _ = load_split("val", stats)
    save_norm_stats(out, stats)

    model = M.build(args.variant, tau=args.tau, anchors_profile=args.anchors,
                    learn_anchors=args.learn_anchors, anchor_init=args.anchor_init,
                    n_anchors=args.n_anchors, gate=args.gate, lm_init=args.lm_init,
                    no_prompt=args.no_prompt, prompt_style=args.prompt_style).to(device)
    n_trainable = M.trainable_params(model)
    print(f"run={out.name} trainable={n_trainable:,} "
          f"(budget 10,000,000: {'OK' if n_trainable <= 10_000_000 else 'OVER'})")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    ckpt_args = {**vars(args), "anchors_profile": args.anchors}

    best_f1, best_epoch = -1.0, -1
    with open(out / "metrics.jsonl", "w") as log:
        for ep in range(args.epochs):
            model.train()
            if hasattr(model, "lm"):
                model.lm.eval()          # the frozen LM never leaves eval mode
            perm = torch.randperm(len(Xtr))
            tot, t0 = 0.0, time.time()
            for i in range(0, len(perm), args.bs):
                idx = perm[i:i + args.bs]
                x, yb = Xtr[idx].to(device), ytr[idx].to(device)
                logits, _ = model(x)
                loss = F.cross_entropy(logits, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += loss.item() * len(idx)
            sched.step()
            f1, _, diag = evaluate(model, Xva, yva, 512, device)
            rec = {"epoch": ep, "train_loss": tot / len(Xtr), "val_macro_f1": round(f1, 4),
                   "sec": round(time.time() - t0, 1), **diag}
            print(rec)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if f1 > best_f1:
                best_f1, best_epoch = f1, ep
                M.save_checkpoint(model, out / "best.pt", ckpt_args, f1)

    summary = {"run": out.name, "variant": args.variant, "tag": tag, "seed": args.seed,
               "trainable_params": n_trainable, "best_val_macro_f1": round(best_f1, 4),
               "best_epoch": best_epoch, "epochs": args.epochs, "lr": args.lr, "tau": args.tau,
               "device": torch.cuda.get_device_name(0) if device == "cuda" else device}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("SUMMARY", json.dumps(summary))


if __name__ == "__main__":
    main()
