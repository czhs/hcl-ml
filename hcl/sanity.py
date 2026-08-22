#!/usr/bin/env python3
"""Environment sanity check: GPU, frozen LM, and trainable parameter counts."""
import torch

from . import model as M
from .paths import LM_ID


def main():
    print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available(),
          "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("LM:", LM_ID)
    direct = M.build("direct").to(device)
    simplex = M.build("simplex").to(device)
    lm_params = sum(p.numel() for p in simplex.lm.parameters())
    emb = simplex.lm.get_input_embeddings()
    print(f"LM params {lm_params/1e6:.1f}M (frozen: {not any(p.requires_grad for p in simplex.lm.parameters())}) "
          f"hidden {emb.embedding_dim} vocab {emb.num_embeddings}")
    print(f"anchors ({simplex.anchors_profile}): {[n.strip() for n in simplex.anchor_names]}")
    print(f"prompt tokens before/after <SENSOR>: {simplex.emb_before.shape[0]}/{simplex.emb_after.shape[0]}")
    x = torch.randn(4, 128, 9, device=device)
    with torch.no_grad():
        for name, m in [("direct", direct), ("simplex", simplex)]:
            m.eval()
            logits, _ = m(x)
            print(f"{name:8s} trainable {M.trainable_params(m):>9,}  forward OK {tuple(logits.shape)}")
    print("SANITY OK")


if __name__ == "__main__":
    main()
