"""Models for the Sensor Context Encoder Challenge.

Shared trunk (identical in every condition, so the comparison isolates the
interface rather than the feature extractor):
  SensorEncoder : 1-D CNN over (B, 128, 9) -> 256-d feature (~160k params)

Conditions
  direct     : feats -> Linear(256, 6)                      [challenge baseline]
  simplex    : feats -> Linear(256, K) -> softmax(z/tau) = w on the simplex,
               e = w @ A with A = K frozen anchor-token embeddings (K = 6,
               profile "balanced": walking, climb, descending, seated, upright,
               lying). e is the ONE continuous context embedding.      [deliverable]
  free       : feats -> Linear(256, 960), RMS-rescaled to the mean anchor norm
               (same radius as the simplex output)          [supplementary ablation]
  bottleneck : simplex code WITHOUT the LM: feats -> softmax w -> Linear(6, 6)
                                                             [supplementary ablation]

LM path (simplex / free): e replaces <SENSOR> inside the mandated prompt's
input-embedding sequence; frozen SmolLM2-360M-Instruct (bf16, embedding table
included) runs forward; a trainable Linear(960, 6) reads the final hidden state
at the last position (after "Activity:"). Gradients flow through the frozen LM
into the projector and encoder.

Ablation switches (all used by scripts/run_ablations.sh):
  anchors_profile  which rows of configs/label_tokens.json form A
  learn_anchors    make A a trainable parameter
  anchor_init      "random": random directions at token-typical norm
  n_anchors        K for random-init anchors
  gate             sigmoid gates instead of softmax (magnitude code)
  lm_init          "random": randomly initialised frozen transformer, pretrained
                   embedding table kept (reservoir control)
  no_prompt        the LM sees only e
  prompt_style     "unrelated": a recipe prompt of similar length
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .paths import LABEL_TOKENS, LM_ID

PROMPT_BEFORE = ("Classify the activity as walking, walking upstairs, walking "
                 "downstairs, sitting, standing, or laying.\n\nSensor context: ")
PROMPT_AFTER = "\nActivity:"
# prompt-content control: comparable length/structure, zero activity semantics
UNRELATED_BEFORE = ("The recipe calls for two cups of flour, one egg, a pinch "
                    "of salt, and a generous splash of cold milk.\n\nKitchen note: ")
UNRELATED_AFTER = "\nResult:"


class SensorEncoder(nn.Module):
    def __init__(self, feat_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(9, 64, 5, stride=2, padding=2), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 128, 5, stride=2, padding=2), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.GELU(),
        )
        self.proj = nn.Sequential(nn.Linear(256, feat_dim), nn.GELU())

    def forward(self, x):                                  # x: (B, 128, 9)
        h = self.net(x.transpose(1, 2))                    # (B, 128, 32)
        h = torch.cat([h.mean(-1), h.amax(-1)], dim=-1)    # (B, 256)
        return self.proj(h)


class DirectClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SensorEncoder()
        self.head = nn.Linear(256, 6)

    def forward(self, x):
        return self.head(self.encoder(x)), {}


class BottleneckClassifier(nn.Module):
    """Simplex bottleneck WITHOUT the LM: trunk -> 6 logits -> softmax w ->
    Linear(6, 6). Isolates the bottleneck's cost from the LM's contribution."""

    def __init__(self, tau=1.0):
        super().__init__()
        self.encoder = SensorEncoder()
        self.projector = nn.Linear(256, 6)
        self.head = nn.Linear(6, 6)
        self.tau = tau

    def forward(self, x):
        w = F.softmax(self.projector(self.encoder(x)) / self.tau, dim=-1)
        ent = -(w * (w + 1e-9).log()).sum(-1)
        return self.head(w), {"w": w, "entropy": ent}


def load_lm(lm_init="pretrained"):
    """Frozen SmolLM2 in bf16 (+ its tokenizer)."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(LM_ID)
    lm = AutoModelForCausalLM.from_pretrained(LM_ID, dtype=torch.bfloat16)
    if lm_init == "random":
        # reservoir control: random frozen transformer, but keep the PRETRAINED
        # embedding table so prompt and anchors are identical — only the lens
        # changes. Seeded by the training seed (set before build()).
        rnd = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(LM_ID)).to(torch.bfloat16)
        rnd.get_input_embeddings().weight.data.copy_(lm.get_input_embeddings().weight.data)
        lm = rnd
    lm.requires_grad_(False).eval()
    lm.config.output_hidden_states = True
    return lm, tok


def anchor_entries(profile):
    cfg = json.loads(Path(LABEL_TOKENS).read_text())
    prof = cfg["profiles"][profile]["tokens"]
    if isinstance(prof, dict):          # class-keyed profiles (canonical/balanced/...)
        return [prof[c] for c in cfg["matrix"]["classes"]]
    return prof                         # list profiles (geometricK ladders)


class LMContextClassifier(nn.Module):
    """Frozen LM reads one context embedding; trainable head on the final hidden state."""

    def __init__(self, variant, tau=1.0, anchors_profile="balanced",
                 learn_anchors=False, anchor_init="tokens", n_anchors=0,
                 gate=False, lm_init="pretrained", no_prompt=False,
                 prompt_style="task"):
        super().__init__()
        assert variant in ("simplex", "free")
        assert anchor_init in ("tokens", "random")
        assert lm_init in ("pretrained", "random")
        self.variant, self.tau = variant, tau
        self.anchors_profile, self.learn_anchors, self.anchor_init = anchors_profile, learn_anchors, anchor_init
        self.gate, self.lm_init, self.no_prompt, self.prompt_style = gate, lm_init, no_prompt, prompt_style
        self.encoder = SensorEncoder()

        lm, tok = load_lm(lm_init)
        self.lm = lm
        d = lm.config.hidden_size
        emb = lm.get_input_embeddings().weight.detach()

        before, after = PROMPT_BEFORE, PROMPT_AFTER
        if prompt_style == "unrelated":
            before, after = UNRELATED_BEFORE, UNRELATED_AFTER
        ids_b = tok(before).input_ids
        ids_a = tok(after, add_special_tokens=False).input_ids
        if no_prompt:                    # the LM sees ONLY the context embedding
            ids_b, ids_a = ids_b[:0], ids_a[:0]
        self.register_buffer("emb_before", emb[ids_b].clone())   # (Tb, d)
        self.register_buffer("emb_after", emb[ids_a].clone())    # (Ta, d)

        entries = anchor_entries(anchors_profile)
        self.anchor_names = [e_["token"] for e_ in entries]
        A = emb[[e_["id"] for e_ in entries]].clone().float()      # (K, d)
        if anchor_init == "random":
            # random directions at token-typical norms — a fair off-manifold init
            K = n_anchors or len(entries)
            R = torch.randn(K, A.shape[1])
            A = R / R.norm(dim=-1, keepdim=True) * A.norm(dim=-1).mean()
            self.anchor_names = [f" rand{k}" for k in range(K)]
        self.n_anchors = A.shape[0]
        if learn_anchors:
            self.anchors = nn.Parameter(A)
        else:
            self.register_buffer("anchors", A)

        if variant == "simplex":
            self.projector = nn.Linear(256, self.n_anchors)
        else:
            self.projector = nn.Linear(256, d)
            self.register_buffer("target_norm", self.anchors.norm(dim=-1).mean())
        self.head = nn.Linear(d, 6)

    def context_embedding(self, feats):
        """feats (B,256) -> (e (B,960), extras). The ONE context embedding."""
        if self.variant == "simplex":
            z = self.projector(feats) / self.tau
            w = torch.sigmoid(z) if self.gate else F.softmax(z, dim=-1)
            e = w @ self.anchors
            ent = -(w * (w + 1e-9).log()).sum(-1)
            return e, {"w": w, "entropy": ent}
        e = self.projector(feats)
        e = e * (self.target_norm / (e.norm(dim=-1, keepdim=True) + 1e-6))
        return e, {}

    def forward(self, x, context_override=None):
        B = x.shape[0]
        if context_override is None:
            e, extras = self.context_embedding(self.encoder(x))
        else:                            # shuffled-embedding control
            e, extras = context_override, {}
        seq = torch.cat([
            self.emb_before.unsqueeze(0).expand(B, -1, -1),
            e.unsqueeze(1).to(self.emb_before.dtype),      # <SENSOR> slot
            self.emb_after.unsqueeze(0).expand(B, -1, -1),
        ], dim=1)
        out = self.lm(inputs_embeds=seq, use_cache=False, output_hidden_states=True)
        h_last = out.hidden_states[-1][:, -1]               # after "Activity:"
        return self.head(h_last.float()), extras


BUILD_KEYS = ("tau", "anchors_profile", "learn_anchors", "anchor_init",
              "n_anchors", "gate", "lm_init", "no_prompt", "prompt_style")


def build(variant, tau=1.0, anchors_profile="balanced", learn_anchors=False,
          anchor_init="tokens", n_anchors=0, gate=False,
          lm_init="pretrained", no_prompt=False, prompt_style="task"):
    if variant == "direct":
        return DirectClassifier()
    if variant == "bottleneck":
        return BottleneckClassifier(tau=tau)
    return LMContextClassifier(variant, tau=tau, anchors_profile=anchors_profile,
                               learn_anchors=learn_anchors, anchor_init=anchor_init,
                               n_anchors=n_anchors, gate=gate, lm_init=lm_init,
                               no_prompt=no_prompt, prompt_style=prompt_style)


def build_from_args(args):
    """Rebuild the model a run was trained with (args = train.py namespace dict)."""
    kw = {k: args[k] for k in BUILD_KEYS if k in args}
    kw.setdefault("anchors_profile", args.get("anchors", "balanced"))
    return build(args["variant"], **kw)


def trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, path, args, val_f1):
    """Only the trainable parts + small buffers; the frozen LM is never saved."""
    state = {k: v for k, v in model.state_dict().items() if not k.startswith("lm.")}
    torch.save({"state": state, "args": args, "val_f1": val_f1}, path)


def load_run(run_dir, device="cuda"):
    """Load a finished run directory -> (model, norm_stats, val_f1, args)."""
    run_dir = Path(run_dir)
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    args = dict(ck["args"])
    model = build_from_args(args).to(device)
    missing, unexpected = model.load_state_dict(ck["state"], strict=False)
    assert not unexpected and all(k.startswith("lm.") for k in missing), (missing, unexpected)
    model.eval()
    s = np.load(run_dir / "norm_stats.npz")
    return model, (s["mean"], s["std"]), float(ck["val_f1"]), args
