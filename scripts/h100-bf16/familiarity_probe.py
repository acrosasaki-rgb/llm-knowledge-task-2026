"""Prompt-point entity-familiarity probe (activation dispersion).

For each city subject, run one forward pass of gemma-3-27b-pt on a fixed
prompt that never asks for the answer ("Who is {name}?"), take the hidden
state at the last prompt token for every layer, and compute two dispersion
statistics over the hidden dimension (99th-percentile winsorised):

    p_i = a_i^2 / sum_j a_j^2 ;  IPR = sum p_i^2 ;  S = -sum p_i log p_i

The same is done for K synthetic controls per subject (first/last names
cycled among the subjects of the same file, so tokens are real but the
entity is almost surely unknown). Output: one JSON row per subject with the
per-layer stats for the real name and each control. Thresholds are NOT
decided here; the gate is defined offline on train and checked on val.

Usage (host, venv with torch/transformers):
  python familiarity_probe.py --model /home/ubuntu/hfmodels/gemma-3-27b-pt \
      --rows train.jsonl --out train-fam.jsonl [--controls 3] [--gpu 0]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re

import torch


def load_city_subjects(path):
    subs = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["Relation"] == "personHasCityOfDeath":
            subs.append(r["SubjectEntity"])
    return subs


def strip_qualifier(name):
    return re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()


def make_controls(subjects, k):
    """Cycle first/last names among subjects: control j of subject i takes the
    first name of subject i and the rest of the name of subject (i + off_j)."""
    parts = []
    for s in subjects:
        base = strip_qualifier(s).split()
        parts.append((base[0], " ".join(base[1:])) if len(base) >= 2 else (base[0], ""))
    n = len(subjects)
    controls = []
    for i, s in enumerate(subjects):
        cs = []
        for j in range(k):
            off = int.from_bytes(hashlib.sha256(f"{s}|{j}".encode()).digest()[:2], "big") % (n - 1) + 1
            first = parts[i][0]
            last = parts[(i + off) % n][1] or parts[(i + off + 1) % n][1]
            cs.append(f"{first} {last}".strip())
        controls.append(cs)
    return controls


_MLP_ACTS = []


def _install_hooks(model):
    layers = model.model.language_model.layers if hasattr(model.model, "language_model") else model.model.layers
    for layer in layers:
        def hook(mod, inp, out, _l=layer):
            _MLP_ACTS.append(inp[0][0, -1].detach())
        layer.mlp.down_proj.register_forward_hook(hook)


@torch.no_grad()
def dispersion(model, tok, text, device):
    enc = tok(text, return_tensors="pt").to(device)
    _MLP_ACTS.clear()
    model(**enc)
    stats = []
    for h in _MLP_ACTS:  # MLP intermediate activation (input of down_proj) at the last prompt token
        a = h.float()
        # winsorise at 99th percentile of |a|
        q = torch.quantile(a.abs(), 0.99)
        a = a.clamp(-q, q)
        p = a.pow(2)
        p = p / p.sum()
        ipr = float((p * p).sum())
        ent = float(-(p * (p + 1e-12).log()).sum())
        stats.append((ipr, ent))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--controls", type=int, default=3)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--template", default="Who is {s}?")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = f"cuda:{a.gpu}"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map={"": device})
    model.eval()
    _install_hooks(model)

    subjects = load_city_subjects(a.rows)
    controls = make_controls(subjects, a.controls)
    with open(a.out, "w", encoding="utf-8") as f:
        for i, (s, cs) in enumerate(zip(subjects, controls)):
            row = {"SubjectEntity": s,
                   "real": dispersion(model, tok, a.template.format(s=strip_qualifier(s)), device),
                   "controls": {c: dispersion(model, tok, a.template.format(s=c), device) for c in cs}}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (i + 1) % 20 == 0:
                print(f"{i + 1}/{len(subjects)}", flush=True)
    print("familiarity probe completed")


if __name__ == "__main__":
    main()
