"""Name-likelihood familiarity probe (teacher-forced logprobs, no generation).

For each city subject: log P(name tokens | prefix) under a fixed prefix, split
into first-name and surname parts, for the real name and K synthetic controls
(names cycled among subjects). Also the entropy of the next-token distribution
right after the name under the occ register (before "Occupation:" is
completed) as a cheap prompt-point signal.
"""
from __future__ import annotations

import argparse
import json
import math
import re

import torch

from familiarity_probe import load_city_subjects, make_controls, strip_qualifier

PREFIX = "Name:"


@torch.no_grad()
def name_logprobs(model, tok, name, device):
    ids_p = tok(PREFIX, return_tensors="pt").input_ids.to(device)
    ids_full = tok(PREFIX + " " + name, return_tensors="pt").input_ids.to(device)
    out = model(ids_full)
    logp = torch.log_softmax(out.logits[0, :-1].float(), dim=-1)
    tgt = ids_full[0, 1:]
    tok_lp = logp[torch.arange(tgt.shape[0]), tgt]
    n_p = ids_p.shape[1]
    name_lp = tok_lp[n_p - 1:]  # tokens belonging to the name
    # first token of the name vs rest
    first = float(name_lp[0]) if name_lp.numel() else 0.0
    rest = float(name_lp[1:].sum()) if name_lp.numel() > 1 else 0.0
    # next-token entropy after the name (before newline)
    p = torch.softmax(out.logits[0, -1].float(), dim=-1)
    ent = float(-(p * (p + 1e-12).log()).sum())
    return {"n_tok": int(name_lp.numel()), "sum": float(name_lp.sum()), "mean": float(name_lp.mean()),
            "first": first, "rest": rest, "next_entropy": ent}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--controls", type=int, default=3)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = f"cuda:{a.gpu}"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map={"": device})
    model.eval()
    subjects = load_city_subjects(a.rows)
    controls = make_controls(subjects, a.controls)
    with open(a.out, "w", encoding="utf-8") as f:
        for s, cs in zip(subjects, controls):
            row = {"SubjectEntity": s, "real": name_logprobs(model, tok, strip_qualifier(s), device),
                   "controls": {c: name_logprobs(model, tok, c, device) for c in cs}}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("name likelihood probe completed")


if __name__ == "__main__":
    main()
