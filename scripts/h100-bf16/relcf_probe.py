"""Relation-counterfactual specificity: logP(city | "{s} died in") vs
max over {"{s} was born in", "{s} lived in", "{s} is from"}, teacher-forced
for the currently predicted city only."""
from __future__ import annotations

import argparse
import json
import re

import torch

TPL = {"died": "{s} died in", "born": "{s} was born in", "lived": "{s} lived in", "from": "{s} is from"}


@torch.no_grad()
def lp(model, tok, prefix, suffix, device):
    ids_p = tok(prefix, return_tensors="pt").input_ids.to(device)
    ids_f = tok(prefix + suffix, return_tensors="pt").input_ids.to(device)
    out = model(ids_f)
    logp = torch.log_softmax(out.logits[0, :-1].float(), dim=-1)
    tgt = ids_f[0, 1:]
    v = logp[torch.arange(tgt.shape[0]), tgt][ids_p.shape[1] - 1:]
    return float(v.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = f"cuda:{a.gpu}"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map={"": device})
    model.eval()
    with open(a.out, "w", encoding="utf-8") as f:
        for l in open(a.preds, encoding="utf-8"):
            r = json.loads(l)
            if not r["ObjectEntities"]:
                continue
            s = re.sub(r"\s*\([^)]*\)\s*$", "", r["SubjectEntity"]).strip()
            c = r["ObjectEntities"][0]
            scores = {k: lp(model, tok, t.format(s=s), " " + c, device) for k, t in TPL.items()}
            f.write(json.dumps({"SubjectEntity": r["SubjectEntity"], "city": c, "scores": scores}, ensure_ascii=False) + "\n")
    print("relcf probe completed")


if __name__ == "__main__":
    main()
