"""Subject-shuffle PMI specificity for personHasCityOfDeath.

For every predicted city c_i (from --preds) and every subject s_j in --rows,
compute the teacher-forced log-probability of " {c_i}\n" after the fixed
city register prompt for s_j. Sharded over cities (--shard k/N).

Output rows: {"city": c, "subject": s, "lp": float}
"""
from __future__ import annotations

import argparse
import json
import re

import torch

REG = (
    "Name: James Gandolfini\nCity of death: Rome\n"
    "Name: Ada Lovelace\nCity of death: London\n"
    "Name: Paul McCartney\nCity of death: (still alive)\n"
    "Name: {s}\nCity of death:"
)


@torch.no_grad()
def batch_suffix_logprob(model, tok, prefixes, suffix, device, bs=16):
    """Sum log-prob of `suffix` tokens after each prefix (left padding)."""
    out = []
    tok.padding_side = "left"
    for i in range(0, len(prefixes), bs):
        chunk = prefixes[i:i + bs]
        full = [p + suffix for p in chunk]
        enc = tok(full, return_tensors="pt", padding=True).to(device)
        n_suf = [len(tok(p + suffix).input_ids) - len(tok(p).input_ids) for p in chunk]
        logits = model(**enc).logits
        logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        tgt = enc.input_ids[:, 1:]
        lp = logp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)  # [b, T-1]
        for r in range(len(chunk)):
            k = n_suf[r]
            out.append(float(lp[r, -k:].sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rows", required=True, help="rows file (city subjects)")
    ap.add_argument("--preds", required=True, help="predictions jsonl (SubjectEntity, ObjectEntities)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    k, N = (int(x) for x in a.shard.split("/"))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = f"cuda:{a.gpu}"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map={"": device})
    model.eval()
    subjects = [json.loads(l)["SubjectEntity"] for l in open(a.rows, encoding="utf-8")
                if json.loads(l)["Relation"] == "personHasCityOfDeath"]
    cities = []
    for l in open(a.preds, encoding="utf-8"):
        r = json.loads(l)
        if r.get("Relation", "personHasCityOfDeath") != "personHasCityOfDeath":
            continue
        for c in r["ObjectEntities"][:1]:
            if c not in cities:
                cities.append(c)
    my = [c for i, c in enumerate(cities) if i % N == k]
    prefixes = [REG.format(s=re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()) for s in subjects]
    with open(a.out, "w", encoding="utf-8") as f:
        for i, c in enumerate(my):
            lps = batch_suffix_logprob(model, tok, prefixes, " " + c + "\n", device)
            for s, lp in zip(subjects, lps):
                f.write(json.dumps({"city": c, "subject": s, "lp": lp}, ensure_ascii=False) + "\n")
            if (i + 1) % 10 == 0:
                print(f"{i + 1}/{len(my)} cities", flush=True)
    print("pmi probe completed")


if __name__ == "__main__":
    main()
