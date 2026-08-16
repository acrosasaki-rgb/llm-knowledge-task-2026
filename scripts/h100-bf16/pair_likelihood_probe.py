"""Answer-free minimal-pair likelihood probes (teacher-forced, no generation).

city (personHasCityOfDeath): living-status contrasts
    s1 = logP(" alive" | "{s} is")            - logP(" dead" | "{s} is")
    s2 = logP(" is"    | "{s}")               - logP(" was"  | "{s}")
    s3 = logP(" lives" | "{s} currently")     - logP(" died" | "{s}")   (token-avg)
    s4 = logP(" is still alive" | "{s}")      - logP(" has died" | "{s}")   (token-avg)
    s5 = logP(" is a living" | "{s}")         - logP(" was a" | "{s}")     (token-avg)
company (companyTradesAtStockExchange): listing-status contrasts
    c1 = logP(" is publicly traded" | "{s}")  - logP(" is privately held" | "{s}")
    c2 = logP(" is currently listed" | "{s}") - logP(" was delisted" | "{s}")
    c3 = logP(" is a listed company" | "{s}") - logP(" is a private company" | "{s}")
All scores are token-averaged log-probs of the suffix given the prefix.
Output: one row per subject with the raw component scores.
"""
from __future__ import annotations

import argparse
import json
import re

import torch

CITY_PAIRS = [
    ("{s} is", " alive", " dead"),
    ("{s}", " is", " was"),
    ("{s} currently", " lives", " died"),
    ("{s}", " is still alive", " has died"),
    ("{s}", " is a living", " was a"),
]
COMPANY_PAIRS = [
    ("{s}", " is publicly traded", " is privately held"),
    ("{s}", " is currently listed", " was delisted"),
    ("{s}", " is a listed company", " is a private company"),
]


def strip_qualifier(name):
    return re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()


@torch.no_grad()
def suffix_logprob(model, tok, prefix, suffix, device):
    ids_p = tok(prefix, return_tensors="pt").input_ids.to(device)
    ids_f = tok(prefix + suffix, return_tensors="pt").input_ids.to(device)
    out = model(ids_f)
    logp = torch.log_softmax(out.logits[0, :-1].float(), dim=-1)
    tgt = ids_f[0, 1:]
    lp = logp[torch.arange(tgt.shape[0]), tgt]
    n_p = ids_p.shape[1]
    suf = lp[n_p - 1:]
    return float(suf.mean()), int(suf.numel())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--relation", default="personHasCityOfDeath")
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = f"cuda:{a.gpu}"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map={"": device})
    model.eval()
    pairs = CITY_PAIRS if a.relation == "personHasCityOfDeath" else COMPANY_PAIRS
    subjects = [json.loads(l)["SubjectEntity"] for l in open(a.rows, encoding="utf-8")
                if json.loads(l)["Relation"] == a.relation]
    with open(a.out, "w", encoding="utf-8") as f:
        for i, s in enumerate(subjects):
            name = strip_qualifier(s)
            scores = []
            for tpl, pos, neg in pairs:
                pre = tpl.format(s=name)
                lp_pos, _ = suffix_logprob(model, tok, pre, pos, device)
                lp_neg, _ = suffix_logprob(model, tok, pre, neg, device)
                scores.append({"pos": lp_pos, "neg": lp_neg, "diff": lp_pos - lp_neg})
            f.write(json.dumps({"SubjectEntity": s, "scores": scores}, ensure_ascii=False) + "\n")
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{len(subjects)}", flush=True)
    print("pair likelihood probe completed")


if __name__ == "__main__":
    main()
