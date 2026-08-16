"""DoLa-dynamic vs plain greedy on the city occ register (gemma-3-27b-pt, HF).

Generates the continuation of the production occ register for every city row
of --rows twice: plain greedy and DoLa (dola_layers='high', dynamic premature
layer selection by JSD). Writes both raw continuations per subject.
"""
from __future__ import annotations

import argparse
import json
import re

import torch

OCC = (
    "Name: James Gandolfini\nOccupation: actor\nCity of death: Rome\n"
    "Name: Ada Lovelace\nOccupation: mathematician\nCity of death: London\n"
    "Name: Paul McCartney\nOccupation: musician\nCity of death: (still alive)\n"
    "Name: {s}\nOccupation:"
)


def parse(text):
    text = text.strip()
    if "alive" in text.lower():
        return []
    m = re.search(r"City of death:\s*([^\n,.(]+)", text)
    if m:
        city = m.group(1).strip()
        return [city] if city and "alive" not in city.lower() else []
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = f"cuda:{a.gpu}"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map={"": device})
    model.eval()
    subjects = [json.loads(l)["SubjectEntity"] for l in open(a.rows, encoding="utf-8")
                if json.loads(l)["Relation"] == "personHasCityOfDeath"]
    stop_ids = None
    with open(a.out, "w", encoding="utf-8") as f:
        for i, s in enumerate(subjects):
            enc = tok(OCC.format(s=s), return_tensors="pt").to(device)
            n_in = enc["input_ids"].shape[1]
            outs = {}
            for mode in ("greedy", "dola"):
                kw = dict(max_new_tokens=32, do_sample=False)
                if mode == "dola":
                    kw.update(dola_layers="high", repetition_penalty=1.2, trust_remote_code=True, output_hidden_states=True, return_dict_in_generate=False)
                with torch.no_grad():
                    g = model.generate(**enc, **kw)
                text = tok.decode(g[0, n_in:], skip_special_tokens=True)
                text = text.split("\nName:")[0]
                outs[mode] = {"raw": text, "parsed": parse(text)}
            f.write(json.dumps({"SubjectEntity": s, **outs}, ensure_ascii=False) + "\n")
            if (i + 1) % 20 == 0:
                print(f"{i + 1}/{len(subjects)}", flush=True)
    print("dola probe completed")


if __name__ == "__main__":
    main()
