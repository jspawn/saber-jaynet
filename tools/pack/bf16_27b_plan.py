#!/usr/bin/env python3
"""Re-source a ``saber-packplan`` 27B plan onto a BF16 Qwen3.8 checkpoint.

The standard plan names the production checkpoint's NVFP4 MLP tensors and
per-channel FP8 attention/GDN tensors. A BF16 checkpoint stores the same decoder
linears as uncompressed BF16 matrices. This tool preserves every record name,
layout, shape and shard kind while changing only their checkpoint sources.

    saber-packplan --tp 2 --head bf16 --checkpoint MODEL --config MODEL/config.json > plan.json
    tools/pack/bf16_27b_plan.py --model MODEL --plan plan.json > plan-bf16.json
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

from quark27b_plan import header, strip


def rewrite(plan: dict, hdr: dict) -> dict:
    consumed = set()
    for record in plan["records"]:
        for source in record["sources"]:
            name = source["weight"]
            quant = source["quant"]
            if quant == "unit_bf16":
                continue
            if name.endswith((".self_attn.k_scale", ".self_attn.v_scale")) and name not in hdr:
                source.update(quant="unit_bf16", scale=None, global_scale=None, input_scale=None)
                continue
            if quant in ("nvfp4", "nvfp4_modelopt", "fp8_channel", "bf16"):
                name = strip(name) + ".weight"
                if hdr.get(name, ("",))[0] != "BF16":
                    raise SystemExit(f"{record['name']}: {name} is {hdr.get(name)}, want BF16")
                source.update(weight=name, quant="bf16", scale=None, global_scale=None,
                              input_scale=None)
            elif quant != "as_stored":
                raise SystemExit(f"{record['name']}: unsupported source quant {quant}")
            if name not in hdr:
                raise SystemExit(f"{record['name']}: {name} is not in the checkpoint")
            consumed.add(name)
    unread_decoder = sorted(
        name for name in hdr
        if name.startswith("model.language_model.layers.") and name not in consumed
    )
    if unread_decoder:
        raise SystemExit(f"decoder tensors read by no record: {unread_decoder[:4]}")
    plan["sources_rewritten_by"] = "tools/pack/bf16_27b_plan.py (Q27: BF16 checkpoint)"
    return plan


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--plan", required=True, help="saber-packplan's output")
    args = parser.parse_args(argv)
    with open(args.plan) as handle:
        plan = json.load(handle)
    if plan.get("producer") != "saber-packplan":
        raise SystemExit("not a saber-packplan plan")
    out = rewrite(plan, header(args.model))
    count = sum(1 for record in out["records"] for source in record["sources"]
                if source["quant"] == "bf16")
    print(f"bf16_27b_plan: {count} BF16 sources, head {out['head']}", file=sys.stderr)
    json.dump(out, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
