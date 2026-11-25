#!/usr/bin/env python3

import copy
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import bf16_27b_plan as bf16  # noqa: E402


def source(name, quant="nvfp4"):
    return {"weight": name, "quant": quant, "scale": name + "_scale",
            "global_scale": name + "_global", "input_scale": name + "_input"}


class Bf16Plan(unittest.TestCase):
    def setUp(self):
        self.plan = {
            "producer": "saber-packplan",
            "head": "bf16",
            "records": [
                {"name": "layer.0.mlp.down_proj.w", "layout_id": 1,
                 "sources": [source("model.language_model.layers.0.mlp.down_proj.weight_packed")]},
                {"name": "layer.0.gdn.out_proj.w", "layout_id": 1,
                 "sources": [source("model.language_model.layers.0.linear_attn.out_proj.weight",
                                    "fp8_channel")]},
                {"name": "layer.0.gdn.in_proj_ba.w", "layout_id": 4,
                 "sources": [source("model.language_model.layers.0.linear_attn.in_proj_b.weight",
                                    "bf16")]},
                {"name": "layer.0.self_attn.k_scale", "layout_id": 0,
                 "sources": [source("model.language_model.layers.0.self_attn.k_scale", "bf16")]},
                {"name": "model.norm.weight", "layout_id": 0,
                 "sources": [source("model.norm.weight", "as_stored")]},
            ],
        }
        self.hdr = {
            "model.language_model.layers.0.mlp.down_proj.weight": ("BF16", [5120, 17408]),
            "model.language_model.layers.0.linear_attn.out_proj.weight": ("BF16", [5120, 6144]),
            "model.language_model.layers.0.linear_attn.in_proj_b.weight": ("BF16", [24, 5120]),
            "model.norm.weight": ("BF16", [5120]),
        }

    def test_rewrites_decoder_sources_and_preserves_records(self):
        original = copy.deepcopy(self.plan)
        out = bf16.rewrite(self.plan, self.hdr)
        for before, after in zip(original["records"], out["records"]):
            self.assertEqual({k: v for k, v in before.items() if k != "sources"},
                             {k: v for k, v in after.items() if k != "sources"})
        sources = [s for record in out["records"] for s in record["sources"]]
        self.assertEqual([s["quant"] for s in sources],
                         ["bf16", "bf16", "bf16", "unit_bf16", "as_stored"])
        self.assertEqual(sources[0]["weight"],
                         "model.language_model.layers.0.mlp.down_proj.weight")
        for item in sources[:3]:
            self.assertIsNone(item["scale"])
            self.assertIsNone(item["global_scale"])
            self.assertIsNone(item["input_scale"])

    def test_rejects_non_bf16_decoder_weight(self):
        name = "model.language_model.layers.0.mlp.down_proj.weight"
        self.hdr[name] = ("F8_E4M3", self.hdr[name][1])
        with self.assertRaisesRegex(SystemExit, "want BF16"):
            bf16.rewrite(self.plan, self.hdr)

    def test_rejects_missing_or_unread_decoder_weights(self):
        name = "model.language_model.layers.0.mlp.down_proj.weight"
        missing = copy.deepcopy(self.hdr)
        del missing[name]
        with self.assertRaisesRegex(SystemExit, "want BF16"):
            bf16.rewrite(copy.deepcopy(self.plan), missing)
        self.hdr["model.language_model.layers.1.mlp.down_proj.weight"] = ("BF16", [5120, 17408])
        with self.assertRaisesRegex(SystemExit, "read by no record"):
            bf16.rewrite(copy.deepcopy(self.plan), self.hdr)


if __name__ == "__main__":
    unittest.main()
