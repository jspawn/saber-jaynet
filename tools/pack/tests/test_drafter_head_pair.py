#!/usr/bin/env python3
"""Host-only tests for pairing DFlash2 target embeddings and heads."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import build_pack as bp  # noqa: E402
import drafter_head_pair as dp  # noqa: E402


def bf16_bytes(seed, shape):
    rng = np.random.default_rng(seed)
    values = rng.standard_normal(shape).astype(np.float32)
    return bp.f32_to_bf16(values).tobytes()


def write_pack(root, rows, tier):
    os.makedirs(root)
    records = [bp.Record(name, dtype, shape, "Replicated", layout)
               for name, dtype, shape, layout, _ in rows]
    plan = bp.plan_rank(records, bp.DEFAULT_CHUNK_BYTES)
    pool = bp.HashPool(1, 1 << 20, bp.DEFAULT_CHUNK_BYTES)
    writer = bp.RankWriter(os.path.join(root, "rank0.pack"), plan, pool)
    for record, row in zip(records, rows):
        writer.begin(record)
        writer.write(row[4])
        writer.end()
    pool.close()
    writer.finish(1, 0, b"P" * 32, bytes.fromhex(tier), 1)
    rank_records = [{
        "name": r.name, "dtype": r.dtype, "shape": r.shape,
        "shard_kind": "Replicated", "layout_id": r.layout_id,
        "layout": bp.LAYOUT_NAMES[r.layout_id], "data_off": r.data_off,
        "length": r.length, "sha256": r.sha256.hex(), "sources": ["source:weight"],
    } for r in records]
    manifest = {
        "magic": "SBRPACK1", "format_version": 2, "header_bytes": 256,
        "record_bytes": 160, "align": 4096, "tp_size": 1,
        "tier": {"id": tier, "descriptor": {}},
        "plan": {"hash": "50" * 32, "descriptor": {}}, "shard_policy": [],
        "ranks": [{"rank": 0, "file": "rank0.pack", "file_bytes": plan.file_bytes,
                   "data_off": plan.data_off, "record_count": len(records),
                   "records": rank_records}],
    }
    with open(os.path.join(root, "manifest.json"), "w") as handle:
        json.dump(manifest, handle)


def record_bytes(root, name):
    manifest = dp.load(root)
    record = next(r for r in dp.recs_of(manifest, 0) if r["name"] == name)
    return b"".join(dp.stream(os.path.join(root, "rank0.pack"),
                             record["data_off"], record["length"]))


class PairEmbedding(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.drafter = os.path.join(self.temp.name, "drafter")
        self.target = os.path.join(self.temp.name, "target")
        self.source_head = bf16_bytes(1, (2, 128))
        self.target_head = bf16_bytes(2, (2, 128))
        self.source_embed = bf16_bytes(3, (2, 128))
        self.target_embed = bf16_bytes(4, (2, 128))
        int2 = dp.int2_rows(np.frombuffer(self.source_head, np.uint16).reshape(2, 128))
        rows = [
            (dp.RERANK, "BF16", [2, 128], 14, self.source_head),
            (dp.INT2[0], "U8", [2, 32], 9, int2[0].tobytes()),
            (dp.INT2[1], "BF16", [2, 1], 10, int2[1].tobytes()),
            (dp.INT2[2], "BF16", [2, 1], 11, int2[2].tobytes()),
            (dp.EMBED, "BF16", [2, 128], 0, self.source_embed),
            ("layer.0.norm", "BF16", [2], 0, bf16_bytes(5, (2,))),
            ("large.record", "U8", [bp.DEFAULT_CHUNK_BYTES + 1], 0,
             bytes(bp.DEFAULT_CHUNK_BYTES + 1)),
        ]
        write_pack(self.drafter, rows, "11" * 32)
        write_pack(self.target, [
            (dp.TARGET_HEAD, "BF16", [2, 128], 14, self.target_head),
            (dp.TARGET_EMBED, "BF16", [2, 128], 0, self.target_embed),
        ], "22" * 32)

    def tearDown(self):
        self.temp.cleanup()

    def run_pair(self, out, pair_embedding):
        argv = ["drafter_head_pair.py", "--drafter-pack", self.drafter,
                "--target-pack", self.target, "--out", out, "--check-int2"]
        if pair_embedding:
            argv.append("--pair-embedding")
        with mock.patch.object(sys, "argv", argv):
            return dp.main()

    def test_different_embedding_requires_opt_in(self):
        out = os.path.join(self.temp.name, "refused")
        with self.assertRaisesRegex(SystemExit, "--pair-embedding"):
            self.run_pair(out, False)
        self.assertFalse(os.path.exists(out))

    def test_pair_embedding_copies_target_and_requantizes_head(self):
        out = os.path.join(self.temp.name, "paired")
        self.assertEqual(self.run_pair(out, True), 0)
        self.assertEqual(record_bytes(out, dp.EMBED), self.target_embed)
        self.assertEqual(record_bytes(out, dp.RERANK), self.target_head)
        self.assertEqual(record_bytes(out, "layer.0.norm"), bf16_bytes(5, (2,)))
        expected = dp.int2_of_head(os.path.join(self.target, "rank0.pack"),
                                   dp.recs_of(dp.load(self.target), 0)[0])
        for name, data in zip(dp.INT2, expected):
            self.assertEqual(record_bytes(out, name), data)
        manifest = dp.load(out)
        target_records = manifest["tier"]["descriptor"]["head_from"]["records"]
        self.assertEqual(target_records["embedding"], dp.TARGET_EMBED)
        self.assertNotEqual(manifest["tier"]["id"], "11" * 32)
        self.assertEqual(hashlib.sha256(self.target_embed).hexdigest(),
                         next(r for r in dp.recs_of(manifest, 0)
                              if r["name"] == dp.EMBED)["sha256"])


if __name__ == "__main__":
    unittest.main()
