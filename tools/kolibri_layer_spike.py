#!/usr/bin/env python3
"""Run Kolibri layer 0 with NumPy and emit the small real-weight GPU-spike artifact."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import struct

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("saber_build_pack", REPO / "tools/pack/build_pack.py")
PACK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACK)

F32 = np.dtype("<f4")
U16 = np.dtype("<u2")


def bf16(values):
    return PACK.f32_to_bf16(np.asarray(values, dtype=F32))


def f32(values):
    return np.asarray(values, dtype=F32)


def from_bf16(values):
    return PACK.bf16_to_f32(np.asarray(values, dtype=U16))


def rmsnorm(values, weight, eps):
    x = from_bf16(values)
    inv = np.reciprocal(np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + np.float32(eps)))
    return bf16(from_bf16(bf16(x * inv)) * from_bf16(weight))


def linear(values, weight):
    x = from_bf16(values)
    w = from_bf16(weight)
    return bf16(x @ w.T)


def silu(values):
    x = from_bf16(values)
    return bf16(x / (1.0 + np.exp(-x)))


class Checkpoint:
    def __init__(self, root: Path, layer: int):
        index = json.loads((root / "model.safetensors.index.json").read_text())
        self.weight_map = index["weight_map"]
        prefix = f"model.layers.{layer}."
        names = [name for name in self.weight_map if name.startswith(prefix)]
        shards = {self.weight_map[name] for name in names}
        self.paths = sorted(root / shard for shard in shards)
        self.shards = {}
        for path in self.paths:
            with path.open("rb") as source:
                header_size = struct.unpack("<Q", source.read(8))[0]
                header = json.loads(source.read(header_size))
                data_start = 8 + header_size
            self.shards[path.name] = (header, np.memmap(path, mode="r", dtype=np.uint8, offset=data_start))

    def raw(self, name):
        header, data = self.shards[self.weight_map[name]]
        entry = header[name]
        begin, end = entry["data_offsets"]
        dtype = {"BF16": U16, "F32": F32, "F8_E4M3": np.uint8}[entry["dtype"]]
        return np.frombuffer(data[begin:end], dtype=dtype).reshape(entry["shape"])

    def dequant(self, name):
        values = self.raw(name)
        if name.endswith(".weight") and values.dtype == U16:
            return values
        if values.dtype != np.uint8:
            raise ValueError(f"{name}: expected FP8 or BF16, got {values.dtype}")
        scales = self.raw(name + "_scale_inv")
        expanded = np.repeat(np.repeat(scales, 128, axis=0), 128, axis=1)
        return bf16(PACK.E4M3[values] * expanded)


def rope_table(head_dim, theta, positions):
    table = np.zeros((len(positions), head_dim), dtype=U16)
    for row, position in enumerate(positions):
        for index in range(head_dim // 2):
            frequency = 1.0 / (theta ** (2.0 * index / head_dim))
            angle = float(position) * frequency
            table[row, index] = bf16(math.cos(angle))
            table[row, head_dim // 2 + index] = bf16(math.sin(angle))
    return table


def norm_rope(values, weight, heads, positions, table, eps, apply_rope):
    shape = values.shape
    values = values.reshape(-1)
    output = np.empty_like(values)
    for row, position in enumerate(positions):
        for head in range(heads):
            base = (row * heads + head) * 128
            normalized = rmsnorm(values[base:base + 128], weight, eps)
            if not apply_rope:
                output[base:base + 128] = normalized
                continue
            x = from_bf16(normalized).copy()
            c = from_bf16(table[position, :64])
            s = from_bf16(table[position, 64:])
            rotated = np.concatenate((x[:64] * c - x[64:] * s, x[64:] * c + x[:64] * s))
            output[base:base + 128] = bf16(rotated)
    return output.reshape(shape)


def attention(cfg, checkpoint, hidden, layer):
    rows = hidden.shape[0]
    h = cfg["hidden_size"]
    nq = cfg["num_attention_heads"]
    nkv = cfg["num_key_value_heads"]
    prefix = f"model.layers.{layer}"
    weights = {
        name: checkpoint.dequant(f"{prefix}.self_attn.{name}.weight")
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]
    }
    q_norm = checkpoint.dequant(f"{prefix}.self_attn.q_norm.weight")
    k_norm = checkpoint.dequant(f"{prefix}.self_attn.k_norm.weight")
    positions = list(range(rows))
    table = rope_table(cfg["head_dim"], cfg["rope_theta"], positions)
    normed = rmsnorm(hidden, checkpoint.dequant(f"{prefix}.input_layernorm.weight"), cfg["rms_norm_eps"])
    q = linear(normed, weights["q_proj"]).reshape(rows, nq, 128)
    k = linear(normed, weights["k_proj"]).reshape(rows, nkv, 128)
    v = linear(normed, weights["v_proj"]).reshape(rows, nkv, 128)
    apply_rope = cfg["layer_types"][layer] == "sliding_attention"
    q = norm_rope(q, q_norm, nq, positions, table, cfg["rms_norm_eps"], apply_rope).reshape(rows, nq, 128)
    k = norm_rope(k, k_norm, nkv, positions, table, cfg["rms_norm_eps"], apply_rope).reshape(rows, nkv, 128)
    k_attention = fp8_rows(k.reshape(-1, 128)).reshape(rows, nkv, 128)
    v_attention = fp8_rows(v.reshape(-1, 128)).reshape(rows, nkv, 128)
    output = np.zeros((rows, nq, 128), dtype=U16)
    for row in range(rows):
        for head in range(nq):
            kv_head = head // (nq // nkv)
            maximum = np.float32(-np.inf)
            denominator = np.float32(0.0)
            accumulated = np.zeros(128, dtype=F32)
            for key in range(row + 1):
                score = np.float32((from_bf16(q[row, head]) @ k_attention[key, kv_head]) / np.float32(math.sqrt(128)))
                next_maximum = np.maximum(maximum, score)
                rescale = np.exp(maximum - next_maximum)
                probability = np.exp(score - next_maximum)
                denominator = denominator * rescale + probability
                accumulated = accumulated * rescale + probability * v_attention[key, kv_head]
                maximum = next_maximum
            output[row, head] = bf16(accumulated / denominator)
    projected = linear(output.reshape(rows, nq * 128), weights["o_proj"])
    post = rmsnorm(projected, checkpoint.dequant(f"{prefix}.post_attn_norm.weight"), cfg["rms_norm_eps"])
    after = bf16(from_bf16(hidden) + from_bf16(post))
    return after


def mlp(values, weights):
    gate = linear(values, weights["gate_proj"])
    up = linear(values, weights["up_proj"])
    return linear(bf16(from_bf16(silu(gate)) * from_bf16(up)), weights["down_proj"])


def fp8_rows(values):
    x = from_bf16(values)
    scales = np.maximum(np.max(np.abs(x), axis=-1, keepdims=True) / np.float32(448.0),
                        np.float32(1.0 / (448.0 * 512.0))).astype(F32)
    codes = PACK.f32_to_e4m3(np.clip(x / scales, -448.0, 448.0))
    return PACK.E4M3[codes] * scales


def mlp_w4a8(values, weights):
    x = fp8_rows(values)
    gate = bf16(x @ from_bf16(weights["gate_proj"]).T)
    up = bf16(x @ from_bf16(weights["up_proj"]).T)
    activated = bf16(from_bf16(silu(gate)) * from_bf16(up))
    return bf16(fp8_rows(activated) @ from_bf16(weights["down_proj"]).T)


def topk_router(logits, bias, count):
    scores = logits + bias
    order = np.argsort(-scores, axis=-1, kind="stable")[:, :count]
    selected = np.take_along_axis(logits, order, axis=1)
    weights = (1.0 / (1.0 + np.exp(-selected))).astype(F32)
    return order.astype(np.int32), weights


def relative_rmse(actual, expected):
    actual = from_bf16(actual) if actual.dtype == U16 else f32(actual)
    expected = from_bf16(expected) if expected.dtype == U16 else f32(expected)
    return float(np.sqrt(np.sum((actual - expected) ** 2) / np.maximum(np.sum(expected ** 2), 1e-30)))


def save_safetensors(path, tensors):
    header = {}
    blobs = []
    offset = 0
    for name, values in tensors.items():
        values = np.asarray(values)
        dtype = {U16: "BF16", F32: "F32", np.dtype("int32"): "I32"}[values.dtype]
        data = values.tobytes(order="C")
        header[name] = {"dtype": dtype, "shape": list(values.shape), "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    with path.open("wb") as output:
        output.write(struct.pack("<Q", len(encoded)))
        output.write(encoded)
        for data in blobs:
            output.write(data)


def run(checkpoint_dir, out, layer, rows=2):
    if rows < 1:
        raise ValueError("rows must be positive")
    cfg = json.loads((checkpoint_dir / "config.json").read_text())
    checkpoint = Checkpoint(checkpoint_dir, layer)
    prefix = f"model.layers.{layer}"
    hidden = bf16((((np.arange(rows * cfg["hidden_size"]) * 37) % 257) - 128).astype(F32) / 128.0).reshape(rows, cfg["hidden_size"])
    after_attention = attention(cfg, checkpoint, hidden, layer)
    moe_input = rmsnorm(after_attention, checkpoint.dequant(f"{prefix}.post_attention_layernorm.weight"), cfg["rms_norm_eps"])
    router_weight = checkpoint.dequant(f"{prefix}.mlp.gate.weight")
    router_bias = from_bf16(checkpoint.raw(f"{prefix}.moe.router.expert_bias")).astype(F32)
    logits = from_bf16(moe_input).astype(F32) @ from_bf16(router_weight).astype(F32).T
    router_ids, router_weights = topk_router(logits, router_bias, cfg["num_experts_per_tok"])
    selected = sorted(set(router_ids.ravel().tolist()))
    weights = {}
    for expert in selected:
        for projection in ["gate_proj", "up_proj", "down_proj"]:
            name = f"{prefix}.mlp.experts.{expert}.{projection}.weight"
            weights[f"expert.{expert}.{projection}"] = checkpoint.dequant(name)
    for projection in ["gate_proj", "up_proj", "down_proj"]:
        weights[f"shared.{projection}"] = checkpoint.dequant(f"{prefix}.mlp.shared_experts.{projection}.weight")
    quantized = {}
    quant_errors = []
    for name, weight in weights.items():
        source = from_bf16(weight)
        codes, scales = PACK.quantize_mxfp4(source)
        restored = PACK.FP4_VALUES[codes].reshape(source.shape[0], -1, 32) * np.ldexp(
            np.float32(1.0), scales.astype(np.int32) - 127
        ).astype(F32)[..., None]
        restored = restored.reshape(source.shape)
        quant_errors.append(relative_rmse(restored, source))
        quantized[name] = bf16(restored)

    def run_moe(source_weights, w4a8=False):
        forward = mlp_w4a8 if w4a8 else mlp
        routed = np.zeros_like(moe_input)
        for row in range(rows):
            for slot in range(cfg["num_experts_per_tok"]):
                expert = int(router_ids[row, slot])
                expert_weights = {projection: source_weights[f"expert.{expert}.{projection}"] for projection in ["gate_proj", "up_proj", "down_proj"]}
                value = forward(moe_input[row:row + 1], expert_weights)[0]
                weighted = bf16(from_bf16(value) * router_weights[row, slot])
                routed[row] = bf16(from_bf16(routed[row]) + from_bf16(weighted))
        shared = forward(moe_input, {projection: source_weights[f"shared.{projection}"] for projection in ["gate_proj", "up_proj", "down_proj"]})
        return bf16(from_bf16(routed) + from_bf16(shared))

    def finish(moe_output):
        post = rmsnorm(moe_output, checkpoint.dequant(f"{prefix}.post_ffn_norm.weight"), cfg["rms_norm_eps"])
        return bf16(from_bf16(after_attention) + from_bf16(post))

    expected = finish(run_moe(weights))
    expected_mxfp4 = finish(run_moe(quantized))
    expected_w4a8 = finish(run_moe(quantized, w4a8=True))
    tensors = {
        "input": hidden,
        "after_attention": after_attention,
        "moe_input": moe_input,
        "router_ids": router_ids,
        "router_weights": router_weights,
        "expected": expected,
        "expected_mxfp4": expected_mxfp4,
        "expected_w4a8": expected_w4a8,
        "input_norm": checkpoint.dequant(f"{prefix}.input_layernorm.weight"),
        "post_attn_norm": checkpoint.dequant(f"{prefix}.post_attn_norm.weight"),
        "post_attention_norm": checkpoint.dequant(f"{prefix}.post_attention_layernorm.weight"),
        "post_ffn_norm": checkpoint.dequant(f"{prefix}.post_ffn_norm.weight"),
        "q_norm": checkpoint.dequant(f"{prefix}.self_attn.q_norm.weight"),
        "k_norm": checkpoint.dequant(f"{prefix}.self_attn.k_norm.weight"),
        "router_weight": router_weight,
        "router_bias": router_bias,
    }
    for projection in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        tensors[projection] = checkpoint.dequant(f"{prefix}.self_attn.{projection}.weight")
    tensors.update(weights)
    out.mkdir(parents=True, exist_ok=True)
    save_safetensors(out / "layer.safetensors", tensors)
    metadata = {
        "source": "Aleph-Alpha/Kolibri-1",
        "source_shards": [path.name for path in checkpoint.paths],
        "layer": layer,
        "selected_experts": selected,
        "artifact_bytes": (out / "layer.safetensors").stat().st_size,
        "mxfp4_selected_expert_weight_relative_rmse": {
            "min": min(quant_errors), "mean": float(np.mean(quant_errors)), "max": max(quant_errors)
        },
        "mxfp4_full_layer_output_relative_rmse": relative_rmse(expected_mxfp4, expected),
        "mxfp4_full_layer_output_max_abs": float(np.max(np.abs(from_bf16(expected_mxfp4) - from_bf16(expected)))),
        "w4a8_full_layer_output_relative_rmse": relative_rmse(expected_w4a8, expected),
        "w4a8_full_layer_output_max_abs": float(np.max(np.abs(from_bf16(expected_w4a8) - from_bf16(expected)))),
        "reference": "NumPy float32 accumulation with BF16 rounding at projection and norm boundaries",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print(f"reference output finite: {bool(np.isfinite(from_bf16(expected)).all())}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("/srv/models/saber-models/models/Kolibri-1"))
    parser.add_argument("--out", type=Path, default=Path("/srv/models/saber-models/spike/kolibri-layer0"))
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--rows", type=int, default=2)
    args = parser.parse_args()
    run(args.checkpoint, args.out, args.layer, args.rows)


if __name__ == "__main__":
    main()
