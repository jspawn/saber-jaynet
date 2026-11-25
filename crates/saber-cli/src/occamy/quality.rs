#![allow(clippy::too_many_arguments)]

use std::path::Path;

use anyhow::{ensure, Context};
use saber_model::qwen3_5_moe::{LayerKind, Qwen35MoeConfig};
use saber_model::qwen4_exp::st::ShardSet;
use serde_json::json;

const H: usize = 2048;
const GKH: usize = 16;
const GVH: usize = 32;
const GD: usize = 128;
const CONV: usize = (2 * GKH + GVH) * GD;
const Z: usize = GVH * GD;
const AH: usize = 16;
const AKVH: usize = 2;
const AD: usize = 256;
const INTER: usize = 512;

fn bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn f32(value: u16) -> f32 {
    f32::from_bits(u32::from(value) << 16)
}

fn f32s(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|value| u16::from_le_bytes([value[0], value[1]]))
        .collect()
}

fn bytes(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn linear(input: &[u16], rows: usize, k: usize, weight: &[u16], n: usize) -> Vec<u16> {
    assert_eq!(input.len(), rows * k);
    assert_eq!(weight.len(), n * k);
    let mut output = vec![0; rows * n];
    for row in 0..rows {
        for col in 0..n {
            let mut sum = 0.0f32;
            for inner in 0..k {
                sum += f32(input[row * k + inner]) * f32(weight[col * k + inner]);
            }
            output[row * n + col] = bf16(sum);
        }
    }
    output
}

fn rms(input: &[u16], weight: &[u16], width: usize) -> Vec<u16> {
    input
        .chunks_exact(width)
        .flat_map(|row| {
            let square: f32 = row.iter().map(|&value| f32(value) * f32(value)).sum();
            let inverse = 1.0 / (square / width as f32 + 1e-6).sqrt();
            row.iter()
                .zip(weight)
                .map(|(&value, &weight)| bf16(f32(value) * inverse * (1.0 + f32(weight))))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn add(left: &[u16], right: &[u16]) -> Vec<u16> {
    left.iter()
        .zip(right)
        .map(|(&left, &right)| bf16(f32(left) + f32(right)))
        .collect()
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        value.exp().ln_1p()
    }
}

struct GdnState {
    history: Vec<u16>,
    recurrent: Vec<f32>,
}

struct AttentionState {
    keys: Vec<u16>,
    values: Vec<u16>,
}

enum LayerState {
    Gdn(GdnState),
    Attention(AttentionState),
}

struct Reference {
    shards: ShardSet,
    config: Qwen35MoeConfig,
    states: Vec<LayerState>,
}

impl Reference {
    fn new(checkpoint: &Path) -> anyhow::Result<Self> {
        let config_text = std::fs::read_to_string(checkpoint.join("config.json"))?;
        let config = Qwen35MoeConfig::from_config_json(&config_text)
            .map_err(|error| anyhow::anyhow!(error))?;
        let states = config
            .layers
            .iter()
            .map(|kind| match kind {
                LayerKind::Gdn => LayerState::Gdn(GdnState {
                    history: vec![0; 3 * CONV],
                    recurrent: vec![0.0; GVH * GD * GD],
                }),
                LayerKind::FullAttention => LayerState::Attention(AttentionState {
                    keys: Vec::new(),
                    values: Vec::new(),
                }),
            })
            .collect();
        Ok(Self {
            shards: ShardSet::open(checkpoint).map_err(|error| anyhow::anyhow!(error))?,
            config,
            states,
        })
    }

    fn matrix(&mut self, name: &str, n: usize, k: usize) -> anyhow::Result<Vec<u16>> {
        let (info, raw) = self
            .shards
            .read(name)
            .map_err(|error| anyhow::anyhow!(error))?;
        ensure!(
            info.dtype == "BF16"
                && info.shape.first() == Some(&(n as u64))
                && info.shape.iter().product::<u64>() == (n * k) as u64,
            "{name}: unexpected tensor {} {:?}",
            info.dtype,
            info.shape
        );
        Ok(f32s(&raw))
    }

    fn vector(&mut self, name: &str, len: usize) -> anyhow::Result<Vec<u16>> {
        let (info, raw) = self
            .shards
            .read(name)
            .map_err(|error| anyhow::anyhow!(error))?;
        ensure!(
            info.dtype == "BF16" && info.shape == [len as u64],
            "{name}: unexpected tensor {} {:?}",
            info.dtype,
            info.shape
        );
        Ok(f32s(&raw))
    }

    fn projected(
        &mut self,
        input: &[u16],
        rows: usize,
        name: &str,
        n: usize,
    ) -> anyhow::Result<Vec<u16>> {
        let weight = self.matrix(name, n, H)?;
        Ok(linear(input, rows, H, &weight, n))
    }

    fn gdn(&mut self, layer: usize, input: &[u16], rows: usize) -> anyhow::Result<Vec<u16>> {
        let prefix = format!("model.language_model.layers.{layer}.linear_attn");
        let qkv = self.projected(input, rows, &format!("{prefix}.in_proj_qkv.weight"), CONV)?;
        let z = self.projected(input, rows, &format!("{prefix}.in_proj_z.weight"), Z)?;
        let b = self.projected(input, rows, &format!("{prefix}.in_proj_b.weight"), GVH)?;
        let a = self.projected(input, rows, &format!("{prefix}.in_proj_a.weight"), GVH)?;
        let convolution = self.matrix(&format!("{prefix}.conv1d.weight"), CONV, 4)?;
        let a_log = self.vector(&format!("{prefix}.A_log"), GVH)?;
        let dt_bias = self.vector(&format!("{prefix}.dt_bias"), GVH)?;
        let norm = self.vector(&format!("{prefix}.norm.weight"), GD)?;
        let LayerState::Gdn(state) = &mut self.states[layer] else {
            unreachable!()
        };
        let mut core = vec![0; rows * Z];
        for token in 0..rows {
            let mut mixed = vec![0; CONV];
            for channel in 0..CONV {
                let mut sum = 0.0f32;
                for tap in 0..4 {
                    let source = token + tap;
                    let value = if source >= 3 {
                        qkv[(source - 3) * CONV + channel]
                    } else {
                        state.history[source * CONV + channel]
                    };
                    sum += f32(value) * f32(convolution[channel * 4 + tap]);
                }
                mixed[channel] = bf16(sum / (1.0 + (-sum).exp()));
            }
            state.history.copy_within(CONV..3 * CONV, 0);
            state.history[2 * CONV..].copy_from_slice(&qkv[token * CONV..(token + 1) * CONV]);
            for head in 0..GVH {
                let key_head = head / (GVH / GKH);
                let mut q = [0.0f32; GD];
                let mut k = [0.0f32; GD];
                for (region, output) in [(0, &mut q), (1, &mut k)] {
                    let base = (region * GKH + key_head) * GD;
                    let square: f32 = mixed[base..base + GD]
                        .iter()
                        .map(|&value| f32(value) * f32(value))
                        .sum();
                    let inverse = 1.0 / (square + 1e-6).sqrt();
                    for dim in 0..GD {
                        output[dim] = f32(bf16(f32(mixed[base + dim]) * inverse));
                    }
                }
                let gate = -f32(a_log[head]).exp()
                    * softplus(f32(a[token * GVH + head]) + f32(dt_bias[head]));
                let decay = gate.exp();
                let beta = 1.0 / (1.0 + (-f32(b[token * GVH + head])).exp());
                for value_dim in 0..GD {
                    let mut memory = 0.0f32;
                    for (key_dim, &k_value) in k.iter().enumerate().take(GD) {
                        let index = (head * GD + key_dim) * GD + value_dim;
                        state.recurrent[index] *= decay;
                        memory += state.recurrent[index] * k_value;
                    }
                    let delta = (f32(mixed[2 * GKH * GD + head * GD + value_dim]) - memory) * beta;
                    let mut value = 0.0f32;
                    for (key_dim, &k_value) in k.iter().enumerate().take(GD) {
                        let index = (head * GD + key_dim) * GD + value_dim;
                        state.recurrent[index] += k_value * delta;
                        value += state.recurrent[index] * q[key_dim] / (GD as f32).sqrt();
                    }
                    core[(token * GVH + head) * GD + value_dim] = bf16(value);
                }
            }
        }
        let mut gated = vec![0; rows * Z];
        for token in 0..rows {
            for head in 0..GVH {
                let base = (token * GVH + head) * GD;
                let square: f32 = core[base..base + GD]
                    .iter()
                    .map(|&value| f32(value) * f32(value))
                    .sum();
                let inverse = 1.0 / (square / GD as f32 + 1e-6).sqrt();
                for dim in 0..GD {
                    let gate = f32(z[base + dim]);
                    gated[base + dim] = bf16(
                        f32(core[base + dim]) * inverse * f32(norm[dim]) * gate
                            / (1.0 + (-gate).exp()),
                    );
                }
            }
        }
        let output = self.matrix(&format!("{prefix}.out_proj.weight"), H, Z)?;
        Ok(linear(&gated, rows, Z, &output, H))
    }

    fn norm_rope(
        &self,
        projected: &[u16],
        weight: &[u16],
        rows: usize,
        heads: usize,
        position: usize,
        stride: usize,
        offset: usize,
        gated: bool,
    ) -> Vec<u16> {
        let mut output = vec![0; rows * heads * AD];
        for token in 0..rows {
            for head in 0..heads {
                let source = token * stride + offset + head * if gated { 2 * AD } else { AD };
                let square: f32 = projected[source..source + AD]
                    .iter()
                    .map(|&value| f32(value) * f32(value))
                    .sum();
                let inverse = 1.0 / (square / AD as f32 + 1e-6).sqrt();
                for dim in 0..AD {
                    let value =
                        bf16(f32(projected[source + dim]) * inverse * (1.0 + f32(weight[dim])));
                    let rotated = if dim < 64 {
                        let pair = if dim < 32 { dim + 32 } else { dim - 32 };
                        let other = bf16(
                            f32(projected[source + pair]) * inverse * (1.0 + f32(weight[pair])),
                        );
                        let angle = (position + token) as f32
                            * 10_000_000f32.powf(-((dim % 32) as f32) / 32.0);
                        let cos = f32(bf16(angle.cos()));
                        let sin = f32(bf16(angle.sin()));
                        if dim < 32 {
                            f32(bf16(f32(value) * cos)) - f32(bf16(f32(other) * sin))
                        } else {
                            f32(bf16(f32(value) * cos)) + f32(bf16(f32(other) * sin))
                        }
                    } else {
                        f32(value)
                    };
                    output[(token * heads + head) * AD + dim] = bf16(rotated);
                }
            }
        }
        output
    }

    fn attention(
        &mut self,
        layer: usize,
        input: &[u16],
        rows: usize,
        position: usize,
    ) -> anyhow::Result<Vec<u16>> {
        let prefix = format!("model.language_model.layers.{layer}.self_attn");
        let projected_q =
            self.projected(input, rows, &format!("{prefix}.q_proj.weight"), AH * 2 * AD)?;
        let k = self.projected(input, rows, &format!("{prefix}.k_proj.weight"), AKVH * AD)?;
        let v = self.projected(input, rows, &format!("{prefix}.v_proj.weight"), AKVH * AD)?;
        let qw = self.vector(&format!("{prefix}.q_norm.weight"), AD)?;
        let kw = self.vector(&format!("{prefix}.k_norm.weight"), AD)?;
        let q = self.norm_rope(&projected_q, &qw, rows, AH, position, AH * 2 * AD, 0, true);
        let k = self.norm_rope(&k, &kw, rows, AKVH, position, AKVH * AD, 0, false);
        let LayerState::Attention(state) = &mut self.states[layer] else {
            unreachable!()
        };
        state.keys.extend_from_slice(&k);
        state.values.extend_from_slice(&v);
        let context = position + rows;
        let mut core = vec![0; rows * AH * AD];
        for token in 0..rows {
            for head in 0..AH {
                let kv_head = head / (AH / AKVH);
                let mut scores = vec![0.0f32; position + token + 1];
                for (key, score) in scores.iter_mut().enumerate() {
                    for dim in 0..AD {
                        *score += f32(q[(token * AH + head) * AD + dim])
                            * f32(state.keys[(key * AKVH + kv_head) * AD + dim]);
                    }
                    *score /= 16.0;
                }
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let denominator: f32 = scores.iter().map(|score| (*score - maximum).exp()).sum();
                for dim in 0..AD {
                    let mut sum = 0.0f32;
                    for (key, score) in scores.iter().enumerate() {
                        let probability = bf16((*score - maximum).exp() / denominator);
                        sum +=
                            f32(probability) * f32(state.values[(key * AKVH + kv_head) * AD + dim]);
                    }
                    core[(token * AH + head) * AD + dim] = bf16(sum);
                }
            }
        }
        assert_eq!(context * AKVH * AD, state.keys.len());
        let mut gated = vec![0; rows * AH * AD];
        for token in 0..rows {
            for head in 0..AH {
                for dim in 0..AD {
                    let gate = f32(projected_q[token * AH * 2 * AD + head * 2 * AD + AD + dim]);
                    let gate = f32(bf16(1.0 / (1.0 + (-gate).exp())));
                    gated[(token * AH + head) * AD + dim] =
                        bf16(f32(core[(token * AH + head) * AD + dim]) * gate);
                }
            }
        }
        let output = self.matrix(&format!("{prefix}.o_proj.weight"), H, AH * AD)?;
        Ok(linear(&gated, rows, AH * AD, &output, H))
    }

    fn expert(&mut self, prefix: &str, input: &[u16]) -> anyhow::Result<Vec<u16>> {
        let gate = self.matrix(&format!("{prefix}.gate_proj.weight"), INTER, H)?;
        let up = self.matrix(&format!("{prefix}.up_proj.weight"), INTER, H)?;
        let gate = linear(input, 1, H, &gate, INTER);
        let up = linear(input, 1, H, &up, INTER);
        let activation: Vec<u16> = gate
            .iter()
            .zip(up)
            .map(|(&gate, up)| {
                let gate = f32(gate);
                bf16(gate / (1.0 + (-gate).exp()) * f32(up))
            })
            .collect();
        let down = self.matrix(&format!("{prefix}.down_proj.weight"), H, INTER)?;
        Ok(linear(&activation, 1, INTER, &down, H))
    }

    fn moe(&mut self, layer: usize, input: &[u16], rows: usize) -> anyhow::Result<Vec<u16>> {
        let prefix = format!("model.language_model.layers.{layer}.mlp");
        let router = self.projected(input, rows, &format!("{prefix}.gate.weight"), 256)?;
        let shared_gate = self.projected(
            input,
            rows,
            &format!("{prefix}.shared_expert_gate.weight"),
            1,
        )?;
        let mut output = vec![0.0f32; rows * H];
        for token in 0..rows {
            let row = &input[token * H..(token + 1) * H];
            let logits = &router[token * 256..(token + 1) * 256];
            let maximum = logits
                .iter()
                .map(|&value| f32(value))
                .fold(f32::NEG_INFINITY, f32::max);
            let denominator: f32 = logits
                .iter()
                .map(|&value| (f32(value) - maximum).exp())
                .sum();
            let mut selected: Vec<(usize, f32)> = logits
                .iter()
                .enumerate()
                .map(|(id, &value)| (id, (f32(value) - maximum).exp() / denominator))
                .collect();
            selected.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));
            selected.truncate(8);
            let sum: f32 = selected.iter().map(|selected| selected.1).sum();
            for (expert, weight) in selected {
                let weight = f32(bf16(weight / sum));
                let expert = self.expert(&format!("{prefix}.experts.{expert}"), row)?;
                for col in 0..H {
                    output[token * H + col] += weight * f32(expert[col]);
                }
            }
            let shared = self.expert(&format!("{prefix}.shared_expert"), row)?;
            let gate = f32(shared_gate[token]);
            let gate = 1.0 / (1.0 + (-gate).exp());
            for col in 0..H {
                output[token * H + col] =
                    f32(bf16(output[token * H + col])) + gate * f32(shared[col]);
            }
        }
        Ok(output.into_iter().map(bf16).collect())
    }

    fn head(&mut self, input: &[u16]) -> anyhow::Result<Vec<u16>> {
        let weight = self.matrix("lm_head.weight", self.config.vocab_size as usize, H)?;
        Ok(linear(
            input,
            1,
            H,
            &weight,
            self.config.vocab_size as usize,
        ))
    }

    fn forward(
        &mut self,
        tokens: &[u32],
        position: usize,
        trace: Option<&Path>,
    ) -> anyhow::Result<Vec<u16>> {
        let rows = tokens.len();
        let embedding = self.matrix(
            "model.language_model.embed_tokens.weight",
            self.config.vocab_size as usize,
            H,
        )?;
        let mut hidden = Vec::with_capacity(rows * H);
        for &token in tokens {
            ensure!(
                (token as usize) < self.config.vocab_size as usize,
                "token {token} is outside the vocabulary"
            );
            hidden.extend_from_slice(&embedding[token as usize * H..(token as usize + 1) * H]);
        }
        if let Some(root) = trace {
            std::fs::write(root.join("embed.bf16"), bytes(&hidden[(rows - 1) * H..]))?;
        }
        drop(embedding);
        for layer in 0..self.config.num_layers() as usize {
            let prefix = format!("model.language_model.layers.{layer}");
            let input_norm = self.vector(&format!("{prefix}.input_layernorm.weight"), H)?;
            let input_normed = rms(&hidden, &input_norm, H);
            let mixed = match self.config.layers[layer] {
                LayerKind::Gdn => self.gdn(layer, &input_normed, rows)?,
                LayerKind::FullAttention => self.attention(layer, &input_normed, rows, position)?,
            };
            let residual = add(&hidden, &mixed);
            let post_norm = self.vector(&format!("{prefix}.post_attention_layernorm.weight"), H)?;
            let moe_input = rms(&residual, &post_norm, H);
            let moe = self.moe(layer, &moe_input, rows)?;
            hidden = add(&residual, &moe);
            if let Some(root) = trace {
                for (stage, values) in [
                    ("input_norm", &input_normed),
                    ("mixer", &mixed),
                    ("mixer_residual", &residual),
                    ("moe_input", &moe_input),
                    ("moe", &moe),
                ] {
                    std::fs::write(
                        root.join(format!("layer{layer:02}.{stage}.bf16")),
                        bytes(&values[(rows - 1) * H..]),
                    )?;
                }
                std::fs::write(
                    root.join(format!("layer{layer:02}.bf16")),
                    bytes(&hidden[(rows - 1) * H..]),
                )?;
            }
            println!("occamy quality reference: layer {layer} done");
        }
        let norm = self.vector("model.language_model.norm.weight", H)?;
        let normed = rms(&hidden[(rows - 1) * H..], &norm, H);
        let logits = self.head(&normed)?;
        if let Some(root) = trace {
            std::fs::write(root.join("logits.bf16"), bytes(&logits))?;
        }
        Ok(logits)
    }
}

fn pick(logits: &[u16]) -> u32 {
    let mut best = 0usize;
    for index in 1..logits.len() {
        if f32(logits[index]) > f32(logits[best]) {
            best = index;
        }
    }
    best as u32
}

pub fn run(checkpoint: &Path, out: &Path, tokens: &[u32], decode: usize) -> anyhow::Result<()> {
    ensure!(!tokens.is_empty(), "at least one prompt token is required");
    ensure!(decode > 0, "--decode must be positive");
    ensure!(
        !out.exists(),
        "{}: output exists; refusing to overwrite",
        out.display()
    );
    std::fs::create_dir_all(out)?;
    let mut reference =
        Reference::new(checkpoint).with_context(|| checkpoint.display().to_string())?;
    let mut generated = Vec::with_capacity(decode);
    let logits = reference.forward(tokens, 0, Some(out))?;
    let mut next = pick(&logits);
    generated.push(next);
    println!("occamy quality reference: prompt argmax {next}");
    for _ in 1..decode {
        let position = tokens.len() + generated.len() - 1;
        let logits = reference.forward(&[next], position, None)?;
        next = pick(&logits);
        generated.push(next);
        println!("occamy quality reference: generated {next}");
    }
    std::fs::write(
        out.join("reference.json"),
        serde_json::to_vec_pretty(&json!({"prompt": tokens, "generated": generated}))?,
    )?;
    println!(
        "occamy quality reference: ids {generated:?}; trace {}",
        out.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_and_norm_cover_reference_rounding() {
        let input = [bf16(1.0), bf16(-2.0)];
        let weight = [bf16(0.5), bf16(0.25), bf16(-1.0), bf16(3.0)];
        assert_eq!(
            linear(&input, 1, 2, &weight, 2),
            vec![bf16(0.0), bf16(-7.0)]
        );
        assert_eq!(rms(&input, &[0, 0], 2).len(), 2);
    }
}
