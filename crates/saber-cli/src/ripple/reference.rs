use std::path::{Path, PathBuf};

use anyhow::{ensure, Context};
use saber_model::checkpoint_head::read_header;
use saber_model::ripple::RippleConfig;

pub fn bf16_to_f32(value: u16) -> f32 {
    f32::from_bits(u32::from(value) << 16)
}

pub fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

pub fn rmsnorm(input: &[u16], weight: &[u16], hidden: usize, eps: f32) -> Vec<u16> {
    let rows = input.len() / hidden;
    let mut output = vec![0; input.len()];
    for row in 0..rows {
        let input = &input[row * hidden..(row + 1) * hidden];
        let mean = input
            .iter()
            .map(|&x| {
                let x = bf16_to_f32(x);
                x * x
            })
            .sum::<f32>()
            / hidden as f32;
        let inv = (mean + eps).sqrt().recip();
        for i in 0..hidden {
            let normalized = bf16_to_f32(f32_to_bf16(bf16_to_f32(input[i]) * inv));
            output[row * hidden + i] = f32_to_bf16(normalized * bf16_to_f32(weight[i]));
        }
    }
    output
}

pub fn linear(input: &[u16], weight: &[u16], rows: usize, n: usize, k: usize) -> Vec<u16> {
    assert_eq!(input.len(), rows * k);
    assert_eq!(weight.len(), n * k);
    let mut output = vec![0; rows * n];
    for row in 0..rows {
        for col in 0..n {
            let mut sum = 0.0f32;
            for i in 0..k {
                sum += bf16_to_f32(input[row * k + i]) * bf16_to_f32(weight[col * k + i]);
            }
            output[row * n + col] = f32_to_bf16(sum);
        }
    }
    output
}

fn norm_rope(
    input: &[u16],
    weight: &[u16],
    rows: usize,
    heads: usize,
    positions: &[i64],
    table: &[u16],
    eps: f32,
) -> Vec<u16> {
    const DIM: usize = 128;
    let mut output = vec![0; input.len()];
    for (row, &position) in positions.iter().enumerate().take(rows) {
        for head in 0..heads {
            let base = (row * heads + head) * DIM;
            let mean = input[base..base + DIM]
                .iter()
                .map(|&x| {
                    let x = bf16_to_f32(x);
                    x * x
                })
                .sum::<f32>()
                / DIM as f32;
            let inv = (mean + eps).sqrt().recip();
            for i in 0..DIM {
                let partner = if i < 64 { i + 64 } else { i - 64 };
                let rotary = if i < 64 { i } else { i - 64 };
                let normalize = |index: usize| {
                    let n = bf16_to_f32(f32_to_bf16(bf16_to_f32(input[base + index]) * inv));
                    bf16_to_f32(f32_to_bf16(n * bf16_to_f32(weight[index])))
                };
                let x = normalize(i);
                let xp = normalize(partner);
                let table_base = position as usize * DIM;
                let c = bf16_to_f32(table[table_base + rotary]);
                let s = bf16_to_f32(table[table_base + 64 + rotary]);
                output[base + i] = f32_to_bf16(if i < 64 {
                    x * c - xp * s
                } else {
                    x * c + xp * s
                });
            }
        }
    }
    output
}

pub fn rope_table(cfg: &RippleConfig, positions: usize) -> Vec<u16> {
    rope_table_values(cfg.rope_theta, cfg.head_dim, positions)
}

pub fn rope_table_values(theta: f64, head_dim: u32, positions: usize) -> Vec<u16> {
    let dim = head_dim as usize;
    let mut table = vec![0; positions * dim];
    for position in 0..positions {
        for i in 0..dim / 2 {
            let frequency = 1.0f32 / (theta as f32).powf((2 * i) as f32 / dim as f32);
            let angle = position as f32 * frequency;
            table[position * dim + i] = f32_to_bf16(angle.cos());
            table[position * dim + dim / 2 + i] = f32_to_bf16(angle.sin());
        }
    }
    table
}

pub struct TensorFile {
    path: PathBuf,
    header: serde_json::Value,
    data_start: u64,
}

impl TensorFile {
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        let header = read_header(path).map_err(|e| anyhow::anyhow!(e))?;
        let mut length = [0u8; 8];
        std::os::unix::fs::FileExt::read_exact_at(&std::fs::File::open(path)?, &mut length, 0)?;
        let data_start = 8 + u64::from_le_bytes(length);
        Ok(Self {
            path: path.into(),
            header,
            data_start,
        })
    }

    fn read(&self, name: &str, dtype: &str) -> anyhow::Result<Vec<u8>> {
        let entry = self
            .header
            .get(name)
            .with_context(|| format!("{}: missing {name}", self.path.display()))?;
        ensure!(entry["dtype"] == dtype, "{name}: expected {dtype}");
        let offsets = entry["data_offsets"].as_array().unwrap();
        let begin = offsets[0].as_u64().unwrap();
        let end = offsets[1].as_u64().unwrap();
        let mut bytes = vec![0u8; (end - begin) as usize];
        std::os::unix::fs::FileExt::read_exact_at(
            &std::fs::File::open(&self.path)?,
            &mut bytes,
            self.data_start + begin,
        )?;
        Ok(bytes)
    }

    pub fn read_bf16(&self, name: &str) -> anyhow::Result<Vec<u16>> {
        Ok(self
            .read(name, "BF16")?
            .chunks_exact(2)
            .map(|x| u16::from_le_bytes([x[0], x[1]]))
            .collect())
    }

    pub fn read_f32(&self, name: &str) -> anyhow::Result<Vec<f32>> {
        Ok(self
            .read(name, "F32")?
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
            .collect())
    }

    pub fn read_i32(&self, name: &str) -> anyhow::Result<Vec<i32>> {
        Ok(self
            .read(name, "I32")?
            .chunks_exact(4)
            .map(|x| i32::from_le_bytes([x[0], x[1], x[2], x[3]]))
            .collect())
    }
}

pub struct AttentionWeights {
    pub q: Vec<u16>,
    pub k: Vec<u16>,
    pub v: Vec<u16>,
    pub g: Vec<u16>,
    pub o: Vec<u16>,
    pub q_norm: Vec<u16>,
    pub k_norm: Vec<u16>,
    pub input_norm: Vec<u16>,
    pub post_norm: Vec<u16>,
    pub lambda: u16,
}

impl AttentionWeights {
    pub fn load(file: &TensorFile, layer: u32) -> anyhow::Result<Self> {
        let base = format!("model.layers.{layer}");
        let matrix = |name: &str| file.read_bf16(&format!("{base}.self_attn.{name}.weight"));
        Ok(Self {
            q: matrix("q_proj")?,
            k: matrix("k_proj")?,
            v: matrix("v_proj")?,
            g: matrix("g_proj")?,
            o: matrix("o_proj")?,
            q_norm: file.read_bf16(&format!("{base}.self_attn.q_norm.weight"))?,
            k_norm: file.read_bf16(&format!("{base}.self_attn.k_norm.weight"))?,
            input_norm: file.read_bf16(&format!("{base}.input_layernorm.weight"))?,
            post_norm: file.read_bf16(&format!("{base}.post_attention_layernorm.weight"))?,
            lambda: file.read_bf16(&format!("{base}.self_attn.value_res_lambda"))?[0],
        })
    }
}

pub fn forward_attention(
    cfg: &RippleConfig,
    layer: u32,
    weights: &AttentionWeights,
    hidden: &[u16],
    initial: &[u16],
    positions: &[i64],
    table: &[u16],
) -> Vec<u16> {
    let rows = positions.len();
    let h = cfg.hidden_size as usize;
    let nq = cfg.num_attention_heads as usize;
    let nkv = cfg.num_key_value_heads as usize;
    let q_width = cfg.q_proj_rows() as usize;
    let kv_width = cfg.kv_proj_rows() as usize;
    let eps = cfg.rms_norm_eps as f32;
    let normed = rmsnorm(hidden, &weights.input_norm, h, eps);
    let q = linear(&normed, &weights.q, rows, q_width, h);
    let k = linear(&normed, &weights.k, rows, kv_width, h);
    let g = linear(&normed, &weights.g, rows, q_width, h);
    let blend = bf16_to_f32(f32_to_bf16(sigmoid(bf16_to_f32(weights.lambda))));
    let value_input: Vec<_> = normed
        .iter()
        .zip(initial)
        .map(|(&current, &first)| {
            f32_to_bf16(bf16_to_f32(first) + blend * (bf16_to_f32(current) - bf16_to_f32(first)))
        })
        .collect();
    let v = linear(&value_input, &weights.v, rows, kv_width, h);
    let q = norm_rope(&q, &weights.q_norm, rows, nq, positions, table, eps);
    let k = norm_rope(&k, &weights.k_norm, rows, nkv, positions, table, eps);
    let mut attention = vec![0; rows * q_width];
    for row in 0..rows {
        for qh in 0..nq {
            let kvh = qh / (nq / nkv);
            let position = positions[row] as usize;
            let first = if cfg.uses_rope(layer) {
                position.saturating_sub(cfg.sliding_window as usize - 1)
            } else {
                0
            };
            let mut scores = Vec::with_capacity(position + 1);
            for key in first..=position {
                let mut dot = 0.0f32;
                for d in 0..128 {
                    dot += bf16_to_f32(q[(row * nq + qh) * 128 + d])
                        * bf16_to_f32(k[(key * nkv + kvh) * 128 + d]);
                }
                scores.push(dot / 128.0f32.sqrt());
            }
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let probabilities: Vec<_> = scores.iter().map(|s| (s - maximum).exp()).collect();
            let denominator: f32 = probabilities.iter().sum();
            for d in 0..128 {
                let mut value = 0.0f32;
                for (key, probability) in (first..=position).zip(&probabilities) {
                    value += probability * bf16_to_f32(v[(key * nkv + kvh) * 128 + d]);
                }
                attention[(row * nq + qh) * 128 + d] = f32_to_bf16(value / denominator);
            }
        }
    }
    let gated: Vec<_> = attention
        .iter()
        .zip(&g)
        .map(|(&attention, &gate)| {
            let gate = bf16_to_f32(f32_to_bf16(sigmoid(bf16_to_f32(gate))));
            f32_to_bf16(bf16_to_f32(attention) * gate)
        })
        .collect();
    let output = linear(&gated, &weights.o, rows, h, q_width);
    let normed = rmsnorm(&output, &weights.post_norm, h, eps);
    hidden
        .iter()
        .zip(normed)
        .map(|(&residual, value)| f32_to_bf16(bf16_to_f32(residual) + bf16_to_f32(value)))
        .collect()
}
