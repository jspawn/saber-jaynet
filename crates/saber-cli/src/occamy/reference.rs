use std::path::Path;

use anyhow::Context;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 40) as f32 / (1u32 << 24) as f32 - 0.5
    }
}

fn bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn f32(value: u16) -> f32 {
    f32::from_bits(u32::from(value) << 16)
}

fn random(count: usize, scale: f32, seed: u64) -> Vec<u16> {
    let mut rng = Rng(seed);
    (0..count).map(|_| bf16(rng.next() * scale)).collect()
}

fn bytes<T: Copy>(values: &[T], width: usize, convert: impl Fn(T) -> [u8; 8]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&value| convert(value)[..width].to_vec())
        .collect()
}

fn bf16_bytes(values: &[u16]) -> Vec<u8> {
    bytes(values, 2, |value| u64::from(value).to_le_bytes())
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    bytes(values, 4, |value| i64::from(value).to_le_bytes())
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn save(root: &Path, name: &str, payload: &[u8]) -> anyhow::Result<()> {
    let path = root.join(format!("{name}.npy"));
    let mut header = format!(
        "{{'descr': '|u1', 'fortran_order': False, 'shape': ({},), }}",
        payload.len()
    )
    .into_bytes();
    while (10 + header.len()) % 64 != 0 {
        header.push(b' ');
    }
    header.push(b'\n');
    let mut file = std::fs::File::create(&path).with_context(|| path.display().to_string())?;
    use std::io::Write;
    file.write_all(b"\x93NUMPY\x01\x00")?;
    file.write_all(&(header.len() as u16).to_le_bytes())?;
    file.write_all(&header)?;
    file.write_all(payload)?;
    Ok(())
}

fn matmul(x: &[u16], rows: usize, k: usize, w: &[u16], n: usize) -> Vec<u16> {
    let mut output = vec![0u16; rows * n];
    for row in 0..rows {
        for col in 0..n {
            let mut sum = 0.0f32;
            for inner in 0..k {
                sum += f32(x[row * k + inner]) * f32(w[col * k + inner]);
            }
            output[row * n + col] = bf16(sum);
        }
    }
    output
}

fn silu_mul(projected: &[u16], rows: usize, inter: usize) -> Vec<u16> {
    let mut output = vec![0u16; rows * inter];
    for row in 0..rows {
        for col in 0..inter {
            let gate = f32(projected[row * 2 * inter + col]);
            let up = f32(projected[row * 2 * inter + inter + col]);
            let silu = gate / (1.0 + (-gate).exp());
            output[row * inter + col] = bf16(silu * up);
        }
    }
    output
}

fn attention(root: &Path) -> anyhow::Result<()> {
    const ROWS: usize = 5;
    const CONTEXT: usize = 32;
    const HEADS: usize = 8;
    const HEAD: usize = 256;
    let q = random(ROWS * HEADS * HEAD, 0.25, 100);
    let kv = random(2 * 16 * 2 * HEAD, 0.25, 101);
    save(root, "attn.q", &bf16_bytes(&q))?;
    save(root, "attn.kv", &bf16_bytes(&kv))?;
    save(root, "attn.block_table", &i32_bytes(&[0, 1]))?;
    save(root, "attn.seqused_k", &i32_bytes(&[CONTEXT as i32]))?;

    let mut expected = vec![0u16; ROWS * HEADS * HEAD];
    for row in 0..ROWS {
        for head in 0..HEADS {
            let mut scores = [0.0f32; CONTEXT];
            let mut maximum = f32::NEG_INFINITY;
            for key in 0..=CONTEXT - ROWS + row {
                let mut score = 0.0f32;
                for dim in 0..HEAD {
                    score +=
                        f32(q[(row * HEADS + head) * HEAD + dim]) * f32(kv[key * 2 * HEAD + dim]);
                }
                scores[key] = score / 16.0;
                maximum = maximum.max(scores[key]);
            }
            let mut total = 0.0f32;
            for score in &mut scores {
                *score = (*score - maximum).exp();
                total += *score;
            }
            for dim in 0..HEAD {
                let mut value = 0.0f32;
                for key in 0..=CONTEXT - ROWS + row {
                    value += scores[key] / total * f32(kv[key * 2 * HEAD + HEAD + dim]);
                }
                expected[(row * HEADS + head) * HEAD + dim] = bf16(value);
            }
        }
    }
    save(root, "attn.expected", &bf16_bytes(&expected))?;
    save(
        root,
        "attn.decode.q",
        &bf16_bytes(&q[(ROWS - 1) * HEADS * HEAD..]),
    )?;
    save(
        root,
        "attn.decode.expected",
        &bf16_bytes(&expected[(ROWS - 1) * HEADS * HEAD..]),
    )
}

fn expert(
    root: &Path,
    x: &[u16],
    global: u32,
    gu_seed: u64,
    down_seed: u64,
) -> anyhow::Result<Vec<u16>> {
    const H: usize = 2048;
    const I: usize = 512;
    let gu = random(2 * I * H, 0.0625, gu_seed);
    let down = random(H * I, 0.0625, down_seed);
    let prefix = if global == u32::MAX {
        "moe.shared".to_string()
    } else {
        format!("moe.expert.{global}")
    };
    save(root, &format!("{prefix}.gate_up"), &bf16_bytes(&gu))?;
    save(root, &format!("{prefix}.down"), &bf16_bytes(&down))?;
    let projected = matmul(x, 3, H, &gu, 2 * I);
    let act = silu_mul(&projected, 3, I);
    Ok(matmul(&act, 3, I, &down, H))
}

fn moe(root: &Path) -> anyhow::Result<()> {
    const T: usize = 3;
    const E: usize = 256;
    const K: usize = 8;
    const H: usize = 2048;
    let chosen = [0usize, 128, 1, 129, 2, 130, 3, 131];
    let x = random(T * H, 0.125, 200);
    let mut logits = random(T * E, 0.002, 201);
    let shared_gate = random(T, 0.5, 202);
    save(root, "moe.x", &bf16_bytes(&x))?;
    for row in 0..T {
        for (slot, &expert) in chosen.iter().enumerate() {
            logits[row * E + expert] = bf16(12.0 - slot as f32);
        }
    }
    save(root, "moe.logits", &bf16_bytes(&logits))?;
    save(root, "moe.shared_gate", &bf16_bytes(&shared_gate))?;

    let mut weights = vec![0.0f32; T * K];
    for row in 0..T {
        let maximum = (0..E)
            .map(|expert| f32(logits[row * E + expert]))
            .fold(f32::NEG_INFINITY, f32::max);
        let denominator: f32 = (0..E)
            .map(|expert| (f32(logits[row * E + expert]) - maximum).exp())
            .sum();
        let selected: f32 = chosen
            .iter()
            .map(|&expert| (f32(logits[row * E + expert]) - maximum).exp() / denominator)
            .sum();
        for (slot, &expert) in chosen.iter().enumerate() {
            weights[row * K + slot] =
                (f32(logits[row * E + expert]) - maximum).exp() / denominator / selected;
        }
    }
    let ids: Vec<i32> = (0..T)
        .flat_map(|_| chosen.iter().map(|&expert| expert as i32))
        .collect();
    save(root, "moe.ids", &i32_bytes(&ids))?;
    save(root, "moe.weights", &f32_bytes(&weights))?;

    let mut outputs = [vec![0.0f32; T * H], vec![0.0f32; T * H]];
    for (slot, &global) in chosen.iter().enumerate() {
        let rank = global / 128;
        let y = expert(
            root,
            &x,
            global as u32,
            300 + global as u64,
            400 + global as u64,
        )?;
        for row in 0..T {
            for col in 0..H {
                outputs[rank][row * H + col] += weights[row * K + slot] * f32(y[row * H + col]);
            }
        }
    }
    let shared = expert(root, &x, u32::MAX, 500, 501)?;
    for row in 0..T {
        let gate = 1.0 / (1.0 + (-f32(shared_gate[row])).exp());
        for col in 0..H {
            outputs[0][row * H + col] += gate * f32(shared[row * H + col]);
        }
    }
    for (rank, output) in outputs.iter().enumerate() {
        let expected: Vec<u16> = output.iter().map(|&value| bf16(value)).collect();
        save(
            root,
            &format!("moe.rank{rank}.expected"),
            &bf16_bytes(&expected),
        )?;
    }
    Ok(())
}

fn dense(root: &Path) -> anyhow::Result<()> {
    const K: usize = 2048;
    for (case, &(m, n)) in [
        (3, 6144),
        (3, 32),
        (3, 2048),
        (3, 4096),
        (3, 256),
        (3, 1),
        (17, 256),
        (33, 256),
        (65, 256),
    ]
    .iter()
    .enumerate()
    {
        let x = random(m * K, 0.0625, 600 + case as u64);
        let w = random(n * K, 0.0625, 700 + case as u64);
        let expected = matmul(&x, m, K, &w, n);
        let prefix = format!("dense.m{m}.n{n}");
        save(root, &format!("{prefix}.x"), &bf16_bytes(&x))?;
        save(root, &format!("{prefix}.w"), &bf16_bytes(&w))?;
        save(root, &format!("{prefix}.expected"), &bf16_bytes(&expected))?;
    }
    Ok(())
}

fn kv_append(root: &Path) -> anyhow::Result<()> {
    const ROWS: usize = 3;
    const HEAD: usize = 256;
    const BLOCK: usize = 16;
    let key = random(ROWS * HEAD, 0.5, 800);
    let value = random(ROWS * HEAD, 0.5, 801);
    let slots = [0i64, 16, 1];
    let mut cache = vec![0u16; 2 * BLOCK * 2 * HEAD];
    for (row, &slot) in slots.iter().enumerate() {
        let offset = slot as usize * 2 * HEAD;
        cache[offset..offset + HEAD].copy_from_slice(&key[row * HEAD..(row + 1) * HEAD]);
        cache[offset + HEAD..offset + 2 * HEAD]
            .copy_from_slice(&value[row * HEAD..(row + 1) * HEAD]);
    }
    save(root, "kv.key", &bf16_bytes(&key))?;
    save(root, "kv.value", &bf16_bytes(&value))?;
    save(
        root,
        "kv.slot",
        &slots
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    save(
        root,
        "kv.ring_slot",
        &[-1i64; ROWS]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    save(root, "kv.expected", &bf16_bytes(&cache))?;
    Ok(())
}

pub fn run(out: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(out)?;
    attention(out)?;
    moe(out)?;
    dense(out)?;
    kv_append(out)?;
    std::fs::write(
        out.join("case.json"),
        "{\"attention\":{\"rows\":5,\"context\":32,\"q_heads\":8,\"kv_heads\":1,\"head_dim\":256},\"moe\":{\"tokens\":3,\"experts\":256,\"local_experts\":128,\"topk\":8,\"hidden\":2048,\"intermediate\":512},\"dense\":{\"k\":2048,\"row_bands\":[3,17,33,65]},\"kv_append\":{\"rows\":3,\"kv_heads\":1,\"head_dim\":256,\"block_size\":16}}\n",
    )?;
    println!("occamy reference: {}", out.display());
    Ok(())
}
