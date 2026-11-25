//! The layer-0/layer-3 BF16 spike pack: dense TP2 and expert parallelism over two ranks.

use serde_json::{json, Value};

use super::config::{LayerKind, Qwen35MoeConfig};
use super::spike::WeightClass;

pub const TP: u32 = 2;
pub const EP: u32 = 2;
pub const SPIKE_LAYERS: [u32; 2] = [0, 3];

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Placement {
    Rank(u32),
    Shared,
}

impl Placement {
    #[must_use]
    pub fn dir(self) -> String {
        match self {
            Self::Rank(rank) => format!("rank{rank}"),
            Self::Shared => "shared".into(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceKind {
    Bf16,
    Bf16Merged,
    F32FromBf16,
}

impl SourceKind {
    #[must_use]
    pub fn key(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Bf16Merged => "bf16_merged",
            Self::F32FromBf16 => "f32_from_bf16",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shard {
    Replicated,
    Dim0,
    Dim1,
    Dim0Segments,
}

impl Shard {
    #[must_use]
    pub fn key(self) -> &'static str {
        match self {
            Self::Replicated => "replicated",
            Self::Dim0 => "dim0",
            Self::Dim1 => "dim1",
            Self::Dim0Segments => "dim0_segments",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Slice {
    pub tensor: String,
    pub shape: Vec<u64>,
    pub rows: (u64, u64),
    pub cols: (u64, u64),
}

impl Slice {
    fn whole(tensor: String, shape: &[u64]) -> Self {
        Self {
            tensor,
            shape: shape.to_vec(),
            rows: (0, shape[0]),
            cols: (0, shape[1..].iter().product::<u64>().max(1)),
        }
    }

    fn rows(tensor: String, shape: &[u64], start: u64, end: u64) -> Self {
        let mut slice = Self::whole(tensor, shape);
        slice.rows = (start, end);
        slice
    }

    fn cols(tensor: String, shape: &[u64], start: u64, end: u64) -> Self {
        let mut slice = Self::whole(tensor, shape);
        slice.cols = (start, end);
        slice
    }

    #[must_use]
    pub fn elems(&self) -> u64 {
        (self.rows.1 - self.rows.0) * (self.cols.1 - self.cols.0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    pub name: String,
    pub class: WeightClass,
    pub placement: Placement,
    pub kind: SourceKind,
    pub shard: Shard,
    pub dtype: &'static str,
    pub shape: Vec<u64>,
    pub sources: Vec<Slice>,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackPlan {
    pub layers: Vec<u32>,
    pub records: Vec<Record>,
}

struct Builder<'a> {
    config: &'a Qwen35MoeConfig,
    records: Vec<Record>,
}

impl Builder<'_> {
    #[allow(clippy::too_many_arguments)]
    fn push(
        &mut self,
        name: String,
        class: WeightClass,
        placement: Placement,
        kind: SourceKind,
        shard: Shard,
        dtype: &'static str,
        shape: Vec<u64>,
        sources: Vec<Slice>,
    ) {
        let elems: u64 = shape.iter().product();
        let bytes = elems
            * match dtype {
                "BF16" => 2,
                "F32" => 4,
                other => panic!("unsupported spike dtype {other}"),
            };
        assert_eq!(
            bytes,
            sources.iter().map(Slice::elems).sum::<u64>()
                * if kind == SourceKind::F32FromBf16 {
                    4
                } else {
                    2
                },
            "{name}: output bytes do not match the source extents"
        );
        self.records.push(Record {
            name,
            class,
            placement,
            kind,
            shard,
            dtype,
            shape,
            sources,
            bytes,
        });
    }

    fn shared(&mut self, name: &str, class: WeightClass, shape: &[u64]) {
        self.push(
            name.into(),
            class,
            Placement::Shared,
            SourceKind::Bf16,
            Shard::Replicated,
            "BF16",
            shape.to_vec(),
            vec![Slice::whole(name.into(), shape)],
        );
    }

    fn dim0(&mut self, name: &str, class: WeightClass, shape: &[u64]) {
        let per = shape[0] / u64::from(TP);
        for rank in 0..TP {
            let start = u64::from(rank) * per;
            let mut local = shape.to_vec();
            local[0] = per;
            self.push(
                name.into(),
                class,
                Placement::Rank(rank),
                SourceKind::Bf16,
                Shard::Dim0,
                "BF16",
                local,
                vec![Slice::rows(name.into(), shape, start, start + per)],
            );
        }
    }

    fn dim1(&mut self, name: &str, class: WeightClass, shape: &[u64]) {
        assert_eq!(shape.len(), 2, "{name}: dim1 records are matrices");
        let per = shape[1] / u64::from(TP);
        for rank in 0..TP {
            let start = u64::from(rank) * per;
            self.push(
                name.into(),
                class,
                Placement::Rank(rank),
                SourceKind::Bf16,
                Shard::Dim1,
                "BF16",
                vec![shape[0], per],
                vec![Slice::cols(name.into(), shape, start, start + per)],
            );
        }
    }

    fn merged(&mut self, name: &str, class: WeightClass, parts: &[(String, Vec<u64>, Vec<u64>)]) {
        for rank in 0..TP {
            let mut sources = Vec::new();
            let mut rows = 0;
            for (tensor, shape, segments) in parts {
                let mut base = 0;
                for &segment in segments {
                    let per = segment / u64::from(TP);
                    let start = base + u64::from(rank) * per;
                    sources.push(Slice::rows(tensor.clone(), shape, start, start + per));
                    rows += per;
                    base += segment;
                }
            }
            let mut shape = vec![rows];
            shape.extend(&parts[0].1[1..]);
            self.push(
                name.into(),
                class,
                Placement::Rank(rank),
                if sources.len() == 1 {
                    SourceKind::Bf16
                } else {
                    SourceKind::Bf16Merged
                },
                Shard::Dim0Segments,
                "BF16",
                shape,
                sources,
            );
        }
    }

    fn layer(&mut self, layer: u32) {
        let c = self.config;
        let h = u64::from(c.hidden_size);
        let p = format!("model.language_model.layers.{layer}");
        for norm in ["input_layernorm", "post_attention_layernorm"] {
            self.shared(&format!("{p}.{norm}.weight"), WeightClass::Norm, &[h]);
        }
        match c.layers[layer as usize] {
            LayerKind::Gdn => {
                let g = &c.gdn;
                let a = format!("{p}.linear_attn");
                let (kd, vd) = (u64::from(g.key_dim()), u64::from(g.value_dim()));
                let qkv = u64::from(g.qkv_rows());
                let nv = u64::from(g.num_value_heads);
                self.merged(
                    &format!("layer.{layer}.gdn.in_proj_qkvz.w"),
                    WeightClass::Gdn,
                    &[
                        (
                            format!("{a}.in_proj_qkv.weight"),
                            vec![qkv, h],
                            vec![kd, kd, vd],
                        ),
                        (format!("{a}.in_proj_z.weight"), vec![vd, h], vec![vd]),
                    ],
                );
                self.merged(
                    &format!("layer.{layer}.gdn.in_proj_ba.w"),
                    WeightClass::Gdn,
                    &[
                        (format!("{a}.in_proj_b.weight"), vec![nv, h], vec![nv]),
                        (format!("{a}.in_proj_a.weight"), vec![nv, h], vec![nv]),
                    ],
                );
                self.merged(
                    &format!("{a}.conv1d.weight"),
                    WeightClass::Gdn,
                    &[(
                        format!("{a}.conv1d.weight"),
                        vec![qkv, 1, u64::from(g.conv_kernel)],
                        vec![kd, kd, vd],
                    )],
                );
                for tensor in ["A_log", "dt_bias"] {
                    let name = format!("{a}.{tensor}");
                    let per = nv / u64::from(TP);
                    for rank in 0..TP {
                        let start = u64::from(rank) * per;
                        self.push(
                            name.clone(),
                            WeightClass::Gdn,
                            Placement::Rank(rank),
                            SourceKind::F32FromBf16,
                            Shard::Dim0,
                            "F32",
                            vec![per],
                            vec![Slice::rows(name.clone(), &[nv], start, start + per)],
                        );
                    }
                }
                self.shared(
                    &format!("{a}.norm.weight"),
                    WeightClass::Gdn,
                    &[u64::from(g.value_head_dim)],
                );
                self.dim1(&format!("{a}.out_proj.weight"), WeightClass::Gdn, &[h, vd]);
            }
            LayerKind::FullAttention => {
                let a = format!("{p}.self_attn");
                let kv = u64::from(c.kv_proj_rows());
                self.dim0(
                    &format!("{a}.q_proj.weight"),
                    WeightClass::Attention,
                    &[u64::from(c.q_proj_rows()), h],
                );
                for projection in ["k_proj", "v_proj"] {
                    self.dim0(
                        &format!("{a}.{projection}.weight"),
                        WeightClass::Attention,
                        &[kv, h],
                    );
                }
                self.dim1(
                    &format!("{a}.o_proj.weight"),
                    WeightClass::Attention,
                    &[
                        h,
                        u64::from(c.attention.num_heads) * u64::from(c.attention.head_dim),
                    ],
                );
                for norm in ["q_norm", "k_norm"] {
                    self.shared(
                        &format!("{a}.{norm}.weight"),
                        WeightClass::Attention,
                        &[u64::from(c.attention.head_dim)],
                    );
                }
            }
        }

        let m = format!("{p}.mlp");
        self.shared(
            &format!("{m}.gate.weight"),
            WeightClass::Router,
            &[u64::from(c.moe.num_experts), h],
        );
        self.shared(
            &format!("{m}.shared_expert_gate.weight"),
            WeightClass::Router,
            &[1, h],
        );
        let shared = format!("{m}.shared_expert");
        let inter = u64::from(c.moe.shared_intermediate);
        self.push(
            format!("layer.{layer}.mlp.shared_expert.gate_up.w"),
            WeightClass::SharedExpert,
            Placement::Shared,
            SourceKind::Bf16Merged,
            Shard::Replicated,
            "BF16",
            vec![2 * inter, h],
            vec![
                Slice::whole(format!("{shared}.gate_proj.weight"), &[inter, h]),
                Slice::whole(format!("{shared}.up_proj.weight"), &[inter, h]),
            ],
        );
        self.shared(
            &format!("{shared}.down_proj.weight"),
            WeightClass::SharedExpert,
            &[h, inter],
        );

        let inter = u64::from(c.moe.intermediate);
        let per = c.moe.num_experts / EP;
        for rank in 0..EP {
            for expert in rank * per..(rank + 1) * per {
                let x = format!("{m}.experts.{expert}");
                self.push(
                    format!("layer.{layer}.expert.{expert}.gate_up.w"),
                    WeightClass::RoutedExpert,
                    Placement::Rank(rank),
                    SourceKind::Bf16Merged,
                    Shard::Dim0,
                    "BF16",
                    vec![2 * inter, h],
                    vec![
                        Slice::whole(format!("{x}.gate_proj.weight"), &[inter, h]),
                        Slice::whole(format!("{x}.up_proj.weight"), &[inter, h]),
                    ],
                );
                self.push(
                    format!("{x}.down_proj.weight"),
                    WeightClass::RoutedExpert,
                    Placement::Rank(rank),
                    SourceKind::Bf16,
                    Shard::Dim0,
                    "BF16",
                    vec![h, inter],
                    vec![Slice::whole(format!("{x}.down_proj.weight"), &[h, inter])],
                );
            }
        }
    }
}

impl PackPlan {
    fn for_layers(config: &Qwen35MoeConfig, layers: Vec<u32>) -> Result<Self, String> {
        config.validate_tp2().map_err(|error| error.to_string())?;
        let mut builder = Builder {
            config,
            records: Vec::new(),
        };
        for layer in layers.iter().copied() {
            builder.layer(layer);
        }
        Ok(Self {
            layers,
            records: builder.records,
        })
    }

    pub fn spike(config: &Qwen35MoeConfig) -> Result<Self, String> {
        if config.layers[SPIKE_LAYERS[0] as usize] != LayerKind::Gdn
            || config.layers[SPIKE_LAYERS[1] as usize] != LayerKind::FullAttention
        {
            return Err("the spike requires GDN layer 0 and full-attention layer 3".into());
        }
        Self::for_layers(config, SPIKE_LAYERS.to_vec())
    }

    pub fn all_layers(config: &Qwen35MoeConfig) -> Result<Self, String> {
        Self::for_layers(config, (0..config.num_layers()).collect())
    }

    pub fn globals(config: &Qwen35MoeConfig) -> Result<Self, String> {
        config.validate_tp2().map_err(|error| error.to_string())?;
        if config.vocab_size % TP != 0 {
            return Err(format!(
                "vocabulary {} is not divisible by TP2",
                config.vocab_size
            ));
        }
        let mut builder = Builder {
            config,
            records: Vec::new(),
        };
        let shape = [u64::from(config.vocab_size), u64::from(config.hidden_size)];
        builder.dim0(
            "model.language_model.embed_tokens.weight",
            WeightClass::Embedding,
            &shape,
        );
        builder.dim0("lm_head.weight", WeightClass::Head, &shape);
        builder.shared(
            "model.language_model.norm.weight",
            WeightClass::Norm,
            &[u64::from(config.hidden_size)],
        );
        Ok(Self {
            layers: Vec::new(),
            records: builder.records,
        })
    }

    #[must_use]
    pub fn to_json(&self) -> Value {
        let slice = |s: &Slice| {
            json!({"tensor": s.tensor, "shape": s.shape, "rows": [s.rows.0, s.rows.1],
                   "cols": [s.cols.0, s.cols.1]})
        };
        let scope = if self.layers.is_empty() {
            "model globals; text only; no MTP or vision".to_string()
        } else if self.layers == SPIKE_LAYERS {
            "layers 0 and 3; text only; no MTP or vision".to_string()
        } else {
            format!(
                "{} decoder layers; text only; no MTP or vision",
                self.layers.len()
            )
        };
        json!({
            "model": "qwen3_5_moe",
            "source": "bf16",
            "scope": scope,
            "layout": {"tp": TP, "ep": EP, "routed_experts_per_rank": 128,
                       "shared_expert": "replicated BF16"},
            "layers": self.layers,
            "records": self.records.iter().map(|r| json!({
                "name": r.name, "class": format!("{:?}", r.class),
                "placement": r.placement.dir(), "kind": r.kind.key(), "shard": r.shard.key(),
                "dtype": r.dtype, "shape": r.shape, "bytes": r.bytes,
                "sources": r.sources.iter().map(slice).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
        })
    }
}
