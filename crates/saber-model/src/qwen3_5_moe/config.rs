//! Occamy's `qwen3_5_moe` configuration, kept separate from Flash-Next's `qwen4_exp` reader.
//!
//! BF16 and official block-128 FP8 checkpoints are accepted. MTP and the vision tower are
//! described here but are not execution features yet.

use serde_json::Value;

use crate::qwen4_exp::fields::Fields;
use crate::ModelError;

fn err(msg: impl Into<String>) -> ModelError {
    ModelError::Config(msg.into())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    Gdn,
    FullAttention,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AttentionConfig {
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub output_gate_sigmoid: bool,
    pub bias: bool,
    pub max_position_embeddings: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RopeConfig {
    pub theta: f64,
    pub partial_rotary_factor: f64,
    pub mrope_section: Vec<u32>,
    pub mrope_interleaved: bool,
}

impl RopeConfig {
    #[must_use]
    pub fn rotary_dim(&self, head_dim: u32) -> u32 {
        (f64::from(head_dim) * self.partial_rotary_factor).round() as u32
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GdnConfig {
    pub num_key_heads: u32,
    pub num_value_heads: u32,
    pub key_head_dim: u32,
    pub value_head_dim: u32,
    pub conv_kernel: u32,
    pub ssm_dtype_f32: bool,
}

impl GdnConfig {
    #[must_use]
    pub fn qkv_rows(&self) -> u32 {
        2 * self.num_key_heads * self.key_head_dim + self.value_dim()
    }

    #[must_use]
    pub fn value_dim(&self) -> u32 {
        self.num_value_heads * self.value_head_dim
    }

    #[must_use]
    pub fn key_dim(&self) -> u32 {
        self.num_key_heads * self.key_head_dim
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MoeConfig {
    pub num_experts: u32,
    pub top_k: u32,
    pub intermediate: u32,
    pub shared_intermediate: u32,
    pub norm_topk_prob: bool,
    pub act_silu: bool,
    pub decoder_sparse_step: u32,
    pub mlp_only_layers: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MtpConfig {
    pub num_layers: u32,
    pub dedicated_embeddings: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VisionConfig {
    pub depth: u32,
    pub hidden_size: u32,
    pub num_heads: u32,
    pub intermediate_size: u32,
    pub out_hidden_size: u32,
    pub patch_size: u32,
    pub temporal_patch_size: u32,
    pub spatial_merge_size: u32,
    pub in_channels: u32,
    pub num_position_embeddings: u32,
    pub deepstack_visual_indexes: Vec<u32>,
    pub act_gelu_tanh: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantConfig {
    Bf16,
    Fp8Block128,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen35MoeConfig {
    pub hidden_size: u32,
    pub layers: Vec<LayerKind>,
    pub full_attention_interval: u32,
    pub vocab_size: u32,
    pub tie_word_embeddings: bool,
    pub rms_norm_eps: f64,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub attention: AttentionConfig,
    pub rope: RopeConfig,
    pub gdn: GdnConfig,
    pub moe: MoeConfig,
    pub mtp: MtpConfig,
    pub vision: VisionConfig,
    pub vision_tokens: [u32; 4],
    pub quant: QuantConfig,
}

impl Qwen35MoeConfig {
    #[must_use]
    pub fn num_layers(&self) -> u32 {
        self.layers.len() as u32
    }

    #[must_use]
    pub fn q_proj_rows(&self) -> u32 {
        let query_rows = self.attention.num_heads * self.attention.head_dim;
        if self.attention.output_gate_sigmoid {
            2 * query_rows
        } else {
            query_rows
        }
    }

    #[must_use]
    pub fn kv_proj_rows(&self) -> u32 {
        self.attention.num_kv_heads * self.attention.head_dim
    }

    /// Qwen3.5's input, post-attention, final and Q/K norms use `x_hat * (1 + weight)`.
    #[must_use]
    pub const fn rms_norm_weight_add(&self) -> f64 {
        1.0
    }

    /// The GDN output norm is an ordinary gated RMS norm and does not use that offset.
    #[must_use]
    pub const fn gdn_norm_weight_add(&self) -> f64 {
        0.0
    }

    pub fn validate_tp2(&self) -> Result<(), ModelError> {
        if self.attention.num_heads % 2 != 0 || self.attention.num_kv_heads % 2 != 0 {
            return Err(err("TP2 requires even attention and KV head counts"));
        }
        if self.gdn.num_key_heads % 2 != 0 || self.gdn.num_value_heads % 2 != 0 {
            return Err(err("TP2 requires even GDN key and value head counts"));
        }
        if self.moe.intermediate % 2 != 0 || self.moe.shared_intermediate % 2 != 0 {
            return Err(err("TP2 requires even routed and shared expert widths"));
        }
        Ok(())
    }

    /// Parse a BF16 `qwen3_5_moe` config, refusing unknown and unsupported fields by name.
    ///
    /// # Errors
    /// [`ModelError::Config`] for an unsupported model, or a JSON parse error.
    pub fn from_config_json(json: &str) -> Result<Self, ModelError> {
        let root: Value = serde_json::from_str(json)?;
        let mut top = Fields::new("", &root)?;

        let arch = top.array("architectures")?;
        if arch.len() != 1 || arch[0].as_str() != Some("Qwen3_5MoeForConditionalGeneration") {
            return Err(err(format!(
                "architectures: expected [\"Qwen3_5MoeForConditionalGeneration\"], got {}",
                Value::Array(arch.clone())
            )));
        }
        top.expect("model_type", &Value::from("qwen3_5_moe"))?;
        top.expect("dtype", &Value::from("bfloat16"))?;
        top.ack("transformers_version");
        let top_hidden_size = top.u32("hidden_size")?;
        let vision_tokens = [
            top.u32("image_token_id")?,
            top.u32("video_token_id")?,
            top.u32("vision_start_token_id")?,
            top.u32("vision_end_token_id")?,
        ];
        let tie_top = top.bool("tie_word_embeddings")?;

        let mut t = top.object("text_config")?;
        t.expect("model_type", &Value::from("qwen3_5_moe_text"))?;
        t.expect("dtype", &Value::from("bfloat16"))?;
        for key in [
            "attention_dropout",
            "initializer_range",
            "output_router_logits",
            "router_aux_loss_coef",
            "use_cache",
        ] {
            t.ack(key);
        }
        match t.value("pad_token_id")? {
            Value::Null => {}
            value => {
                return Err(err(format!(
                    "text_config.pad_token_id: expected null, got {value}"
                )))
            }
        }

        let hidden_size = t.u32("hidden_size")?;
        if hidden_size != top_hidden_size {
            return Err(err(format!(
                "text_config.hidden_size: {hidden_size}, top-level hidden_size is {top_hidden_size}"
            )));
        }
        let num_layers = t.u32("num_hidden_layers")?;
        let interval = t.u32("full_attention_interval")?;
        if num_layers == 0 || interval == 0 {
            return Err(err(
                "text_config.num_hidden_layers / full_attention_interval: must be nonzero",
            ));
        }
        let types = t.array("layer_types")?;
        if types.len() != num_layers as usize {
            return Err(err(format!(
                "text_config.layer_types: {} entries, num_hidden_layers is {num_layers}",
                types.len()
            )));
        }
        let mut layers = Vec::with_capacity(types.len());
        for (index, value) in types.iter().enumerate() {
            let path = format!("text_config.layer_types[{index}]");
            let kind = match value.as_str() {
                Some("linear_attention") => LayerKind::Gdn,
                Some("full_attention") => LayerKind::FullAttention,
                Some(other) => return Err(err(format!("{path}: unknown layer type `{other}`"))),
                None => return Err(err(format!("{path}: expected a string"))),
            };
            let wanted = if (index as u32 + 1) % interval == 0 {
                LayerKind::FullAttention
            } else {
                LayerKind::Gdn
            };
            if kind != wanted {
                return Err(err(format!(
                    "{path}: {kind:?}, but full_attention_interval {interval} puts {wanted:?} there"
                )));
            }
            layers.push(kind);
        }

        let vocab_size = t.u32("vocab_size")?;
        let tie_text = t.bool("tie_word_embeddings")?;
        if tie_text != tie_top {
            return Err(err(format!(
                "tie_word_embeddings: top level {tie_top}, text_config {tie_text}"
            )));
        }
        if tie_text {
            return Err(err(
                "tie_word_embeddings: the initial pack supports the released untied head only",
            ));
        }
        let rms_norm_eps = t.f64("rms_norm_eps")?;
        if rms_norm_eps <= 0.0 {
            return Err(err("text_config.rms_norm_eps: must be positive"));
        }
        let bos_token_id = t.u32("bos_token_id")?;
        let eos_token_id = t.u32("eos_token_id")?;
        if t.str("hidden_act")? != "silu" {
            return Err(err("text_config.hidden_act: only `silu` is supported"));
        }

        let output_gate_sigmoid = t.bool("attn_output_gate")?;
        if !output_gate_sigmoid {
            return Err(err(
                "text_config.attn_output_gate: the initial graph requires the sigmoid output gate",
            ));
        }
        let attention = AttentionConfig {
            num_heads: t.u32("num_attention_heads")?,
            num_kv_heads: t.u32("num_key_value_heads")?,
            head_dim: t.u32("head_dim")?,
            output_gate_sigmoid,
            bias: t.bool("attention_bias")?,
            max_position_embeddings: t.u32("max_position_embeddings")?,
        };
        if attention.bias {
            return Err(err("text_config.attention_bias: true is not supported"));
        }
        if attention.num_heads == 0
            || attention.num_kv_heads == 0
            || attention.num_heads % attention.num_kv_heads != 0
            || attention.head_dim == 0
        {
            return Err(err(format!(
                "text_config attention geometry: {} query heads, {} KV heads, head_dim {}",
                attention.num_heads, attention.num_kv_heads, attention.head_dim
            )));
        }

        let partial_rotary_factor = t.f64("partial_rotary_factor")?;
        let mut rp = t.object("rope_parameters")?;
        let rope = RopeConfig {
            theta: rp.f64("rope_theta")?,
            partial_rotary_factor: rp.f64("partial_rotary_factor")?,
            mrope_section: rp.u32_array("mrope_section")?,
            mrope_interleaved: rp.bool("mrope_interleaved")?,
        };
        if rp.str("rope_type")? != "default" {
            return Err(err("text_config.rope_parameters.rope_type: only `default`"));
        }
        rp.finish()?;
        if rope.partial_rotary_factor != partial_rotary_factor {
            return Err(err(format!(
                "partial_rotary_factor: text_config {partial_rotary_factor}, rope_parameters {}",
                rope.partial_rotary_factor
            )));
        }
        let rotary_dim = rope.rotary_dim(attention.head_dim);
        if rotary_dim == 0 || rotary_dim % 2 != 0 || rotary_dim > attention.head_dim {
            return Err(err(format!(
                "text_config.partial_rotary_factor: rotary dim {rotary_dim} is invalid for head_dim {}",
                attention.head_dim
            )));
        }
        if rope.mrope_section.len() != 3 || rope.mrope_section.iter().sum::<u32>() != rotary_dim / 2
        {
            return Err(err(format!(
                "text_config.rope_parameters.mrope_section: {:?} must have three entries summing to {}",
                rope.mrope_section,
                rotary_dim / 2
            )));
        }
        if !rope.mrope_interleaved {
            return Err(err(
                "text_config.rope_parameters.mrope_interleaved: only interleaved M-RoPE is supported",
            ));
        }

        let gdn = GdnConfig {
            num_key_heads: t.u32("linear_num_key_heads")?,
            num_value_heads: t.u32("linear_num_value_heads")?,
            key_head_dim: t.u32("linear_key_head_dim")?,
            value_head_dim: t.u32("linear_value_head_dim")?,
            conv_kernel: t.u32("linear_conv_kernel_dim")?,
            ssm_dtype_f32: match t.str("mamba_ssm_dtype")? {
                "float32" => true,
                other => {
                    return Err(err(format!(
                        "text_config.mamba_ssm_dtype: `{other}` (the checkpoint's state is float32)"
                    )))
                }
            },
        };
        if gdn.num_key_heads == 0
            || gdn.num_value_heads == 0
            || gdn.num_value_heads % gdn.num_key_heads != 0
        {
            return Err(err(format!(
                "text_config GDN heads: {} value heads must be a nonzero multiple of {} key heads",
                gdn.num_value_heads, gdn.num_key_heads
            )));
        }
        if gdn.key_head_dim != 128 || gdn.value_head_dim != 128 || gdn.conv_kernel != 4 {
            return Err(err(
                "text_config GDN geometry: the initial kernels require key/value head_dim 128 and conv width 4",
            ));
        }

        let moe = MoeConfig {
            num_experts: t.u32("num_experts")?,
            top_k: t.u32("num_experts_per_tok")?,
            intermediate: t.u32("moe_intermediate_size")?,
            shared_intermediate: t.u32("shared_expert_intermediate_size")?,
            norm_topk_prob: declared_bool(&mut t, "norm_topk_prob", true)?,
            act_silu: true,
            decoder_sparse_step: match t.optional("decoder_sparse_step") {
                None => 1,
                Some(value) => value
                    .as_u64()
                    .and_then(|x| u32::try_from(x).ok())
                    .ok_or_else(|| {
                        err(format!(
                            "text_config.decoder_sparse_step: expected an integer, got {value}"
                        ))
                    })?,
            },
            mlp_only_layers: match t.optional("mlp_only_layers") {
                None | Some(Value::Null) => Vec::new(),
                Some(_) => t.u32_array("mlp_only_layers")?,
            },
        };
        if moe.decoder_sparse_step != 1 || !moe.mlp_only_layers.is_empty() {
            return Err(err(
                "text_config.decoder_sparse_step / mlp_only_layers: only an all-MoE stack is supported",
            ));
        }
        if moe.num_experts == 0
            || moe.top_k == 0
            || moe.top_k > moe.num_experts
            || moe.intermediate == 0
            || moe.shared_intermediate == 0
        {
            return Err(err(
                "text_config MoE geometry: invalid expert count, top-k or width",
            ));
        }

        let mtp = MtpConfig {
            num_layers: t.u32("mtp_num_hidden_layers")?,
            dedicated_embeddings: t.bool("mtp_use_dedicated_embeddings")?,
        };
        if mtp.num_layers != 0 || mtp.dedicated_embeddings {
            return Err(err(
                "text_config MTP: the initial port requires mtp_num_hidden_layers 0 and shared embeddings",
            ));
        }
        t.finish()?;

        let mut v = top.object("vision_config")?;
        v.expect("model_type", &Value::from("qwen3_5_moe_vision"))?;
        v.expect("dtype", &Value::from("bfloat16"))?;
        v.ack("initializer_range");
        let vision = VisionConfig {
            depth: v.u32("depth")?,
            hidden_size: v.u32("hidden_size")?,
            num_heads: v.u32("num_heads")?,
            intermediate_size: v.u32("intermediate_size")?,
            out_hidden_size: v.u32("out_hidden_size")?,
            patch_size: v.u32("patch_size")?,
            temporal_patch_size: v.u32("temporal_patch_size")?,
            spatial_merge_size: v.u32("spatial_merge_size")?,
            in_channels: v.u32("in_channels")?,
            num_position_embeddings: v.u32("num_position_embeddings")?,
            deepstack_visual_indexes: v.u32_array("deepstack_visual_indexes")?,
            act_gelu_tanh: v.str("hidden_act")? == "gelu_pytorch_tanh",
        };
        if !vision.act_gelu_tanh {
            return Err(err("vision_config.hidden_act: only `gelu_pytorch_tanh`"));
        }
        if vision.out_hidden_size != hidden_size {
            return Err(err(
                "vision_config.out_hidden_size: must equal the text hidden size",
            ));
        }
        if !vision.deepstack_visual_indexes.is_empty() {
            return Err(err(
                "vision_config.deepstack_visual_indexes: the released tower has no deepstack mergers",
            ));
        }
        v.finish()?;

        let quant = if let Some(value) = top.optional("quantization_config") {
            let mut q = Fields::new("quantization_config", value)?;
            q.expect("quant_method", &Value::from("fp8"))?;
            q.expect("fmt", &Value::from("e4m3"))?;
            q.expect("activation_scheme", &Value::from("dynamic"))?;
            if q.u32_array("weight_block_size")? != [128, 128] {
                return Err(err(
                    "quantization_config.weight_block_size: Occamy FP8 requires [128, 128]",
                ));
            }
            q.ack("modules_to_not_convert");
            q.finish()?;
            QuantConfig::Fp8Block128
        } else {
            QuantConfig::Bf16
        };
        top.finish()?;

        let config = Self {
            hidden_size,
            layers,
            full_attention_interval: interval,
            vocab_size,
            tie_word_embeddings: tie_top,
            rms_norm_eps,
            bos_token_id,
            eos_token_id,
            attention,
            rope,
            gdn,
            moe,
            mtp,
            vision,
            vision_tokens,
            quant,
        };
        config.validate_tp2()?;
        Ok(config)
    }
}

fn declared_bool(fields: &mut Fields<'_>, key: &str, default: bool) -> Result<bool, ModelError> {
    match fields.optional(key) {
        None => Ok(default),
        Some(value) => value.as_bool().ok_or_else(|| {
            err(format!(
                "text_config.{key}: expected a boolean, got {value}"
            ))
        }),
    }
}
