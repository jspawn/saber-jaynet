use serde::Deserialize;

use crate::ModelError;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LayerKind {
    SlidingAttention,
    FullAttention,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct QuantizationConfig {
    pub quant_method: String,
    pub activation_scheme: String,
    pub weight_block_size: [u32; 2],
    pub modules_to_not_convert: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct KolibriConfig {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub hidden_size: u32,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    pub hidden_act: String,
    pub max_position_embeddings: u32,
    pub rms_norm_eps: f64,
    pub vocab_size: u32,
    pub rope_theta: f64,
    pub num_experts: u32,
    pub num_experts_per_tok: u32,
    pub moe_intermediate_size: u32,
    pub shared_expert_intermediate_size: u32,
    pub norm_topk_prob: bool,
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub tie_word_embeddings: bool,
    pub transformers_version: String,
    pub use_cache: bool,
    pub use_sliding_window: bool,
    pub sliding_window: u32,
    pub layer_types: Vec<LayerKind>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
    pub dtype: String,
    pub head_dtype: String,
    pub quantization_config: Option<QuantizationConfig>,
}

impl KolibriConfig {
    pub fn from_config_json(json: &str) -> Result<Self, ModelError> {
        let cfg: Self = serde_json::from_str(json)?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        if self.model_type != "kolibri1" || self.architectures != ["Kolibri1ForCausalLM"] {
            return self.fail("expected Kolibri1ForCausalLM / kolibri1");
        }
        if self.hidden_act != "silu" || self.dtype != "bfloat16" || self.head_dtype != "float32" {
            return self.fail("Kolibri requires SiLU, bfloat16 storage and a float32 head");
        }
        if !self.use_cache
            || !self.use_sliding_window
            || self.norm_topk_prob
            || self.attention_bias
            || self.attention_dropout != 0.0
            || self.tie_word_embeddings
        {
            return self.fail("unsupported attention, cache, routing or embedding option");
        }
        for (name, value) in [
            ("hidden_size", self.hidden_size),
            ("head_dim", self.head_dim),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("num_experts", self.num_experts),
            ("num_experts_per_tok", self.num_experts_per_tok),
            ("moe_intermediate_size", self.moe_intermediate_size),
            (
                "shared_expert_intermediate_size",
                self.shared_expert_intermediate_size,
            ),
            ("sliding_window", self.sliding_window),
            ("max_position_embeddings", self.max_position_embeddings),
            ("vocab_size", self.vocab_size),
        ] {
            if value == 0 {
                return self.fail(format!("{name} must be nonzero"));
            }
        }
        if self.head_dim != 128 || self.head_dim % 2 != 0 {
            return self.fail("the initial Kolibri port requires even head_dim 128");
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return self.fail("attention heads must be divisible by KV heads");
        }
        if self.num_experts_per_tok > self.num_experts {
            return self.fail("num_experts_per_tok exceeds num_experts");
        }
        if !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 0.0
        {
            return self.fail("RMS epsilon and RoPE theta must be finite and positive");
        }
        if self.sliding_window > self.max_position_embeddings {
            return self.fail("sliding_window exceeds max_position_embeddings");
        }
        if self.layer_types.len() != self.num_hidden_layers as usize {
            return self.fail(format!(
                "layer_types has {} entries, expected {}",
                self.layer_types.len(),
                self.num_hidden_layers
            ));
        }
        for (index, kind) in self.layer_types.iter().enumerate() {
            let expected = if index % 5 == 4 {
                LayerKind::FullAttention
            } else {
                LayerKind::SlidingAttention
            };
            if *kind != expected {
                return self.fail(format!("layer_types[{index}] breaks the 4:1 schedule"));
            }
        }
        for (name, id) in [
            ("eos_token_id", self.eos_token_id),
            ("pad_token_id", self.pad_token_id),
        ] {
            if id >= self.vocab_size || self.bos_token_id.is_some_and(|bos| bos >= self.vocab_size)
            {
                return self.fail(format!("{name} is outside vocab_size"));
            }
        }
        if let Some(quant) = &self.quantization_config {
            if quant.quant_method != "fp8"
                || quant.activation_scheme != "dynamic"
                || quant.weight_block_size != [128, 128]
            {
                return self.fail("only dynamic FP8 with 128x128 weight blocks is supported");
            }
            let expected: Vec<_> = std::iter::once("lm_head".to_owned())
                .chain(
                    (0..self.num_hidden_layers)
                        .map(|layer| format!("model.layers.{layer}.mlp.gate")),
                )
                .collect();
            if quant.modules_to_not_convert != expected {
                return self.fail("FP8 exclusions must name lm_head and every MoE router in order");
            }
        }
        Ok(())
    }

    pub fn validate_tp(&self, tp: u32) -> Result<(), ModelError> {
        if tp == 0 {
            return self.fail("tp must be nonzero");
        }
        for (name, value) in [
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("vocab_size", self.vocab_size),
            ("moe_intermediate_size", self.moe_intermediate_size),
            (
                "shared_expert_intermediate_size",
                self.shared_expert_intermediate_size,
            ),
        ] {
            if value % tp != 0 {
                return self.fail(format!("{name}={value} is not divisible by tp={tp}"));
            }
        }
        if self.q_proj_rows() % (128 * tp) != 0
            || self.kv_proj_rows() % (128 * tp) != 0
            || self.moe_intermediate_size % (128 * tp) != 0
        {
            return self.fail("TP shards must preserve 128-wide FP8 blocks");
        }
        Ok(())
    }

    #[must_use]
    pub fn layer_kind(&self, layer: u32) -> LayerKind {
        self.layer_types[layer as usize]
    }

    #[must_use]
    pub fn uses_rope(&self, layer: u32) -> bool {
        self.layer_kind(layer) == LayerKind::SlidingAttention
    }

    #[must_use]
    pub fn full_attention_layers(&self) -> u32 {
        self.layer_types
            .iter()
            .filter(|&&kind| kind == LayerKind::FullAttention)
            .count() as u32
    }

    #[must_use]
    pub fn q_proj_rows(&self) -> u32 {
        self.num_attention_heads * self.head_dim
    }

    #[must_use]
    pub fn kv_proj_rows(&self) -> u32 {
        self.num_key_value_heads * self.head_dim
    }

    #[must_use]
    pub fn kv_cache_bytes_total(&self, context: u32, element_bytes: u64) -> u64 {
        let full = u64::from(self.full_attention_layers()) * u64::from(context);
        let local = u64::from(self.num_hidden_layers - self.full_attention_layers())
            * u64::from(context.min(self.sliding_window));
        (full + local)
            * u64::from(self.num_key_value_heads)
            * u64::from(self.head_dim)
            * 2
            * element_bytes
    }

    fn fail<T>(&self, message: impl Into<String>) -> Result<T, ModelError> {
        Err(ModelError::Config(message.into()))
    }
}
