use std::collections::{BTreeMap, BTreeSet};

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
pub struct QuantSpec {
    pub dynamic: bool,
    pub num_bits: u32,
    #[serde(rename = "type")]
    pub kind: String,
    pub group_size: u32,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct QuantGroup {
    pub input_activations: QuantSpec,
    pub weights: QuantSpec,
    pub targets: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct QuantProducer {
    pub name: String,
    pub version: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct QuantizationConfig {
    pub config_groups: BTreeMap<String, QuantGroup>,
    pub ignore: Vec<String>,
    pub quant_algo: String,
    pub producer: QuantProducer,
    pub quant_method: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RippleConfig {
    pub architectures: Vec<String>,
    pub auto_map: BTreeMap<String, String>,
    pub bos_token_id: u32,
    pub dtype: Option<String>,
    pub embed_scale: Option<f64>,
    pub eos_token_id: Vec<u32>,
    pub ffn_clamp_threshold: Option<f64>,
    pub gate_full_precision: bool,
    pub head_dim: u32,
    pub hidden_act: String,
    pub hidden_size: u32,
    pub hybrid_layer_pattern: Vec<u32>,
    pub intermediate_size: u32,
    pub is_hybrid_swa: bool,
    pub layer_types: Vec<LayerKind>,
    pub max_position_embeddings: u32,
    pub model_type: String,
    pub num_attention_heads: u32,
    pub num_hidden_layers: u32,
    pub num_key_value_heads: u32,
    pub pad_token_id: u32,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub sliding_window: u32,
    pub tie_word_embeddings: bool,
    pub transformers_version: String,
    pub use_cache: bool,
    pub use_gated_attention: bool,
    pub use_valres: bool,
    pub vocab_size: u32,
    pub quantization_config: Option<QuantizationConfig>,
}

impl RippleConfig {
    pub fn from_config_json(json: &str) -> Result<Self, ModelError> {
        let cfg: Self = serde_json::from_str(json)?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        if self.model_type != "ricursive" {
            return self.fail(format!(
                "model_type must be ricursive, got {}",
                self.model_type
            ));
        }
        if self.architectures != ["RicursiveForCausalLM"] {
            return self.fail("architectures must contain only RicursiveForCausalLM");
        }
        if self.hidden_act != "silu" {
            return self.fail(format!("hidden_act must be silu, got {}", self.hidden_act));
        }
        if self.dtype.as_deref().is_some_and(|d| d != "bfloat16") {
            return self.fail("dtype must be bfloat16 when present");
        }
        if !self.use_cache
            || !self.use_gated_attention
            || !self.gate_full_precision
            || !self.use_valres
        {
            return self.fail(
                "Ripple serving requires use_cache, gated attention, full-precision gates, and value residuals",
            );
        }
        if self.tie_word_embeddings {
            return self.fail("tied embeddings are not supported by this checkpoint layout");
        }
        for (name, value) in [
            ("hidden_size", self.hidden_size),
            ("head_dim", self.head_dim),
            ("intermediate_size", self.intermediate_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("sliding_window", self.sliding_window),
            ("max_position_embeddings", self.max_position_embeddings),
            ("vocab_size", self.vocab_size),
        ] {
            if value == 0 {
                return self.fail(format!("{name} must be nonzero"));
            }
        }
        if self.head_dim % 2 != 0 {
            return self.fail("head_dim must be even for RoPE");
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return self.fail("num_attention_heads must be divisible by num_key_value_heads");
        }
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return self.fail("rms_norm_eps must be finite and positive");
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return self.fail("rope_theta must be finite and positive");
        }
        if self.embed_scale.is_some_and(|v| !v.is_finite() || v <= 0.0) {
            return self.fail("embed_scale must be finite and positive when present");
        }
        if self
            .ffn_clamp_threshold
            .is_some_and(|v| !v.is_finite() || v <= 0.0)
        {
            return self.fail("ffn_clamp_threshold must be finite and positive when present");
        }
        if self.layer_types.len() != self.num_hidden_layers as usize {
            return self.fail(format!(
                "layer_types has {} entries, expected {}",
                self.layer_types.len(),
                self.num_hidden_layers
            ));
        }
        if self.hybrid_layer_pattern.len() != self.layer_types.len() {
            return self.fail("hybrid_layer_pattern must have one entry per layer");
        }
        for (i, (&kind, &pattern)) in self
            .layer_types
            .iter()
            .zip(&self.hybrid_layer_pattern)
            .enumerate()
        {
            let expected = u32::from(kind == LayerKind::SlidingAttention);
            if pattern != expected {
                return self.fail(format!(
                    "hybrid_layer_pattern[{i}] must be {expected}, got {pattern}"
                ));
            }
        }
        let hybrid = self.layer_types.contains(&LayerKind::SlidingAttention)
            && self.layer_types.contains(&LayerKind::FullAttention);
        if self.is_hybrid_swa != hybrid {
            return self.fail("is_hybrid_swa disagrees with layer_types");
        }
        for (name, id) in [
            ("bos_token_id", self.bos_token_id),
            ("pad_token_id", self.pad_token_id),
        ] {
            if id >= self.vocab_size {
                return self.fail(format!("{name} is outside vocab_size"));
            }
        }
        if self.eos_token_id.is_empty() || self.eos_token_id.iter().any(|&id| id >= self.vocab_size)
        {
            return self.fail("eos_token_id must contain valid vocabulary ids");
        }
        if let Some(q) = &self.quantization_config {
            self.validate_quantization(q)?;
        }
        Ok(())
    }

    fn validate_quantization(&self, q: &QuantizationConfig) -> Result<(), ModelError> {
        if q.quant_method != "modelopt" || q.quant_algo != "NVFP4" {
            return self.fail("only ModelOpt NVFP4 quantization is supported");
        }
        if q.producer.name != "modelopt" {
            return self.fail("quantization producer must be modelopt");
        }
        if q.config_groups.len() != 1 || !q.config_groups.contains_key("group_0") {
            return self.fail("NVFP4 must declare exactly config_groups.group_0");
        }
        let group = &q.config_groups["group_0"];
        if group.targets != ["Linear"] {
            return self.fail("NVFP4 group_0 targets must contain only Linear");
        }
        for (name, spec) in [
            ("weights", &group.weights),
            ("input_activations", &group.input_activations),
        ] {
            if spec.dynamic || spec.num_bits != 4 || spec.kind != "float" || spec.group_size != 16 {
                return self.fail(format!(
                    "quantization_config.{name} must be static float4 with group_size 16"
                ));
            }
        }
        let expected: BTreeSet<String> = std::iter::once("lm_head".to_owned())
            .chain(std::iter::once("model.embed_tokens".to_owned()))
            .chain((0..self.num_hidden_layers).map(|i| format!("model.layers.{i}.self_attn*")))
            .collect();
        let actual: BTreeSet<String> = q.ignore.iter().cloned().collect();
        if actual != expected || actual.len() != q.ignore.len() {
            return self.fail(
                "NVFP4 ignore must name lm_head, model.embed_tokens, and every self_attn exactly once",
            );
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
            ("intermediate_size", self.intermediate_size),
            ("vocab_size", self.vocab_size),
        ] {
            if value % tp != 0 {
                return self.fail(format!("{name}={value} is not divisible by tp={tp}"));
            }
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
    pub fn nvfp4_mlp(&self) -> bool {
        self.quantization_config.is_some()
    }

    #[must_use]
    pub fn kv_cache_bytes_total(&self, context: u32, element_bytes: u64) -> u64 {
        let full = u64::from(self.full_attention_layers()) * u64::from(context);
        let local = u64::from(self.num_hidden_layers - self.full_attention_layers())
            * u64::from(context.min(self.sliding_window));
        let kv_elements = u64::from(self.num_key_value_heads) * u64::from(self.head_dim) * 2;
        (full + local) * kv_elements * element_bytes
    }

    fn fail<T>(&self, msg: impl Into<String>) -> Result<T, ModelError> {
        Err(ModelError::Config(msg.into()))
    }
}
