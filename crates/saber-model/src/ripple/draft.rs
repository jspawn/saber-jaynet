use std::collections::BTreeSet;

use serde::Deserialize;

use crate::dflash_graph::{
    Checkpoint, DflashConfig, DraftGraph, DraftQuant, DraftRecord, DraftRope,
};
use crate::graph::{
    Dtype, LAYOUT_BF16_FROM_FP8_BLOCK, LAYOUT_FP8_BLOCK_SCALE, LAYOUT_FP8_PRESHUFFLE,
    LAYOUT_INT2_HEAD, LAYOUT_INT2_HEAD_SCALE, LAYOUT_INT2_HEAD_ZERO,
};
use crate::ModelError;

use super::{LayerKind, RippleConfig};

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DflashSettings {
    pub block_size: u32,
    pub conv_group_size: u32,
    pub conv_kernel_size: u32,
    pub mask_token_id: u32,
    pub selector_rank: u32,
    pub selector_top_k: u32,
    pub target_layer_ids: Vec<u32>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RopeParameters {
    pub rope_theta: f64,
    pub rope_type: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RippleDraftConfig {
    pub architectures: Vec<String>,
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub bos_token_id: u32,
    pub dflash_config: DflashSettings,
    pub draft_vocab_size: u32,
    pub dtype: String,
    pub eos_token_id: Vec<u32>,
    pub head_dim: u32,
    pub hidden_act: String,
    pub hidden_size: u32,
    pub initializer_range: f64,
    pub intermediate_size: u32,
    pub is_causal: bool,
    pub layer_types: Vec<LayerKind>,
    pub max_position_embeddings: u32,
    pub max_window_layers: u32,
    pub model_type: String,
    pub no_rope_layers: Vec<u32>,
    pub num_attention_heads: u32,
    pub num_hidden_layers: u32,
    pub num_key_value_heads: u32,
    pub num_target_layers: u32,
    pub pad_token_id: u32,
    pub rms_norm_eps: f64,
    pub rope_parameters: RopeParameters,
    pub rope_theta: f64,
    pub sliding_window: u32,
    pub tie_word_embeddings: bool,
    pub transformers_version: String,
    pub use_cache: bool,
    pub use_sliding_window: bool,
    pub vocab_size: u32,
}

impl RippleDraftConfig {
    pub fn from_config_json(json: &str, target: &RippleConfig) -> Result<Self, ModelError> {
        let cfg: Self = serde_json::from_str(json)?;
        cfg.validate(target)?;
        Ok(cfg)
    }

    fn validate(&self, target: &RippleConfig) -> Result<(), ModelError> {
        let fail = |msg: &str| Err(ModelError::Config(format!("Ripple DFlash2: {msg}")));
        if self.architectures != ["DFlash2DraftModel"] || self.model_type != "qwen3" {
            return fail("unexpected architecture or model_type");
        }
        if self.dtype != "bfloat16" || self.hidden_act != "silu" || self.attention_bias {
            return fail("expected BF16 SwiGLU layers without attention bias");
        }
        if self.is_causal || !self.use_cache || !self.use_sliding_window {
            return fail("the drafter must be non-causal with its sliding-window cache enabled");
        }
        if self.num_hidden_layers != 5
            || self.layer_types.len() != 5
            || self.max_window_layers != 5
            || self.layer_types[..4] != [LayerKind::SlidingAttention; 4]
            || self.layer_types[4] != LayerKind::FullAttention
            || self.no_rope_layers != [4]
        {
            return fail("expected four sliding layers and one global NoPE layer");
        }
        if self.hidden_size != target.hidden_size
            || self.intermediate_size != target.intermediate_size
            || self.num_attention_heads != target.num_attention_heads
            || self.num_key_value_heads != target.num_key_value_heads
            || self.head_dim != target.head_dim
            || self.num_target_layers != target.num_hidden_layers
            || self.vocab_size != target.vocab_size
            || self.draft_vocab_size != target.vocab_size
            || self.rope_theta != target.rope_theta
            || self.rope_parameters.rope_theta != target.rope_theta
        {
            return fail("target and drafter geometry disagree");
        }
        if self.rope_parameters.rope_type != "default" || self.sliding_window != 4096 {
            return fail("expected default RoPE and a 4096-token local window");
        }
        let d = &self.dflash_config;
        if d.block_size != 16
            || d.conv_kernel_size != 4
            || d.conv_group_size != 16
            || d.selector_rank != 256
            || d.selector_top_k != 32
            || d.target_layer_ids != [1, 20, 39, 58, 77]
        {
            return fail("unexpected block, convolution, selector, or target taps");
        }
        if d.mask_token_id >= self.vocab_size {
            return fail("mask_token_id is outside the draft vocabulary");
        }
        Ok(())
    }

    #[must_use]
    pub fn dflash_config(&self, tp: u32) -> DflashConfig {
        DflashConfig {
            tp,
            hidden: self.hidden_size,
            layers: self.num_hidden_layers,
            intermediate: self.intermediate_size,
            q_heads: self.num_attention_heads,
            kv_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            block_size: self.dflash_config.block_size,
            conv_taps: self.dflash_config.conv_kernel_size,
            conv_group: self.dflash_config.conv_group_size,
            selector_rank: self.dflash_config.selector_rank,
            selector_top_k: self.dflash_config.selector_top_k,
            target_layers: self.dflash_config.target_layer_ids.len() as u32,
            vocab: self.vocab_size,
            rerank: 80,
            head_block_n: 64,
            head_kcand: 8,
            iu8: false,
        }
    }
}

pub fn validate_draft_tensors<'a>(
    cfg: &RippleDraftConfig,
    tp: u32,
    max_positions: u32,
    names: impl IntoIterator<Item = &'a str>,
) -> Result<(), ModelError> {
    let graph = DraftGraph::build(cfg.dflash_config(tp));
    let records = graph.records(DraftRope::qwen38_dflash2(max_positions));
    let expected: BTreeSet<String> = records
        .iter()
        .flat_map(|record| record.sources.iter())
        .filter(|source| source.checkpoint == Checkpoint::Drafter)
        .map(|source| source.weight.clone())
        .collect();
    let actual: BTreeSet<String> = names.into_iter().map(str::to_owned).collect();
    if actual != expected {
        let missing: Vec<_> = expected.difference(&actual).collect();
        let unused: Vec<_> = actual.difference(&expected).collect();
        return Err(ModelError::Config(format!(
            "Ripple DFlash2 tensor coverage: missing {missing:?}; unused {unused:?}"
        )));
    }
    Ok(())
}

pub fn draft_pack_records(
    cfg: &RippleDraftConfig,
    tp: u32,
    max_positions: u32,
) -> Vec<DraftRecord> {
    let graph = DraftGraph::build(cfg.dflash_config(tp));
    let rope = DraftRope {
        theta: cfg.rope_theta,
        rotary_dim: cfg.head_dim,
        max_pos: max_positions,
    };
    let mut records = graph.records(rope);
    for record in &mut records {
        for source in &mut record.sources {
            if source.checkpoint == Checkpoint::Drafter && source.quant == DraftQuant::Fp8Block {
                source.quant = DraftQuant::Bf16;
                source.scale = None;
            }
            if source.checkpoint == Checkpoint::Target {
                source.quant = DraftQuant::Bf16;
                source.scale = None;
                if source.weight == "model.language_model.embed_tokens.weight" {
                    source.weight = "model.embed_tokens.weight".into();
                }
            }
        }
    }
    records
}

pub fn draft_runtime_pack_records(
    cfg: &RippleDraftConfig,
    tp: u32,
    max_positions: u32,
    bf16: bool,
) -> Vec<DraftRecord> {
    draft_pack_records(cfg, tp, max_positions)
        .into_iter()
        .filter(|record| {
            record.name != "input_embedding.weight"
                && !record.name.starts_with("head.rerank.")
                && ![
                    LAYOUT_INT2_HEAD,
                    LAYOUT_INT2_HEAD_SCALE,
                    LAYOUT_INT2_HEAD_ZERO,
                ]
                .contains(&record.layout_id)
        })
        .filter(|record| !bf16 || record.layout_id != LAYOUT_FP8_BLOCK_SCALE)
        .map(|mut record| {
            if bf16 && record.layout_id == LAYOUT_FP8_PRESHUFFLE {
                record.layout_id = LAYOUT_BF16_FROM_FP8_BLOCK;
                record.dtype = Dtype::Bf16;
                record.shape = vec![record.shape[0] * 16, record.shape[1] / 16];
            }
            record
        })
        .collect()
}
