//! Qwen3.5 MoE (`model_type qwen3_5_moe`), starting with Occamy's BF16 checkpoint.

pub mod config;
pub mod pack_plan;
pub mod spike;

pub use config::{LayerKind, QuantConfig, Qwen35MoeConfig};
