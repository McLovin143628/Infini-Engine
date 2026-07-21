//! The material node graph → WGSL (P7.2).
//!
//! A `.inf_mat` material is authored as a pure-data node DAG over the shared
//! [`inf_graph`] substrate — the same substrate the Blueprint editor uses, with
//! a material-specific [`nodekit::material_registry`] and no exec pins. A single
//! `output.surface` SINK node collects the PBR channels; [`emit::emit_wgsl`]
//! walks the compiled graph to generate a `material_surface` WGSL function and
//! validates it with naga, mapping errors back onto the offending nodes.

pub mod emit;
pub mod nodekit;

pub use emit::{
    emit_texture_compute, emit_wgsl, MatIssue, MatSeverity, MaterialCompile, TextureBinding,
    TextureCompile,
};
pub use nodekit::{material_registry, MatType};
