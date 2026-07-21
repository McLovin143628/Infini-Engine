//! The domain value trait: what flows on a wire at runtime.
//!
//! `inf-graph` is agnostic about the values its nodes produce — a blueprint
//! data graph passes `f64`/`bool`/`Entity`; a material graph would pass WGSL
//! expression handles. The domain supplies the concrete type and its RAM cost
//! (so the [`NodeCache`](crate::cache::NodeCache) can bound memory).

/// A runtime value carried between nodes. Cheap to clone (values are cached by
/// `Arc`-style sharing in richer domains; scalars are `Copy`).
pub trait GraphValue: Clone {
    /// Approximate in-RAM footprint in bytes, for cache eviction accounting.
    fn ram_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}
