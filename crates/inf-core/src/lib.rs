//! Core primitives shared across the engine.
//!
//! Today this crate owns the **compute job system** ([`job`]) — the single
//! Ring-0 entry point for CPU data-parallelism (a rayon worker pool + flume
//! channels; §2.5 of the roadmap). Other low-level primitives (ids, error
//! plumbing, the frame clock) land here alongside their first real consumers.
//!
//! The job system is re-exported at the crate root for ergonomics:
//!
//! ```
//! // Deterministic, in-order parallel map on the process-wide pool.
//! let doubled = inf_core::parallel_map(vec![1, 2, 3], |x| x * 2);
//! assert_eq!(doubled, vec![2, 4, 6]);
//! ```

pub mod job;

pub use job::{
    bounded_channel, channel, global, join, parallel_for, parallel_map, parallel_map_ref, scope,
    JobPool,
};
