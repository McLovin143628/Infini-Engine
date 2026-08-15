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
pub mod log;

/// The composed-frame budget, in milliseconds — the §8 verification-strategy
/// figure every phase gate measures against.
///
/// It lives here, in the crate everything depends on, because it is a **policy
/// number**: `inf-render`'s frame-budget suite, the Phase 18 gate and the Phase
/// 19 gate all assert against it, and three private copies of a policy number is
/// three places for it to drift — or, worse, three places for somebody to raise
/// one of them instead of investigating a regression. §8's rule is that a gate
/// over budget is a bug to fix, never a constant to raise; keeping one copy is
/// what makes that rule enforceable by review.
///
/// 33 ms is a 30 fps floor. It is deliberately *not* the 16.7 ms of a 60 fps
/// target: these gates run on CI machines with software adapters, and a budget
/// nobody can meet is a budget everybody disables.
pub const FRAME_BUDGET_MS: f64 = 33.0;

pub use job::{
    bounded_channel, channel, global, join, parallel_for, parallel_map, parallel_map_ref, scope,
    JobPool,
};
pub use log::{BoundedLog, DEFAULT_LOG_CAPACITY};
