//! The engine's compute job system — the single Ring-0 entry point for CPU
//! data-parallelism (§2.5 of the roadmap).
//!
//! Doctrine: **rayon for compute, flume for hand-off, tokio never enters here.**
//! A [`JobPool`] wraps a rayon worker pool with named threads sized to the
//! machine; gameplay/render/asset code reaches for [`parallel_for`] /
//! [`parallel_map`] / [`scope`] rather than spawning raw `std::thread`s or
//! pulling rayon in directly. Keeping the pool behind one crate means the whole
//! engine shares one set of workers (no oversubscription) and one place to size,
//! name, and later instrument them.
//!
//! ## Determinism
//!
//! [`parallel_map`] preserves input order in its output regardless of how many
//! worker threads run it — it is a parallel *pure map*, so the result is a
//! function of the input alone. This is the property the P7.0 determinism guard
//! pins (`same input → same output regardless of pool size`) and the reason the
//! parallel asset-import and graph-compile paths can use the pool without
//! threatening the engine's bit-determinism guarantees.

use std::sync::OnceLock;

use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

/// A named rayon worker pool. Construct one explicitly for a subsystem that
/// wants an isolated pool, or reach for the process-wide [`global`] pool via the
/// free functions ([`parallel_for`], [`parallel_map`], [`join`], [`scope`]).
pub struct JobPool {
    pool: ThreadPool,
}

impl JobPool {
    /// Build a pool with `threads` named workers (`inf-job-{i}`). A `threads` of
    /// 0 is treated as 1 — a pool always has at least one worker.
    pub fn new(threads: usize) -> Self {
        let threads = threads.max(1);
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("inf-job-{i}"))
            .build()
            .expect("job pool build");
        Self { pool }
    }

    /// A pool sized to the machine's available parallelism (the default engine
    /// pool).
    pub fn with_available_parallelism() -> Self {
        Self::new(default_thread_count())
    }

    /// How many worker threads back this pool.
    pub fn thread_count(&self) -> usize {
        self.pool.current_num_threads()
    }

    /// Run `f` inside this pool. rayon calls made *inside* `f` (`par_iter`,
    /// `join`, `scope`) execute on this pool's workers, so this is the seam that
    /// binds ad-hoc rayon usage to the engine's single pool.
    pub fn install<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        self.pool.install(f)
    }

    /// Apply `f` to every element in parallel, for side effects. Order of
    /// application is unspecified; use it when the work commutes.
    pub fn parallel_for<T, F>(&self, items: &[T], f: F)
    where
        T: Sync,
        F: Fn(&T) + Send + Sync,
    {
        self.install(|| items.par_iter().for_each(f));
    }

    /// A parallel **pure map**: apply `f` to each element and collect the
    /// results *in input order*. The output is a deterministic function of the
    /// input — identical for any worker count — which is what lets the engine
    /// parallelize import/compile without perturbing determinism.
    pub fn parallel_map<T, R, F>(&self, items: Vec<T>, f: F) -> Vec<R>
    where
        T: Send,
        R: Send,
        F: Fn(T) -> R + Send + Sync,
    {
        self.install(move || items.into_par_iter().map(f).collect())
    }

    /// A parallel pure map over borrowed items, in input order (see
    /// [`parallel_map`](Self::parallel_map)).
    pub fn parallel_map_ref<T, R, F>(&self, items: &[T], f: F) -> Vec<R>
    where
        T: Sync,
        R: Send,
        F: Fn(&T) -> R + Send + Sync,
    {
        self.install(|| items.par_iter().map(f).collect())
    }

    /// Fork-join two closures, running them potentially in parallel and
    /// returning both results.
    pub fn join<A, B, RA, RB>(&self, a: A, b: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        self.install(|| rayon::join(a, b))
    }

    /// Open a scoped fan-out: `f` receives a [`rayon::Scope`] onto which it can
    /// `spawn` an arbitrary, dynamic number of tasks that borrow the enclosing
    /// stack frame; all spawned tasks complete before `scope` returns.
    pub fn scope<'scope, F, R>(&self, f: F) -> R
    where
        F: FnOnce(&rayon::Scope<'scope>) -> R + Send,
        R: Send,
    {
        self.install(|| rayon::scope(f))
    }
}

/// Default worker count: the machine's available parallelism, clamped to at
/// least one.
fn default_thread_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

static GLOBAL: OnceLock<JobPool> = OnceLock::new();

/// The process-wide engine job pool, sized to available parallelism on first
/// use. The free functions in this module dispatch here.
pub fn global() -> &'static JobPool {
    GLOBAL.get_or_init(JobPool::with_available_parallelism)
}

/// [`JobPool::parallel_for`] on the [`global`] pool.
pub fn parallel_for<T, F>(items: &[T], f: F)
where
    T: Sync,
    F: Fn(&T) + Send + Sync,
{
    global().parallel_for(items, f);
}

/// [`JobPool::parallel_map`] on the [`global`] pool — deterministic, in-order.
pub fn parallel_map<T, R, F>(items: Vec<T>, f: F) -> Vec<R>
where
    T: Send,
    R: Send,
    F: Fn(T) -> R + Send + Sync,
{
    global().parallel_map(items, f)
}

/// [`JobPool::parallel_map_ref`] on the [`global`] pool — deterministic,
/// in-order.
pub fn parallel_map_ref<T, R, F>(items: &[T], f: F) -> Vec<R>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> R + Send + Sync,
{
    global().parallel_map_ref(items, f)
}

/// [`JobPool::join`] on the [`global`] pool.
pub fn join<A, B, RA, RB>(a: A, b: B) -> (RA, RB)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
    RA: Send,
    RB: Send,
{
    global().join(a, b)
}

/// [`JobPool::scope`] on the [`global`] pool.
pub fn scope<'scope, F, R>(f: F) -> R
where
    F: FnOnce(&rayon::Scope<'scope>) -> R + Send,
    R: Send,
{
    global().scope(f)
}

/// A fresh unbounded MPMC channel for work hand-off / result collection. This is
/// the engine's channel primitive (flume): usable from both pool workers and
/// ordinary threads, and — unlike `std::sync::mpsc` — multi-consumer, which the
/// worker-pool import/cook paths rely on.
pub fn channel<T>() -> (flume::Sender<T>, flume::Receiver<T>) {
    flume::unbounded()
}

/// A bounded MPMC channel with `cap` slots (back-pressures producers).
pub fn bounded_channel<T>(cap: usize) -> (flume::Sender<T>, flume::Receiver<T>) {
    flume::bounded(cap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_map_preserves_order() {
        let out = parallel_map((0..1000).collect(), |x| x * 2);
        let expected: Vec<i32> = (0..1000).map(|x| x * 2).collect();
        assert_eq!(out, expected);
    }

    /// The P7.0 determinism guard: a pure parallel map yields byte-identical
    /// output no matter how many workers run it.
    #[test]
    fn parallel_map_is_pool_size_invariant() {
        let input: Vec<u64> = (0..5000).collect();
        // A deliberately order-sensitive fold-into-vec: each output depends only
        // on its own input, but a non-order-preserving collect would scramble it.
        let f = |x: u64| x.wrapping_mul(2654435761).rotate_left((x % 31) as u32);

        let single = JobPool::new(1).parallel_map(input.clone(), f);
        let few = JobPool::new(2).parallel_map(input.clone(), f);
        let many = JobPool::new(8).parallel_map(input.clone(), f);

        assert_eq!(single, few);
        assert_eq!(few, many);
        // And equal to the serial reference.
        let serial: Vec<u64> = input.into_iter().map(f).collect();
        assert_eq!(single, serial);
    }

    #[test]
    fn parallel_for_visits_all() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let counter = AtomicUsize::new(0);
        let items: Vec<usize> = (0..256).collect();
        JobPool::new(4).parallel_for(&items, |_| {
            counter.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(counter.load(Ordering::Relaxed), 256);
    }

    #[test]
    fn join_returns_both() {
        let (a, b) = JobPool::new(2).join(|| 1 + 1, || 2 + 2);
        assert_eq!((a, b), (2, 4));
    }

    #[test]
    fn scope_runs_dynamic_tasks() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let sum = AtomicUsize::new(0);
        JobPool::new(4).scope(|s| {
            for i in 0..100 {
                let sum = &sum;
                s.spawn(move |_| {
                    sum.fetch_add(i, Ordering::Relaxed);
                });
            }
        });
        assert_eq!(sum.load(Ordering::Relaxed), (0..100).sum());
    }

    #[test]
    fn channel_round_trips() {
        let (tx, rx) = channel::<u32>();
        tx.send(7).unwrap();
        drop(tx);
        assert_eq!(rx.recv().unwrap(), 7);
        assert!(rx.recv().is_err());
    }

    #[test]
    fn pool_is_at_least_one_thread() {
        assert!(JobPool::new(0).thread_count() >= 1);
        assert!(global().thread_count() >= 1);
    }
}
