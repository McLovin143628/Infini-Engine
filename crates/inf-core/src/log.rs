//! A session log that cannot outgrow its ceiling (Hardening D).
//!
//! # The class this exists to close
//!
//! Four independent `Vec`s in this tree accumulated one entry per fixed step, per
//! streaming event or per line of subprocess output, for the life of a session,
//! and every one of them had **only test consumers**:
//!
//! * `RuntimeSim::audio_log` and `SimSession::audio_log` — at least one
//!   `SetListener` per fixed step, i.e. ~216 000 commands per hour, **in the
//!   shipped player**;
//! * `RuntimeSim::logs` and `SimSession::logs` — `debug.print` output *and* a
//!   failed event dispatch, which pushes a formatted `String` every tick, so the
//!   growth rate is reachable from authored content;
//! * `CellStreaming::events` — one per cell activation/deactivation for the
//!   session;
//! * the PIE monitor's captured stderr — every line the player ever writes.
//!
//! A ring is the shape that fits all four, because the alternative — draining
//! into a host — needs a host that *wants* the data, and the shipped player has
//! no consumer for any of it. What a ring must not do is lie about it: dropping
//! the oldest entries silently would turn "the audio stream is exactly this" into
//! "the audio stream ends with this", which is a different claim and a weaker
//! gate. So [`BoundedLog::dropped`] counts what fell off the front, and every
//! accessor that hands out the retained slice sits beside it.
//!
//! # Why a `Vec` and not a `VecDeque`
//!
//! Every consumer of these logs reads them as a contiguous `&[T]` — an existing,
//! widely-called shape (`sim.logs()`, `session.audio_command_log()`,
//! `streaming.events()`), and `VecDeque` cannot hand one out without `&mut self`.
//! Eviction therefore drops a **batch** from the front rather than one entry per
//! push: one `Vec::drain` of `cap / 4` every `cap / 4` pushes is amortized O(1)
//! per push, and it keeps the accessor's signature — and therefore every caller —
//! unchanged.

/// A log with a hard entry ceiling and an honest count of what it dropped.
///
/// Push-only; the oldest entries are evicted in batches once the ceiling is
/// reached. See the module docs for why this shape rather than a drain.
#[derive(Debug, Clone)]
pub struct BoundedLog<T> {
    items: Vec<T>,
    cap: usize,
    dropped: u64,
}

impl<T> BoundedLog<T> {
    /// A log holding at most `cap` entries. `cap` is clamped to at least 1 — a
    /// zero-capacity log would drop everything and report a growing count, which
    /// reads as a defect rather than a configuration.
    pub fn new(cap: usize) -> Self {
        Self {
            items: Vec::new(),
            cap: cap.max(1),
            dropped: 0,
        }
    }

    /// Append one entry, evicting the oldest batch when the ceiling is reached.
    pub fn push(&mut self, item: T) {
        if self.items.len() >= self.cap {
            self.evict();
        }
        self.items.push(item);
    }

    /// Append many, one at a time (so the ceiling holds mid-extend).
    pub fn extend(&mut self, items: impl IntoIterator<Item = T>) {
        for item in items {
            self.push(item);
        }
    }

    /// Drop the oldest quarter of the ceiling (at least one entry), counting it.
    fn evict(&mut self) {
        let n = (self.cap / 4).max(1).min(self.items.len());
        self.items.drain(..n);
        self.dropped = self.dropped.saturating_add(n as u64);
    }

    /// The retained entries, oldest first.
    pub fn as_slice(&self) -> &[T] {
        &self.items
    }

    /// How many entries were evicted to keep the ceiling. **Non-zero means
    /// [`as_slice`](Self::as_slice) is a tail, not the whole stream** — every
    /// consumer that reports a log is expected to say so.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// The entry ceiling this log was built with.
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Retained entry count (never above [`capacity`](Self::capacity)).
    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Take the retained entries, leaving the log empty. The dropped count is
    /// **kept**: it describes the whole session, not the current window.
    pub fn take(&mut self) -> Vec<T> {
        std::mem::take(&mut self.items)
    }

    /// Empty the log and forget what it dropped — for a container being reused
    /// for a *new* session, where the previous session's count would be a lie.
    pub fn reset(&mut self) {
        self.items.clear();
        self.dropped = 0;
    }
}

impl<T> Default for BoundedLog<T> {
    /// 8 192 entries: comfortably past every assertion in the tree (the longest
    /// scripted gate runs a few thousand steps) and a hard bound in the shipped
    /// player, which is the point.
    fn default() -> Self {
        Self::new(DEFAULT_LOG_CAPACITY)
    }
}

/// The default ceiling for a session log. See [`BoundedLog::default`].
pub const DEFAULT_LOG_CAPACITY: usize = 8192;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_log_never_exceeds_its_ceiling() {
        let mut log = BoundedLog::new(16);
        for i in 0..1000 {
            log.push(i);
        }
        assert!(log.len() <= 16, "len {} exceeded the ceiling", log.len());
        assert_eq!(log.dropped(), 1000 - log.len() as u64);
    }

    #[test]
    fn eviction_drops_the_oldest_and_keeps_the_newest() {
        let mut log = BoundedLog::new(8);
        log.extend(0..12);
        let kept = log.as_slice();
        assert_eq!(
            *kept.last().expect("non-empty"),
            11,
            "the newest entry is always retained"
        );
        assert!(
            kept.windows(2).all(|w| w[1] == w[0] + 1),
            "retained entries stay in push order, contiguously"
        );
        assert!(
            !kept.contains(&0),
            "the oldest entries are the ones that go"
        );
    }

    #[test]
    fn a_log_inside_its_ceiling_drops_nothing() {
        let mut log = BoundedLog::new(64);
        log.extend(0..64);
        assert_eq!(log.dropped(), 0);
        assert_eq!(log.len(), 64);
        assert_eq!(log.as_slice(), (0..64).collect::<Vec<_>>().as_slice());
    }

    #[test]
    fn take_empties_the_window_and_keeps_the_session_count() {
        let mut log = BoundedLog::new(4);
        log.extend(0..10);
        let dropped = log.dropped();
        assert!(dropped > 0);
        let taken = log.take();
        assert!(!taken.is_empty());
        assert!(log.is_empty());
        assert_eq!(
            log.dropped(),
            dropped,
            "the dropped count describes the session, not the window"
        );
        log.reset();
        assert_eq!(log.dropped(), 0, "a reused container starts honest");
    }

    #[test]
    fn a_zero_capacity_is_clamped_rather_than_swallowing_everything() {
        let mut log = BoundedLog::new(0);
        log.push(1);
        assert_eq!(log.as_slice(), &[1]);
        assert_eq!(log.capacity(), 1);
    }
}
