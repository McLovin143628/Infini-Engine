//! **One feedback/readback ledger** (P28.3, clause 1): the pinned latency, the
//! slot arithmetic, and the accounting every consumer that reads the GPU back
//! shares.
//!
//! `inf_render::readback::ReadbackRing` owns the `wgpu` staging buffers and
//! keeps owning them. What moves here is the part that has to be **one** thing
//! for a trace to be a function of committed input: the latency constant, the
//! modular arithmetic that keeps the slot being recorded distinct from the slot
//! being read, and the ledger that says which frame each consumer actually got.
//!
//! # One ring or one domain — the measurement
//!
//! The obvious reading of "one feedback/readback ring" is one buffer. It was
//! measured and refused, and the refusal is structural rather than about cost.
//! The two producers are re-created on different events: the virtual-texture
//! mask is sized from the texture *registry* and rebuilt when a registration
//! grows it; the shadow mask is sized from the *light set* and rebuilt when
//! that changes. One buffer means either event rebuilds both, so **registering
//! a texture would drop an in-flight shadow mask** — turning two independent,
//! counted misses into one coupled miss, on a frame where nothing about
//! shadows changed. The saving would have been one `copy_buffer_to_buffer` and
//! one staging allocation per frame. So: two buffers, one domain, and
//! [`RingLedger`] is what makes the domain observable — including the property
//! that matters, which is that both consumers read the **same** source frame.
//!
//! # The latency is a constant and not a setting
//!
//! Restated from `inf_render::readback`, where it lived alone: it is the offset
//! a residency trace is a pure function of, so a host that could tune it could
//! make two machines disagree about one scripted path. A read is for
//! **exactly** `frame − latency` or it is a miss; never the newest ready slot,
//! never an adjacent frame.

/// **The pinned feedback latency, in frames.** Two.
///
/// One is not safely reachable — a frame's copy rides the frame's own command
/// buffer, so the earliest a caller could ask is the next frame's start, and on
/// a GPU one frame deep that read finds the copy still executing on most
/// frames, which makes "late" the normal case and the analytic floor the normal
/// answer. Three is a third of a second of staleness at 10 fps for nothing.
pub const FEEDBACK_LATENCY_FRAMES: u64 = 2;

/// Slots a ring needs at `latency`: `latency + 1`.
///
/// The minimum that keeps the slot being recorded this frame distinct from the
/// slot being read this frame — at frame `F` a ring records into
/// `F % slots` and reads `(F − latency) % slots`, and those differ for every
/// `F` exactly when `slots > latency`. One more slot buys nothing but bytes.
#[inline]
pub const fn ring_slots(latency: u64) -> u64 {
    latency + 1
}

/// The slot frame `frame` records into.
#[inline]
pub const fn record_slot(frame: u64, slots: u64) -> u64 {
    frame % slots
}

/// **The frame a read at `frame` is allowed to consume** — exactly
/// `frame − latency`, or `None` for the first `latency` frames of a session,
/// whose answers were recorded before the ring existed.
///
/// `None` is a **miss**, counted like any other, not a silent zero: a feedback
/// loop whose ring never lands is indistinguishable from one with no feedback
/// at all, and both look like a scene that simply took the floor.
#[inline]
pub const fn read_frame(frame: u64, latency: u64) -> Option<u64> {
    frame.checked_sub(latency)
}

/// One consumer's line in the ledger.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RingReader {
    /// Reads that returned bytes.
    pub hits: u64,
    /// Reads that found their slot not ready. **The anti-vacuity number.**
    pub misses: u64,
    /// The source frame of the last hit, or 0 if there has not been one.
    /// **A frame index, not a stamp** — it is compared between consumers, which
    /// is exactly what this ledger exists to make possible.
    pub last_source: u64,
}

impl RingReader {
    /// Reads attempted.
    #[inline]
    pub fn reads(&self) -> u64 {
        self.hits + self.misses
    }
}

/// The readback accounting for every consumer, under one latency.
///
/// Deliberately indexed by [`Consumer`](crate::budget::Consumer) rather than by
/// a name: the ledger is the arbiter's, and the arbiter knows exactly three
/// consumers. `Geometry` never reads a ring today — the meshlet streamer's
/// wants are CPU-derived by ruling (`inf_vgeom::stream`'s "why not cull
/// feedback") — so its line reads zero and that is a statement rather than an
/// absence.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RingLedger {
    readers: [RingReader; crate::budget::CONSUMERS],
}

impl RingLedger {
    /// **Record a read attempt at `frame`** and answer which source frame the
    /// consumer may take, or `None`.
    ///
    /// `ready` is the consumer's own answer to "are those bytes actually
    /// mapped" — the ledger pins *which* frame is legal, the ring says whether
    /// it arrived.
    pub fn read(
        &mut self,
        consumer: crate::budget::Consumer,
        frame: u64,
        ready: impl FnOnce(u64) -> bool,
    ) -> Option<u64> {
        let line = &mut self.readers[consumer.index()];
        match read_frame(frame, FEEDBACK_LATENCY_FRAMES) {
            Some(src) if ready(src) => {
                line.hits += 1;
                line.last_source = src;
                Some(src)
            }
            _ => {
                line.misses += 1;
                None
            }
        }
    }

    #[inline]
    pub fn reader(&self, consumer: crate::budget::Consumer) -> RingReader {
        self.readers[consumer.index()]
    }

    /// Reads that returned bytes, across every consumer.
    pub fn hits(&self) -> u64 {
        self.readers.iter().map(|r| r.hits).sum()
    }

    /// Reads that missed, across every consumer.
    pub fn misses(&self) -> u64 {
        self.readers.iter().map(|r| r.misses).sum()
    }

    /// **Whether every consumer that has read anything last read the same
    /// source frame.**
    ///
    /// The property "one domain" buys and "two rings" alone cannot: two
    /// consumers reading at frame `F` consume frame `F − 2`, both of them, or
    /// one of them missed. A consumer that has never hit is not evidence
    /// against it and is skipped.
    pub fn readers_agree(&self) -> bool {
        let mut seen: Option<u64> = None;
        for r in &self.readers {
            if r.hits == 0 {
                continue;
            }
            match seen {
                None => seen = Some(r.last_source),
                Some(s) if s == r.last_source => {}
                Some(_) => return false,
            }
        }
        true
    }

    /// A one-line human summary, in the shape the other streaming summaries
    /// ship.
    pub fn summary(&self) -> String {
        let mut s = format!(
            "stream ring: latency {FEEDBACK_LATENCY_FRAMES}, {} landed / {} missed",
            self.hits(),
            self.misses()
        );
        for c in crate::budget::Consumer::ALL {
            let r = self.reader(c);
            if r.reads() > 0 {
                s.push_str(&format!(" [{} {}/{}]", c.label(), r.hits, r.reads()));
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::Consumer;

    /// The geometry: `latency + 1` slots, and the slot recorded at frame `F` is
    /// never the slot read at frame `F`.
    ///
    /// A pure arithmetic arm, at both the shipped latency and the boundary a
    /// smaller ring would sit on — the property that makes the ring correct,
    /// stated as the one modular equation it is.
    #[test]
    fn a_recorded_slot_is_never_the_slot_read_in_the_same_frame() {
        for latency in 1..=4u64 {
            let slots = ring_slots(latency);
            for frame in latency..latency + 64 {
                assert_ne!(
                    record_slot(frame, slots),
                    record_slot(frame - latency, slots),
                    "latency {latency} frame {frame}: the ring would read the slot it \
                     is recording into"
                );
            }
        }
        // …and a ring with only `latency` slots collides on every frame, which
        // is the mistake the `+ 1` avoids.
        for frame in 2u64..66 {
            assert_eq!(record_slot(frame, 2), record_slot(frame - 2, 2));
        }
    }

    /// The latency is pinned at two, and a read is for exactly `frame − 2`.
    #[test]
    fn the_latency_is_pinned_at_two_frames() {
        assert_eq!(FEEDBACK_LATENCY_FRAMES, 2);
        assert_eq!(read_frame(0, 2), None);
        assert_eq!(read_frame(1, 2), None);
        assert_eq!(read_frame(2, 2), Some(0));
        assert_eq!(read_frame(97, 2), Some(95));
    }

    /// **Two consumers reading at one frame read one source frame** — and the
    /// ledger can tell that from a consumer that simply has not read yet.
    #[test]
    fn two_consumers_reading_at_one_frame_read_one_source_frame() {
        let mut led = RingLedger::default();
        assert!(led.readers_agree(), "an empty ledger cannot disagree");
        for frame in 0..6u64 {
            let t = led.read(Consumer::Texture, frame, |_| true);
            let s = led.read(Consumer::Shadow, frame, |_| true);
            assert_eq!(t, s, "frame {frame}");
            assert!(led.readers_agree(), "frame {frame}");
        }
        // The first two frames are misses for both — their answers predate the
        // ring — and they are counted, not silent.
        assert_eq!(led.reader(Consumer::Texture).misses, 2);
        assert_eq!(led.reader(Consumer::Texture).hits, 4);
        assert_eq!(led.reader(Consumer::Geometry).reads(), 0);
        assert!(
            led.readers_agree(),
            "a consumer that never reads is not evidence"
        );

        // FALSIFIABLE: a consumer that took an adjacent frame breaks agreement.
        let mut led = RingLedger::default();
        led.read(Consumer::Texture, 6, |_| true);
        led.read(Consumer::Shadow, 7, |_| true);
        assert!(
            !led.readers_agree(),
            "the ledger cannot see two consumers on two different frames"
        );
    }

    /// A late slot is a **miss**, not an older answer.
    #[test]
    fn a_slot_that_is_not_ready_is_a_miss_and_never_an_adjacent_frame() {
        let mut led = RingLedger::default();
        // Frame 5 wants frame 3 and only frame 2 is mapped: nothing at all.
        assert_eq!(led.read(Consumer::Texture, 5, |src| src == 2), None);
        assert_eq!(led.reader(Consumer::Texture).misses, 1);
        assert_eq!(led.reader(Consumer::Texture).last_source, 0);
        assert_eq!(led.read(Consumer::Texture, 5, |src| src == 3), Some(3));
        assert_eq!(led.reader(Consumer::Texture).hits, 1);
    }

    /// The summary names every consumer that read and none that did not.
    #[test]
    fn the_ring_line_says_who_read_what() {
        let mut led = RingLedger::default();
        for f in 0..4u64 {
            led.read(Consumer::Texture, f, |_| true);
            led.read(Consumer::Shadow, f, |_| f > 2);
        }
        let s = led.summary();
        assert!(s.contains("stream ring"), "{s}");
        assert!(s.contains("latency 2"), "{s}");
        assert!(s.contains("[texture 2/4]"), "{s}");
        assert!(s.contains("[shadow 1/4]"), "{s}");
        assert!(!s.contains("geometry"), "a consumer that never read: {s}");
    }
}
