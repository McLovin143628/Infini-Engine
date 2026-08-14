//! **The non-blocking readback ring** (P26.4) — the codebase's first, and
//! deliberately not a virtual-texturing detail.
//!
//! Every GPU→CPU read in this renderer before now was a *blocking* one:
//! `passes::vgeom::map_u32`, `pick.rs`, `headless.rs`, the GI and sky LUT
//! readbacks all `map_async` and then `poll(wait_indefinitely())`. That is right
//! for a picker (the answer is needed before the click is), for a golden
//! (there is no next frame) and for a one-shot bake. It is wrong for a
//! *per-frame* signal: waiting on the GPU inside the frame loop serializes the
//! pipeline and costs the whole frame's latency to save two frames of staleness.
//!
//! So this is the shared primitive, and the three systems that will consume it
//! are named up front: P26.4's virtual-texture coverage feedback, P27.1's
//! shadow-page marking, and P28.3's unified streamer, which folds all three into
//! one ring. It lives in `inf_render::readback` — not in `vt.rs`, not behind a
//! VT-shaped API — for exactly that reason.
//!
//! # The pinned latency is the whole determinism argument
//!
//! `inf_vgeom::stream`'s ruling (lines 52–62) rejects GPU feedback as a
//! residency input because *"a GPU readback is one frame latent, so the want set
//! would depend on the frame history, and two runs differing by a single dropped
//! frame would diverge"*. That objection is about **"whatever arrived"**, not
//! about readback. A ring answers it by refusing to be opportunistic:
//!
//! * [`ReadbackRing::take`] is asked for a specific frame and returns the bytes
//!   recorded at **exactly** `frame − latency` or nothing at all. It never
//!   returns the newest ready slot, never returns `frame − 1`, and never returns
//!   `frame − 3`. A slot that is late is not a slightly older answer, it is
//!   **no** answer, and the caller degrades to its analytic floor — which is
//!   deterministic, and is what makes "a dropped feedback frame degrades to the
//!   floor" a property rather than a hope.
//! * the mapping is pumped with [`PollType::Poll`](wgpu::PollType::Poll), which
//!   checks the device once and returns; nothing here ever blocks.
//!
//! Two runs of one scripted path therefore see the same feedback at the same
//! frames, because the frame each read comes from is a *constant offset* rather
//! than a race. What is not promised is that the GPU is always fast enough —
//! and the honest handling of that is the floor, plus [`ReadbackRing::misses`],
//! which counts every time it happened so a gate can assert that it did not.
//!
//! # Sizing
//!
//! `latency + 1` slots, which is the minimum that keeps the slot being recorded
//! this frame distinct from the slot being read this frame: at frame `F` the ring
//! records into `F % slots` and reads `(F − latency) % slots`, and those differ
//! for every `F` exactly when `slots > latency`. One more slot would buy nothing
//! but bytes.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// **The pinned feedback latency, in frames.** Two.
///
/// One would be tighter and is not safely reachable: a frame's copy is recorded
/// into the same command buffer as the frame's own work, so the earliest a
/// caller could ask for it is the *next* frame's start, and on a GPU one frame
/// deep that read would find the copy still executing on most frames — turning
/// "sometimes late" into the normal case and the floor into the normal answer.
/// Three would be a third of a second of staleness at 10 fps for nothing.
///
/// It is a **constant and not a setting** on purpose: it is the offset a trace
/// is a pure function of, so a host that could tune it could make two machines
/// disagree about one scripted path.
///
/// **Since P28.3 the constant lives in `inf_stream::ring`** and this is a
/// re-export. The module's opening paragraph named *"P28.3's unified streamer,
/// which folds all three into one ring"* as a consumer; what that fold turned
/// out to be is one **domain** rather than one buffer — the latency, the slot
/// arithmetic and the accounting are shared, the staging buffers are not, and
/// `inf_stream::ring`'s docs carry the measurement that decided it (one buffer
/// means a texture registration drops an in-flight shadow mask).
pub use inf_stream::FEEDBACK_LATENCY_FRAMES as READBACK_LATENCY_FRAMES;

/// What one slot is doing.
///
/// The `Recorded → InFlight` split is not bookkeeping for its own sake: it is a
/// `wgpu` rule. `map_async` puts a buffer into a pending-map state immediately,
/// and submitting a command buffer that writes a pending-map buffer is a
/// validation error — *"Buffer … is still mapped"*, measured. So the map is
/// requested in [`ReadbackRing::poll`], **after** the caller's submit, never
/// beside the `copy_buffer_to_buffer` that fills it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotState {
    /// Never used, or already consumed and unmapped.
    Free,
    /// A copy for this frame is in an encoder the caller has not submitted yet.
    Recorded(u64),
    /// The copy is submitted and the map has been requested.
    InFlight(u64),
    /// The map callback fired: the bytes for this frame are readable.
    Ready(u64),
}

struct Slot {
    buffer: wgpu::Buffer,
    state: SlotState,
    /// Set by the `map_async` callback. `Arc` because the callback outlives the
    /// borrow that created it, and `AtomicBool` rather than a channel because
    /// the only question is "has it fired", asked once per frame.
    mapped: Arc<AtomicBool>,
    /// Whether the map callback reported failure (a lost device, a buffer
    /// destroyed under it). Distinguished from "not yet" so a permanently
    /// broken slot is recycled rather than waited on for ever.
    failed: Arc<AtomicBool>,
}

/// A ring of mappable staging buffers read at a **pinned** frame latency.
///
/// One ring reads one GPU buffer. A system with two signals uses two rings, or
/// packs them into one buffer — the ring has no opinion about what the bytes
/// mean.
pub struct ReadbackRing {
    slots: Vec<Slot>,
    /// Bytes per slot.
    size: u64,
    /// Frames between recording and reading. See [`READBACK_LATENCY_FRAMES`].
    latency: u64,
    /// Reads that found their slot not ready — the instrument that makes "the
    /// ring kept up" assertable rather than assumed.
    misses: u64,
    /// Reads that returned bytes.
    hits: u64,
}

impl ReadbackRing {
    /// A ring of `size` bytes per slot at the pinned latency.
    ///
    /// `size` is rounded up to 4 so a `u32` view of the bytes is always whole,
    /// and floored at 4 because `wgpu` refuses a zero-sized buffer.
    pub fn new(device: &wgpu::Device, label: &str, size: u64) -> Self {
        Self::with_latency(device, label, size, READBACK_LATENCY_FRAMES)
    }

    /// [`new`](Self::new) with an explicit latency — **for tests only**, so the
    /// pinning itself can be exercised (a ring at latency 1 and one at latency 2
    /// must read different frames from the same recording sequence). Shipping
    /// callers take [`READBACK_LATENCY_FRAMES`]; see the constant for why it is
    /// not a setting.
    pub fn with_latency(device: &wgpu::Device, label: &str, size: u64, latency: u64) -> Self {
        let size = size.next_multiple_of(4).max(4);
        let slots = (0..inf_stream::ring_slots(latency))
            .map(|i| Slot {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("{label}-readback-{i}")),
                    size,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                state: SlotState::Free,
                mapped: Arc::new(AtomicBool::new(false)),
                failed: Arc::new(AtomicBool::new(false)),
            })
            .collect();
        Self {
            slots,
            size,
            latency,
            misses: 0,
            hits: 0,
        }
    }

    /// Bytes per slot.
    #[inline]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// The pinned latency this ring reads at.
    #[inline]
    pub fn latency(&self) -> u64 {
        self.latency
    }

    /// Slots — `latency + 1`.
    #[inline]
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Reads that found no ready slot for the frame they were asked for. **The
    /// anti-vacuity number**: a feedback loop whose ring never lands is
    /// indistinguishable from one with no feedback at all, and both look like a
    /// scene that simply took the floor.
    #[inline]
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Reads that returned bytes.
    #[inline]
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// **Record frame `frame`'s copy** of `src` into this frame's slot.
    ///
    /// Call once per frame, into the encoder the frame is about to submit — so
    /// the copy is ordered after whatever wrote `src`, by the ordinary command
    /// ordering, with no barrier of our own.
    ///
    /// Returns `false` when the slot is not free (the GPU is more than a ring
    /// behind, or the caller never polled) — the copy is skipped rather than
    /// racing a mapped buffer, and the read `latency` frames later misses and
    /// takes the floor.
    pub fn record(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Buffer,
        frame: u64,
    ) -> bool {
        let n = self.slots.len() as u64;
        let idx = inf_stream::record_slot(frame, n) as usize;
        // Reclaim first: the slot this frame wants may still be holding the
        // mapping from `frame - slots`, which nothing consumed (a caller that
        // skipped a `take`, or a miss that was never collected).
        self.reclaim(idx);
        let copy = self.size.min(src.size());
        let slot = &mut self.slots[idx];
        if slot.state != SlotState::Free || copy == 0 {
            return false;
        }
        encoder.copy_buffer_to_buffer(src, 0, &slot.buffer, 0, copy);
        slot.state = SlotState::Recorded(frame);
        slot.mapped.store(false, Ordering::Release);
        slot.failed.store(false, Ordering::Release);
        true
    }

    /// **Pump the ring once, without blocking.** Call once per frame, *after*
    /// the frame's `queue.submit`.
    ///
    /// Two things happen here and the order between them and the caller's submit
    /// is load-bearing. Recorded slots have their map **requested** — which must
    /// not happen before the submit that fills them, because `map_async` puts
    /// the buffer into a pending-map state and `wgpu` then refuses the submit
    /// outright (*"Buffer … is still mapped"*, measured). And in-flight slots
    /// whose callback has fired become readable. `map_async` callbacks are only
    /// invoked from inside a device poll, so a ring that is never polled never
    /// becomes ready — and the caller would see a permanent floor with no error
    /// anywhere.
    pub fn poll(&mut self, device: &wgpu::Device) {
        for slot in &mut self.slots {
            if let SlotState::Recorded(f) = slot.state {
                let (mapped, failed) = (slot.mapped.clone(), slot.failed.clone());
                slot.buffer
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |r| {
                        if r.is_err() {
                            failed.store(true, Ordering::Release);
                        }
                        mapped.store(true, Ordering::Release);
                    });
                slot.state = SlotState::InFlight(f);
            }
        }
        // `Poll` checks the device a single time and returns; the blocking
        // variant is what this whole module exists to avoid.
        let _ = device.poll(wgpu::PollType::Poll);
        for slot in &mut self.slots {
            if let SlotState::InFlight(f) = slot.state {
                if slot.mapped.load(Ordering::Acquire) {
                    slot.state = if slot.failed.load(Ordering::Acquire) {
                        // A failed map leaves nothing mapped, so there is nothing
                        // to unmap; the slot is simply available again.
                        SlotState::Free
                    } else {
                        SlotState::Ready(f)
                    };
                }
            }
        }
    }

    /// **Read the bytes recorded at exactly `frame − latency`**, or nothing.
    ///
    /// Never the newest ready slot and never an adjacent frame: see the module
    /// docs. The slot is released as this returns, so the bytes are handed to a
    /// closure rather than out of the ring — which is also what keeps the
    /// `wgpu` mapping's lifetime from escaping.
    ///
    /// The first `latency` frames of a session have nothing to read — their
    /// answers were recorded before the ring existed — and that is a **miss**,
    /// counted like any other, not a silent zero.
    pub fn take<R>(&mut self, frame: u64, read: impl FnOnce(&[u8]) -> R) -> Option<R> {
        let Some(want) = inf_stream::read_frame(frame, self.latency) else {
            self.misses += 1;
            return None;
        };
        let idx = inf_stream::record_slot(want, self.slots.len() as u64) as usize;
        if self.slots[idx].state != SlotState::Ready(want) {
            self.misses += 1;
            return None;
        }
        let out = {
            let view = self.slots[idx].buffer.slice(..).get_mapped_range().ok()?;
            read(&view)
        };
        self.slots[idx].buffer.unmap();
        self.slots[idx].state = SlotState::Free;
        self.hits += 1;
        Some(out)
    }

    /// Release a slot that is `Ready` and was never consumed.
    fn reclaim(&mut self, idx: usize) {
        if let SlotState::Ready(_) = self.slots[idx].state {
            self.slots[idx].buffer.unmap();
            self.slots[idx].state = SlotState::Free;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The geometry: `latency + 1` slots, and the slot recorded at frame `F` is
    /// never the slot read at frame `F`.
    ///
    /// A pure arithmetic arm — no device — because this is the property that
    /// makes the ring correct and it is one modular equation. `slots > latency`
    /// is exactly the condition, so it is asserted at both the shipped latency
    /// and at the boundary a smaller ring would sit on.
    ///
    /// Kept after P28.3 moved the arithmetic into `inf_stream::ring`, and
    /// written against **this** module's spelling of it: the shared functions
    /// are what the buffers are indexed by, so an arm that stopped reading them
    /// here would stop being about this ring.
    #[test]
    fn a_recorded_slot_is_never_the_slot_read_in_the_same_frame() {
        for latency in 1..=4u64 {
            let slots = latency + 1;
            for frame in latency..latency + 64 {
                assert_ne!(
                    frame % slots,
                    (frame - latency) % slots,
                    "latency {latency} frame {frame}: the ring would read the slot \
                     it is recording into"
                );
            }
        }
        // …and a ring with only `latency` slots would collide on every frame,
        // which is the mistake the `+ 1` avoids and the reason it is asserted.
        for frame in 2u64..66 {
            assert_eq!(frame % 2, (frame - 2) % 2);
        }
    }

    /// `READBACK_LATENCY_FRAMES` is pinned at two.
    ///
    /// A constant with an arm looks like ceremony and is not: the number is the
    /// offset a residency trace is a pure function of, so moving it changes what
    /// every P26.4 gate compares — and the failure would be a trace mismatch
    /// three crates away rather than a line pointing here.
    #[test]
    fn the_latency_is_pinned_at_two_frames() {
        assert_eq!(READBACK_LATENCY_FRAMES, 2);
    }
}
