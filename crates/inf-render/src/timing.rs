//! **The per-pass GPU clock** (island wave I4) — the half of the fps instrument
//! that lives in the renderer.
//!
//! Until this module the tree had no GPU timing at all: the whole repo contained
//! zero `QuerySet`s and **sixty-one** literal `timestamp_writes: None`s (the
//! wave's own ledger said sixty; `git grep -o` on its base tree says 61, and a
//! number that only lives in prose drifts), and every frame
//! number ever quoted was a CPU wall clock around
//! `render(); poll(wait_indefinitely())`. That number is honest about the frame
//! as a whole and says nothing about *where the frame went*, which is the
//! question a 60 fps target actually asks.
//!
//! # Why the timestamps go in the encoder and not in the passes
//!
//! wgpu offers two places to write one: `RenderPassDescriptor::timestamp_writes`
//! (needs `TIMESTAMP_QUERY` alone) and `CommandEncoder::write_timestamp` (needs
//! `TIMESTAMP_QUERY_INSIDE_ENCODERS` as well). The first would mean editing all
//! twenty-nine pass descriptors and every compute pass beside them, and would
//! leave the work this renderer records *outside* the graph — the VT sync point,
//! the VSM caster raster, the feedback ring copy, the VSM marking — unmeasured,
//! which is exactly the work a streaming engine's frame is made of.
//!
//! The second needs **one** seam. `RenderGraph::run` already brackets every node
//! with a `tracing` span; a timestamp goes in beside it, and the four
//! out-of-graph segments are bracketed by hand in `EngineRenderer::render`. So
//! the instrument sees the whole submitted frame, node by node, and no pass had
//! to learn about it.
//!
//! # Off is off
//!
//! A [`FrameTimer`] exists only when a host asks for one
//! ([`EngineRenderer::set_gpu_timing`](crate::EngineRenderer::set_gpu_timing),
//! default `false`), and when there is none the mark call sites are a `None`
//! check. A frame with timing off therefore records **byte-identical commands**
//! to a frame built before this module existed — which is what keeps the 54
//! frozen goldens frozen while an instrument that reads the same renderer exists
//! beside them.
//!
//! # What a number here means
//!
//! A timestamp is written **between two commands**, so `passes[i].ms` is the GPU
//! time from the previous mark to this one: the segment's cost *as the device
//! scheduled it*, including any stall the segment's first command waited on. It
//! is not a sum over the pass's own dispatches, and two adjacent segments can
//! trade time between them when the driver overlaps them. What the sum of every
//! segment is, exactly, is [`FrameTimings::total_ms`] — the first timestamp to
//! the last — and that is the number to compare against a frame budget.

use crate::gpu::GpuContext;

/// One timed segment of a frame.
#[derive(Debug, Clone, PartialEq)]
pub struct PassTime {
    /// The segment's name — a [`RenderNode::name`](crate::graph::RenderNode::name)
    /// for a graph node, or the hand-written label of an out-of-graph segment.
    pub name: &'static str,
    /// GPU milliseconds from the previous mark to this one.
    pub ms: f64,
    /// **CPU milliseconds spent RECORDING this segment** (island wave I4b) —
    /// the wall clock between the previous mark and this one, on the thread that
    /// built the command buffer.
    ///
    /// The GPU number above says what the device did; this says what it cost to
    /// *ask*. They are different questions and on a lit frame they had different
    /// answers: wave I4b measured a 1080p frame whose GPU half was 23.9 ms and
    /// whose `render (record)` CPU stage was **18.7 ms**, and with only the GPU
    /// column the next reader would have gone looking on the wrong processor.
    /// Same construction as the GPU column: each value is "since the previous
    /// mark".
    ///
    /// **What they tile is the MARKED SPAN, not the whole record stage** (the
    /// I4b audit — the first write-up said "exactly as the GPU segments tile the
    /// frame", and that is one word too strong). [`FrameTimer::begin`] opens at
    /// the frame's first *command*, so whatever `EngineRenderer::render` does
    /// before it — the view matrices, the light and deform uniform writes, the
    /// encoder itself — is inside a caller's `render (record)` clock and inside
    /// no segment, and so are `encoder.finish()` and the submit after the last
    /// mark. `fps_instrument.rs` prints that residue every run beside the column,
    /// and bounds the one direction that must hold: the segments may not sum past
    /// the stage that contains them.
    pub cpu_ms: f64,
}

/// One frame's GPU timings, newest resolved frame first-and-only.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FrameTimings {
    /// First timestamp to last — the whole submitted frame on the device.
    pub total_ms: f64,
    /// Every segment, in submission order.
    pub passes: Vec<PassTime>,
}

impl FrameTimings {
    /// The whole frame's **recording** cost on the CPU — the sum of the
    /// segments' [`cpu_ms`](PassTime::cpu_ms) (island wave I4b).
    pub fn cpu_total_ms(&self) -> f64 {
        self.passes.iter().map(|p| p.cpu_ms).sum()
    }

    /// The segments sorted by CPU recording cost, dearest first.
    pub fn by_cpu_cost(&self) -> Vec<(&'static str, f64)> {
        let mut v: Vec<(&'static str, f64)> =
            self.passes.iter().map(|p| (p.name, p.cpu_ms)).collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v
    }
}

impl FrameTimings {
    /// The segments sorted dearest-first, as `(name, ms)` — what a report prints.
    ///
    /// Ties keep submission order, so a run of zero-cost segments reads in the
    /// order the frame recorded them rather than in an arbitrary one.
    pub fn by_cost(&self) -> Vec<(&'static str, f64)> {
        let mut v: Vec<(&'static str, f64)> = self.passes.iter().map(|p| (p.name, p.ms)).collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v
    }
}

/// **THE RECORD PATH'S OWN BREAKDOWN** (island wave I7b) — where the CPU
/// milliseconds inside `EngineRenderer::render` go.
///
/// # Why this exists beside [`PassTime::cpu_ms`]
///
/// The per-segment record column above tiles the **marked span**, and the marked
/// span is not the record stage: [`FrameTimer::begin`] opens at the frame's first
/// *command*, so the view matrices, the uniform writes, the encoder, the cluster
/// plan and the submit are all inside a caller's `render (record)` clock and
/// inside no segment. The I4b audit measured that residue at **two thirds of a
/// 3.0 ms stage** on the composed city and printed it as one unattributed
/// number; wave I7's island measured the same stage at **10.874 ms against a
/// 6.657 ms GPU frame**, which is a diagnostic saying "look on the CPU" and then
/// declining to say where.
///
/// So the record path gets what the fixed step got in wave I4b: named phases
/// that **tile it by construction**, each mark charging the time since the
/// previous one. The phases here cover the whole of `render`, including the
/// parts no GPU segment can reach.
///
/// Off unless the GPU timer is armed — the two are read together, and neither is
/// on in a shipped build.
pub const RECORD_PHASES: usize = 15;

/// The record phases, in the order `EngineRenderer::render` runs them.
///
/// What each covers is documented on its constant in [`record`], deliberately
/// and not as a trailing comment here: `rustfmt` aligns trailing comments into
/// runs of spaces on lines carrying a string literal, and `inf_packager`'s
/// workspace-wide eaten-continuation sweep reads such a run as a mangled `\`.
pub const RECORD_PHASE_NAMES: [&str; RECORD_PHASES] = [
    "targets + luts",
    "view uniforms",
    "encoder",
    "cluster plan",
    "cluster wants",
    "vt stream",
    "cluster commit",
    "vsm sync",
    "vsm raster",
    "frame data",
    "graph",
    "vt feedback",
    "vsm mark",
    "submit",
    "epilogue",
];

/// Record-phase indices, named so a mark cannot drift from its meaning.
pub mod record {
    /// Surface-size checks, target rebuilds, the atmosphere and GI LUT rebuilds.
    pub const TARGETS: usize = 0;
    /// The view matrices, the TAA jitter, and the view / wetness / deform
    /// uniform writes — the three per-frame `write_buffer`s before any command.
    pub const VIEW_UNIFORMS: usize = 1;
    /// `create_command_encoder` and the timer's own `begin`.
    pub const ENCODER: usize = 2;
    /// P28.2 cluster page-in planning (`VgeomNode::plan_cluster_pages`).
    pub const CLUSTER_PLAN: usize = 3;
    /// The VT tile wants derived from the plan (`cluster_tile_wants`) — a
    /// separate row from the plan itself because island wave I7b found the
    /// island's whole record stage inside one of the two and could not have
    /// said which from a single number.
    pub const CLUSTER_WANTS: usize = 4;
    /// P26 virtual-texture streaming.
    pub const VT_STREAM: usize = 5;
    /// Committing the cluster page pairs/retractions planned above.
    pub const CLUSTER_COMMIT: usize = 6;
    /// P27 VSM residency + the receiver slot table.
    pub const VSM_SYNC: usize = 7;
    /// P27 VSM caster packing, invalidation and the page raster.
    pub const VSM_RASTER: usize = 8;
    /// Assembling the `FrameData` the graph borrows.
    pub const FRAME_DATA: usize = 9;
    /// The render graph's nodes — the half [`super::PassTime::cpu_ms`] breaks
    /// down further.
    pub const GRAPH: usize = 10;
    /// The VT feedback ring's readback finish.
    pub const VT_FEEDBACK: usize = 11;
    /// The VSM marking pass and the timer's resolve/copy.
    pub const VSM_MARK: usize = 12;
    /// `encoder.finish()` and `queue.submit`.
    pub const SUBMIT: usize = 13;
    /// Everything after the submit: the previous-VP snapshot and the counters.
    pub const EPILOGUE: usize = 14;
}

/// One frame's record-path phase milliseconds.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RecordProfile {
    /// Milliseconds per phase, indexed by [`RECORD_PHASE_NAMES`].
    pub ms: [f64; RECORD_PHASES],
}

impl RecordProfile {
    /// The whole record path — the sum of its phases, which is what "the phases
    /// tile the record" means arithmetically.
    pub fn total_ms(&self) -> f64 {
        self.ms.iter().sum()
    }

    /// `(name, milliseconds)` in record order.
    pub fn rows(&self) -> impl Iterator<Item = (&'static str, f64)> + '_ {
        RECORD_PHASE_NAMES.iter().copied().zip(self.ms)
    }

    /// The phases sorted dearest-first.
    pub fn dearest_first(&self) -> Vec<(&'static str, f64)> {
        let mut v: Vec<(&'static str, f64)> = self.rows().collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v
    }

    /// Fold `other` in (for a mean over many frames).
    pub fn accumulate(&mut self, other: &RecordProfile) {
        for (a, b) in self.ms.iter_mut().zip(other.ms) {
            *a += b;
        }
    }

    /// Scale every phase by `k`.
    pub fn scale(&mut self, k: f64) {
        for a in self.ms.iter_mut() {
            *a *= k;
        }
    }
}

/// The stopwatch `EngineRenderer::render` marks its phases into.
///
/// Deliberately the same shape as `inf_player::step_profile::StepClock`, down to
/// the private `mark_at` seam that takes the clock reading as an
/// argument: the I7 CI-red was a structural property (a phase marked twice
/// **sums**) that could only be reached through a wall clock, and a shared
/// runner answered. Every arm below drives the arithmetic through `mark_at` and
/// reads no clock at all.
pub struct RecordClock {
    at: Option<std::time::Instant>,
    ms: [f64; RECORD_PHASES],
}

impl RecordClock {
    /// Start a clock, or a no-op one when `on` is false.
    pub fn start(on: bool) -> Self {
        Self {
            at: on.then(std::time::Instant::now),
            ms: [0.0; RECORD_PHASES],
        }
    }

    /// Charge everything since the previous mark to `phase`. One predictable
    /// branch when the clock is off.
    #[inline]
    pub fn mark(&mut self, phase: usize) {
        if self.at.is_some() {
            self.mark_at(phase, std::time::Instant::now());
        }
    }

    /// [`mark`](Self::mark) with the clock read **supplied rather than taken** —
    /// the seam that holds the whole of the arithmetic, so an arm can drive the
    /// shipped code with decided timestamps.
    #[inline]
    fn mark_at(&mut self, phase: usize, now: std::time::Instant) {
        if let Some(at) = self.at.as_mut() {
            self.ms[phase] += now.duration_since(*at).as_secs_f64() * 1000.0;
            *at = now;
        }
    }

    /// The finished profile, or `None` when the clock was never running.
    pub fn finish(self) -> Option<RecordProfile> {
        self.at.map(|_| RecordProfile { ms: self.ms })
    }
}

/// How many timestamps one frame may write.
///
/// The graph is 31 nodes and the renderer brackets five more segments around it,
/// plus the frame's own origin mark: 37. Sixty-four leaves room for the passes a
/// later phase adds without anyone having to remember this constant exists, and
/// costs 512 bytes of query-set storage.
pub const MAX_FRAME_MARKS: u32 = 64;

/// The frame writes 31 graph nodes + 5 out-of-graph segments + the origin mark.
///
/// A **compile-time** assertion rather than a test, because both sides are
/// constants and clippy is right that a runtime `assert!` over two `const`s is
/// an assertion with a constant value. What it guards is real: a timer that
/// silently drops the tail would report a frame whose segments no longer sum to
/// its total, and a diagnostic that lies is worse than none. The *runtime* twin
/// — that the report names every node the renderer actually built — is
/// `gpu_timing::the_report_names_every_pass_and_the_segments_tile_the_frame`,
/// which reads `EngineRenderer::pass_names` and cannot be folded away.
/// 31 graph nodes (wave VIS1b added `exposure` and `flare`), 5 out-of-graph
/// segments (island wave I4b split `vsm-sync` out of `vsm-raster`), and the
/// frame's origin mark.
const FRAME_MARKS_NEEDED: u32 = 37;
const _: () = assert!(MAX_FRAME_MARKS >= FRAME_MARKS_NEEDED);

/// A per-frame GPU stopwatch over one `wgpu::QuerySet`.
///
/// Single-buffered on purpose: the harness that reads it polls the device every
/// frame anyway (a frame time measured without a sync point is a submission
/// time), so a ring would add latency to a number the caller is already waiting
/// for. [`take`](FrameTimer::take) blocks on the map exactly as
/// `EngineRenderer::vgeom_audit` does.
pub struct FrameTimer {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    read: wgpu::Buffer,
    /// Nanoseconds per timestamp tick, from `Queue::get_timestamp_period`.
    period_ns: f32,
    /// Names of the segments marked so far this frame; `names[i]` describes the
    /// interval that ENDS at timestamp `i + 1`.
    names: Vec<&'static str>,
    /// CPU recording milliseconds per segment, index-aligned with `names`
    /// (island wave I4b).
    cpu_ms: Vec<f64>,
    /// When the segment currently being recorded started, on the CPU.
    at: Option<std::time::Instant>,
    /// Timestamps written this frame (`names.len() + 1` once `begin` has run).
    written: u32,
    /// Whether a resolved frame is sitting in `read` waiting to be taken.
    ready: bool,
}

impl FrameTimer {
    /// Build a timer, or `None` when the device cannot time an encoder segment.
    ///
    /// The `None` is not an error and is not logged as one: a software adapter,
    /// a paravirtual runner and a downlevel driver all legitimately answer it,
    /// and the instrument's contract is that it reports CPU frame time on any
    /// device and per-pass GPU time on the ones that can.
    pub fn new(gpu: &GpuContext) -> Option<Self> {
        if !gpu.supports_timestamp_query() {
            return None;
        }
        let bytes = u64::from(MAX_FRAME_MARKS) * wgpu::QUERY_SIZE as u64;
        // The resolve destination's offset must be 256-aligned; its SIZE need
        // only hold the queries, but rounding up keeps both buffers one shape.
        let bytes = bytes.next_multiple_of(wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT);
        Some(Self {
            set: gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("inf-frame-timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: MAX_FRAME_MARKS,
            }),
            resolve: gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("inf-frame-timestamps-resolve"),
                size: bytes,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            read: gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("inf-frame-timestamps-read"),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            period_ns: gpu.queue.get_timestamp_period(),
            names: Vec::with_capacity(MAX_FRAME_MARKS as usize),
            cpu_ms: Vec::with_capacity(MAX_FRAME_MARKS as usize),
            at: None,
            written: 0,
            ready: false,
        })
    }

    /// Open a frame: clear last frame's marks and stamp the frame's origin.
    pub fn begin(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.names.clear();
        self.cpu_ms.clear();
        self.written = 0;
        self.ready = false;
        encoder.write_timestamp(&self.set, 0);
        self.written = 1;
        self.at = Some(std::time::Instant::now());
    }

    /// Close the segment that has just been recorded and name it.
    ///
    /// Silently drops marks past [`MAX_FRAME_MARKS`] rather than panicking: a
    /// frame that grew a thirty-fifth pass should lose the tail of a diagnostic,
    /// not take the process down. The dropped names are visible as a `passes`
    /// list shorter than the graph.
    pub fn mark(&mut self, encoder: &mut wgpu::CommandEncoder, name: &'static str) {
        if self.written == 0 || self.written >= MAX_FRAME_MARKS {
            return;
        }
        encoder.write_timestamp(&self.set, self.written);
        // The CPU half of the same segment (island wave I4b): the wall clock
        // since the previous mark, which is the time this node spent building
        // commands. Pushed with the name so the two can never fall out of step.
        let now = std::time::Instant::now();
        let since = self.at.unwrap_or(now);
        self.cpu_ms
            .push(now.duration_since(since).as_secs_f64() * 1000.0);
        self.at = Some(now);
        self.names.push(name);
        self.written += 1;
    }

    /// Close the frame: resolve the queries written and stage them for reading.
    pub fn end(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if self.written < 2 {
            return;
        }
        encoder.resolve_query_set(&self.set, 0..self.written, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(
            &self.resolve,
            0,
            &self.read,
            0,
            u64::from(self.written) * wgpu::QUERY_SIZE as u64,
        );
        self.ready = true;
    }

    /// Read the last resolved frame, blocking on the map.
    ///
    /// `None` until a frame has been closed by [`end`](FrameTimer::end); `None`
    /// again once taken, so a caller that reads twice per frame gets the frame
    /// once rather than twice.
    pub fn take(&mut self, gpu: &GpuContext) -> Option<FrameTimings> {
        if !self.ready {
            return None;
        }
        self.ready = false;
        let count = self.written as usize;
        let slice = self
            .read
            .slice(..u64::from(self.written) * wgpu::QUERY_SIZE as u64);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        if gpu
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .is_err()
        {
            return None;
        }
        if !matches!(rx.recv(), Ok(Ok(()))) {
            return None;
        }
        let Ok(data) = slice.get_mapped_range() else {
            return None;
        };
        let mut ticks = Vec::with_capacity(count);
        for chunk in data.chunks_exact(wgpu::QUERY_SIZE as usize).take(count) {
            ticks.push(u64::from_le_bytes(chunk.try_into().unwrap_or([0; 8])));
        }
        drop(data);
        self.read.unmap();

        let ns = f64::from(self.period_ns);
        // A tick counter can wrap and a driver can hand back an out-of-order
        // pair; `saturating_sub` reports such a segment as zero rather than as a
        // nonsense duration, which is the honest reading of "the device did not
        // tell us".
        let ms = |a: u64, b: u64| b.saturating_sub(a) as f64 * ns / 1.0e6;
        let passes = self
            .names
            .iter()
            .enumerate()
            .filter_map(|(i, name)| {
                let (a, b) = (*ticks.get(i)?, *ticks.get(i + 1)?);
                Some(PassTime {
                    name,
                    ms: ms(a, b),
                    cpu_ms: self.cpu_ms.get(i).copied().unwrap_or(0.0),
                })
            })
            .collect();
        Some(FrameTimings {
            total_ms: match (ticks.first(), ticks.last()) {
                (Some(a), Some(b)) => ms(*a, *b),
                _ => 0.0,
            },
            passes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The record phases SUM rather than replace**, and the arithmetic is
    /// driven with decided timestamps and no clock at all.
    ///
    /// That is the I7 CI-red's own lesson, applied to this module the day it was
    /// written rather than after a shared runner found it: the property is
    /// arithmetic, so an arm that reaches it through `sleep` is measuring the
    /// machine. Three unequal stretches, so "sums" is distinguishable from
    /// "keeps the largest".
    #[test]
    fn a_record_phase_marked_twice_sums_rather_than_replaces() {
        use std::time::Duration;
        let base = std::time::Instant::now();
        let mut c = RecordClock {
            at: Some(base),
            ms: [0.0; RECORD_PHASES],
        };
        c.mark_at(record::GRAPH, base + Duration::from_millis(2));
        c.mark_at(record::SUBMIT, base + Duration::from_millis(8));
        c.mark_at(record::GRAPH, base + Duration::from_millis(12));
        let p = c.finish().expect("the clock was running");
        // 2 ms, then 6, then 4 — the graph row carries the first and the third.
        assert!((p.ms[record::GRAPH] - 6.0).abs() < 1.0e-9, "{:?}", p.ms);
        assert!((p.ms[record::SUBMIT] - 6.0).abs() < 1.0e-9, "{:?}", p.ms);
        assert!((p.total_ms() - 12.0).abs() < 1.0e-9, "a mark spilled");
        // …and no other phase moved.
        assert_eq!(
            p.ms.iter().filter(|m| **m != 0.0).count(),
            2,
            "a third phase was charged: {:?}",
            p.ms
        );
    }

    /// A clock that was never armed measures nothing and answers `None`, so a
    /// shipped frame pays one branch per phase and reports no numbers at all.
    #[test]
    fn a_disarmed_record_clock_measures_nothing() {
        let mut c = RecordClock::start(false);
        c.mark(record::GRAPH);
        c.mark(record::SUBMIT);
        assert!(c.finish().is_none());
    }

    /// The names and the indices are one table — a constant that drifted by one
    /// would print the right milliseconds against the wrong phase, which is the
    /// failure this module exists to remove reintroduced one level down.
    #[test]
    fn the_record_names_and_indices_are_one_table() {
        assert_eq!(RECORD_PHASE_NAMES.len(), RECORD_PHASES);
        assert_eq!(RECORD_PHASE_NAMES[record::TARGETS], "targets + luts");
        assert_eq!(RECORD_PHASE_NAMES[record::CLUSTER_WANTS], "cluster wants");
        assert_eq!(RECORD_PHASE_NAMES[record::VSM_RASTER], "vsm raster");
        assert_eq!(RECORD_PHASE_NAMES[record::GRAPH], "graph");
        assert_eq!(RECORD_PHASE_NAMES[record::SUBMIT], "submit");
        assert_eq!(RECORD_PHASE_NAMES[record::EPILOGUE], "epilogue");
        let all = [
            record::TARGETS,
            record::VIEW_UNIFORMS,
            record::ENCODER,
            record::CLUSTER_PLAN,
            record::CLUSTER_WANTS,
            record::VT_STREAM,
            record::CLUSTER_COMMIT,
            record::VSM_SYNC,
            record::VSM_RASTER,
            record::FRAME_DATA,
            record::GRAPH,
            record::VT_FEEDBACK,
            record::VSM_MARK,
            record::SUBMIT,
            record::EPILOGUE,
        ];
        assert_eq!(all.len(), RECORD_PHASES);
        let mut seen = [false; RECORD_PHASES];
        for (i, p) in all.iter().enumerate() {
            assert!(*p < RECORD_PHASES, "constant {i} is {p}, past the slots");
            assert!(!seen[*p], "two record phases share slot {p}");
            seen[*p] = true;
        }
    }

    /// `dearest_first` is a sort, `accumulate`/`scale` are a mean — none of them
    /// loses or invents a phase.
    #[test]
    fn the_record_profile_folds_without_losing_a_phase() {
        let mut a = RecordProfile::default();
        a.ms[record::GRAPH] = 4.0;
        a.ms[record::SUBMIT] = 1.0;
        let mut b = RecordProfile::default();
        b.ms[record::GRAPH] = 2.0;
        b.ms[record::VSM_RASTER] = 3.0;
        a.accumulate(&b);
        assert_eq!(a.total_ms(), 10.0);
        a.scale(0.5);
        assert_eq!(a.total_ms(), 5.0);
        let d = a.dearest_first();
        assert_eq!(d.len(), RECORD_PHASES);
        assert_eq!(d[0], ("graph", 3.0));
        assert_eq!(d[1], ("vsm raster", 1.5));
        assert_eq!(a.rows().count(), RECORD_PHASES);
    }

    /// `by_cost` is a sort and not a re-derivation — the same names, the same
    /// milliseconds, dearest first, ties in submission order.
    #[test]
    fn by_cost_sorts_without_losing_a_segment() {
        let t = FrameTimings {
            total_ms: 6.0,
            passes: vec![
                PassTime {
                    name: "sky",
                    ms: 1.0,
                    cpu_ms: 0.0,
                },
                PassTime {
                    name: "mesh",
                    ms: 4.0,
                    cpu_ms: 0.0,
                },
                PassTime {
                    name: "grid",
                    ms: 0.0,
                    cpu_ms: 0.0,
                },
                PassTime {
                    name: "debug-lines",
                    ms: 0.0,
                    cpu_ms: 0.0,
                },
                PassTime {
                    name: "composite",
                    ms: 1.0,
                    cpu_ms: 0.0,
                },
            ],
        };
        assert_eq!(
            t.by_cost(),
            vec![
                ("mesh", 4.0),
                ("sky", 1.0),
                ("composite", 1.0),
                ("grid", 0.0),
                ("debug-lines", 0.0),
            ]
        );
        let sum: f64 = t.passes.iter().map(|p| p.ms).sum();
        assert_eq!(sum, t.total_ms, "the segments must tile the frame");
    }
}
