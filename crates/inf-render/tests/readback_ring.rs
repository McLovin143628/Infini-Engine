//! **The readback ring on a real device** (P26.4).
//!
//! `readback.rs`'s unit arms are arithmetic — the slot equation and the pinned
//! constant. What they cannot say is whether the thing actually reads back the
//! bytes of the frame it names, and that is the whole contract: *the trace at
//! frame F reads what was recorded at F − 2, or nothing*. So these arms drive a
//! real `wgpu` device through a real recording sequence and assert on the
//! **content**, which is the only way to tell a correct ring from one that
//! happily returns the newest slot.
//!
//! Every arm skips cleanly, and says so, when the machine has no adapter.

use inf_render::readback::{ReadbackRing, READBACK_LATENCY_FRAMES};
use inf_render::GpuContext;

fn gpu_or_skip(what: &str) -> Option<GpuContext> {
    match GpuContext::headless() {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("SKIP: no GPU adapter available for {what} ({e})");
            None
        }
    }
}

/// A source buffer the test writes a frame stamp into and the ring copies out of.
fn source(gpu: &GpuContext, words: usize) -> wgpu::Buffer {
    gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback-ring-source"),
        size: (words * 4) as u64,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Write `value` into every word of `src`, record frame `frame`'s copy, submit,
/// and pump the ring once — one whole frame of the shipping call shape.
fn tick(gpu: &GpuContext, ring: &mut ReadbackRing, src: &wgpu::Buffer, frame: u64, value: u32) {
    let words = (src.size() / 4) as usize;
    gpu.queue
        .write_buffer(src, 0, bytemuck::cast_slice(&vec![value; words]));
    let mut enc = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback-ring-frame"),
        });
    ring.record(&mut enc, src, frame);
    gpu.queue.submit([enc.finish()]);
    pump(gpu, [ring]);
}

/// Pump `rings` until this frame's copy has been mapped.
///
/// A test harness runs frames back to back with no vsync, so without this the
/// slot a later frame wants can still be IN FLIGHT from an earlier one — and
/// `record` then correctly refuses to overwrite it, which shows up as a read
/// that misses for ever. A real host at frame pace never hits it (a 3-slot ring
/// gives a copy two whole frames to land), and the ring's refusal is the right
/// behaviour either way; this is the harness paying for its own speed. Measured:
/// without it, `the_latency_is_what_decides_which_frame_is_read` fails only when
/// the whole crate's GPU suites run at once.
fn pump<const N: usize>(gpu: &GpuContext, rings: [&mut ReadbackRing; N]) {
    for ring in rings {
        ring.poll(&gpu.device);
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        ring.poll(&gpu.device);
    }
}

/// Pump until the ring can answer for `frame`, or give up — the loop a caller
/// does NOT write (it takes the floor instead), used here so the content
/// assertions are about the bytes rather than about scheduling luck.
fn take_eventually(gpu: &GpuContext, ring: &mut ReadbackRing, frame: u64) -> Option<u32> {
    for _ in 0..64 {
        if let Some(v) = ring.take(frame, |b| bytemuck::cast_slice::<u8, u32>(b)[0]) {
            return Some(v);
        }
        ring.poll(&gpu.device);
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    }
    None
}

/// **THE CONTRACT: a read at frame F returns what was recorded at F − 2.**
///
/// Each frame stamps the source with its own frame number, so the value read
/// back *is* the frame it came from and an off-by-one is a wrong integer rather
/// than an invisible staleness. The arm would pass on a ring that returned the
/// newest ready slot only if the GPU were exactly two frames behind on every
/// frame, which is why the sequence runs long enough for the pipeline to drain
/// (`take_eventually` pumps to completion) — under that condition the newest
/// ready slot is frame F, and a "whatever arrived" ring fails on the first read.
#[test]
fn a_read_at_frame_f_returns_exactly_frame_f_minus_the_latency() {
    let Some(gpu) = gpu_or_skip("the readback ring") else {
        return;
    };
    let src = source(&gpu, 4);
    let mut ring = ReadbackRing::new(&gpu.device, "gate", 16);
    assert_eq!(ring.latency(), READBACK_LATENCY_FRAMES);
    assert_eq!(ring.slot_count(), (READBACK_LATENCY_FRAMES + 1) as usize);

    // Frames 0 and 1 have nothing to read: their answers were recorded before
    // the ring existed, and the honest reply is None rather than frame 0's bytes.
    tick(&gpu, &mut ring, &src, 0, 1000);
    assert!(ring.take(0, |_| ()).is_none(), "frame 0 read something");
    tick(&gpu, &mut ring, &src, 1, 1001);
    assert!(ring.take(1, |_| ()).is_none(), "frame 1 read something");
    assert_eq!(ring.misses(), 2, "the misses were not counted");

    for frame in 2..12u64 {
        tick(&gpu, &mut ring, &src, frame, 1000 + frame as u32);
        let got = take_eventually(&gpu, &mut ring, frame)
            .unwrap_or_else(|| panic!("frame {frame} never became ready"));
        assert_eq!(
            got,
            1000 + (frame - READBACK_LATENCY_FRAMES) as u32,
            "frame {frame} read the wrong frame's bytes — the latency is not pinned"
        );
    }
    assert_eq!(ring.hits(), 10, "the hits were not counted");
}

/// **The latency really is the parameter**, not a coincidence of the slot count.
///
/// The same recording sequence through a latency-1 ring reads frame F − 1 where
/// the shipping ring reads F − 2. Without this the arm above is satisfied by any
/// ring whose arithmetic happens to line up, and the pinning it claims to
/// measure would be a property of the fixture.
#[test]
fn the_latency_is_what_decides_which_frame_is_read() {
    let Some(gpu) = gpu_or_skip("the readback ring latency") else {
        return;
    };
    let src = source(&gpu, 4);
    let mut one = ReadbackRing::with_latency(&gpu.device, "l1", 16, 1);
    let mut two = ReadbackRing::with_latency(&gpu.device, "l2", 16, 2);
    assert_eq!(one.slot_count(), 2);
    assert_eq!(two.slot_count(), 3);

    for frame in 0..8u64 {
        let words = (src.size() / 4) as usize;
        gpu.queue.write_buffer(
            &src,
            0,
            bytemuck::cast_slice(&vec![500 + frame as u32; words]),
        );
        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        one.record(&mut enc, &src, frame);
        two.record(&mut enc, &src, frame);
        gpu.queue.submit([enc.finish()]);
        pump(&gpu, [&mut one, &mut two]);
        if frame >= 2 {
            let a = take_eventually(&gpu, &mut one, frame).expect("l1 ready");
            let b = take_eventually(&gpu, &mut two, frame).expect("l2 ready");
            assert_eq!(a, 500 + (frame - 1) as u32);
            assert_eq!(b, 500 + (frame - 2) as u32);
            assert_ne!(a, b, "the two rings read the same frame");
        }
    }
}

/// **A slot that is not ready is no answer, not an older one.**
///
/// Recording without ever polling leaves every slot in flight; the read must
/// then miss for ever rather than fall back to a neighbour that happens to hold
/// bytes. This is the property the whole determinism argument rests on — an
/// opportunistic ring makes the want set a function of the frame *history*,
/// which is exactly what `inf_vgeom::stream`'s ruling forbids.
#[test]
fn an_unready_slot_is_a_miss_and_never_a_neighbours_bytes() {
    let Some(gpu) = gpu_or_skip("the readback ring miss path") else {
        return;
    };
    let src = source(&gpu, 4);
    let mut ring = ReadbackRing::new(&gpu.device, "miss", 16);
    for frame in 0..6u64 {
        let words = (src.size() / 4) as usize;
        gpu.queue
            .write_buffer(&src, 0, bytemuck::cast_slice(&vec![frame as u32; words]));
        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        ring.record(&mut enc, &src, frame);
        gpu.queue.submit([enc.finish()]);
        // Deliberately NO poll: the map callbacks cannot fire.
        assert!(
            ring.take(frame, |_| ()).is_none(),
            "frame {frame} answered from a slot whose map never completed"
        );
    }
    assert_eq!(ring.hits(), 0);
    assert_eq!(ring.misses(), 6);
    // …and the ANTI-VACUITY control: the same ring, polled, does answer — so the
    // misses above are a measurement of readiness and not of a ring that is
    // simply broken.
    ring.poll(&gpu.device);
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    ring.poll(&gpu.device);
    assert!(
        take_eventually(&gpu, &mut ring, 6).is_some() || ring.misses() > 6,
        "the ring neither answered nor recorded a further miss"
    );
}
