//! **Two processes derive the same clip into the same bytes** (P29.5, pillar S2).
//!
//! Derivation writes into a committed `.inf_anim`: the root-motion track, the
//! distance track, the foot-plant markers, the footstep notifies and five curve
//! channels. Those bytes go into a project, into a cook, into a pack, and into
//! every determinism gate downstream — and each of those gates compares one
//! machine's answer with *its own*. So a derivation that depended on anything
//! per-process would put two different clip files in two developers' checkouts
//! and every gate would go on passing, because each machine agrees with itself.
//!
//! A single-process arm cannot see that. This is the twin.
//!
//! # Two halves, and the second is the sharper one
//!
//! 1. **Generate and derive in a fresh process.** The fixture rig and its walk
//!    cycle are themselves generated (`build_template` + `build_locomotion`), so
//!    this compares the whole chain — a generator that hashed a pointer or read
//!    a clock would fail here.
//! 2. **Derive bytes the parent produced.** The source clip is encoded, handed
//!    over as a file, decoded in the child and derived there. A difference in
//!    this half is the *derivation* and nothing upstream of it, which is what
//!    makes the failure diagnosable rather than merely visible.
//!
//! # Why a subprocess and not two threads
//!
//! P29.2 shipped a mechanism whose only arm ran in-process and its ledger said
//! so; the P29.4 wave closed that boundary for the machine and this closes it
//! for the content. Address-space layout, allocator addresses, hash seeds and
//! iteration order are all per-process state — two threads share every one of
//! them, so two threads agreeing proves nothing about two machines.

use std::process::Command;

/// The env var that turns this binary into the probe. Its value is the path of
/// the encoded source clip for half 2, or empty for half 1.
const PROBE: &str = "INF_P295_DERIVE_PROBE";

fn rig() -> inf_anim::SkeletonAsset {
    inf_anim::build_template(inf_anim::BodyPlan::Biped, &inf_anim::BodyParams::default()).unwrap()
}

fn source() -> inf_anim::AnimClip {
    inf_anim::build_locomotion(
        inf_anim::BodyPlan::Biped,
        &rig(),
        &inf_anim::locomotion::GaitParams::default(),
    )
    .unwrap()
    .walk
}

/// The derived clip's bytes, as hex — the whole payload, not a summary, so a
/// single moved float in a single curve key is a different string.
fn derived_hex(clip: &inf_anim::AnimClip) -> String {
    let (out, _) =
        inf_anim::derive_clip(clip, &rig().skeleton, &inf_anim::DeriveOptions::default())
            .expect("the fixture derives");
    let bytes = bincode::serde::encode_to_vec(&out, inf_asset::bincode_config()).unwrap();
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The child half. Ignored by name so a normal run never executes it; the parent
/// runs it by exact filter with [`PROBE`] set.
#[test]
#[ignore = "the subprocess half of anim_derivation_twin, run by its parent"]
fn anim_derive_probe() {
    let path = std::env::var(PROBE).expect("the probe env var");
    let clip = if path.is_empty() {
        // Half 1: generate the fixture here, in this process.
        source()
    } else {
        // Half 2: derive exactly the bytes the parent handed over.
        let bytes = std::fs::read(&path).expect("read the source clip");
        let (clip, _): (inf_anim::AnimClip, usize) =
            bincode::serde::decode_from_slice(&bytes, inf_asset::bincode_config())
                .expect("the source clip decodes");
        clip
    };
    println!("derived={}", derived_hex(&clip));
}

/// Run the probe and answer what it derived.
fn probe(arg: &str) -> String {
    let exe = std::env::current_exe().expect("this test binary");
    let out = Command::new(exe)
        .args(["anim_derive_probe", "--exact", "--ignored", "--nocapture"])
        .env(PROBE, arg)
        .output()
        .expect("spawn the derivation probe");
    assert!(
        out.status.success(),
        "the derivation probe failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout)
        .expect("probe stdout utf8")
        .lines()
        .find_map(|l| l.strip_prefix("derived=").map(str::to_string))
        .expect("the probe printed derived=")
}

#[test]
fn a_second_process_derives_the_same_clip_into_the_same_bytes() {
    let mine = derived_hex(&source());

    // Anti-vacuity: the hex is a hash of something substantial. Two empty clips
    // compare equal perfectly.
    assert!(
        mine.len() > 4096,
        "the derived clip encodes to {} hex chars — too small to be a walk cycle \
         with a root-motion track, a distance track, markers and five channels",
        mine.len()
    );

    // Half 1: the whole chain, generated afresh over there.
    assert_eq!(
        probe(""),
        mine,
        "a freshly generated and derived clip differs between processes — \
         something on the generate-or-derive path depends on per-process state"
    );

    // Half 2: the derivation alone, over bytes this process produced.
    let dir = std::env::temp_dir();
    let path = dir.join(format!("inf_p295_twin_{}.bin", std::process::id()));
    let bytes =
        bincode::serde::encode_to_vec(source(), inf_asset::bincode_config()).expect("encode");
    std::fs::write(&path, &bytes).expect("write the source clip");
    let theirs = probe(path.to_str().expect("a utf8 temp path"));
    let _ = std::fs::remove_file(&path);
    assert_eq!(
        theirs, mine,
        "the SAME source bytes derived differently in a fresh process — the \
         defect is in the derivation itself"
    );
}
