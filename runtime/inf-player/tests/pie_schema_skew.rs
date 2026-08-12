//! **A version-skewed editor is refused by NAME, not by a clean exit** (P24.4,
//! closing the P24.1 carried entry).
//!
//! # What was measured, and why it looked fine
//!
//! `ScenePayload::check_version` refuses a mismatched envelope in both
//! directions and is unit-tested in both — but only the *newer* direction was
//! reachable end to end. A payload from an **older** build is a **short read**:
//! the envelope is positional bincode, an older build writes fewer tail fields,
//! and `decode_from_slice` runs off the end inside
//! `inf_runtime::pie::read_msg`. The player's reader thread was
//! `while let Ok(msg) = read_msg(..)`, so any error — a clean EOF and a decode
//! failure alike — **broke the loop**, dropped the channel, and the main loop
//! printed "editor closed the channel; exiting" and returned `ExitCode::SUCCESS`.
//! `check_version` never ran, because it lives inside
//! `build_world_from_payload`, which that path never reaches.
//!
//! To the editor a version-skewed pair therefore looked like a player that
//! started, said `Ready`, and quietly went away — indistinguishable from a user
//! pressing Stop.
//!
//! # The fix, and what this file holds it to
//!
//! The reader thread now distinguishes `UnexpectedEof` (the editor went away —
//! the ordinary end of a session) from every other error (the stream said
//! something this build cannot read). A decode fault is reported over its own
//! channel, and the loop answers with a `PlayerToEditor::Error` naming the
//! schema before exiting **non-zero**.
//!
//! Four arms, and the last two are what keep the first two honest:
//!
//! * a stale payload gets a named `Error` and a failing exit;
//! * a **current** payload does not (or "the player always errors" would satisfy
//!   the first arm);
//! * a session ended with `Stop` still exits **0** (or "the player always fails"
//!   would satisfy both);
//! * **a pipe that simply CLOSES still exits 0, silently** — the other half of
//!   the split, and the one the `Stop` arm cannot see. `Stop` returns from the
//!   loop through its own arm and never reaches `report_stream_end`, so with
//!   only the three arms above the whole `UnexpectedEof` classification could be
//!   deleted — every clean editor disconnect answered with a bogus schema
//!   `Error` and a failing exit — and every test in this repository stayed
//!   green. Measured (P24.4 audit F2), not reasoned about.

use std::io::{BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdout, Command, Stdio};
use std::time::Duration;

use inf_runtime::pie::{read_msg, write_msg, EditorToPlayer, PlayerToEditor, ScenePayload};
use serde::Serialize;
use uuid::Uuid;

fn player_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_inf-player"))
}

/// **A v7 `ScenePayload`, field for field.**
///
/// The real envelope is v8; v8's additions were `cloths` / `hairs` / `materials`
/// / `textures`, all at the tail. So this is exactly what the previous build
/// writes — the same fields in the same order, four short — and that is what
/// makes it a *short read* rather than a value `check_version` could have
/// refused.
///
/// **Kept at exactly one version behind** (P26.3b): the fixture was the v6 shape
/// while the wire was v7, and it stays one rung down as the wire moves, because
/// "an editor built from the previous commit" is the skew a user actually hits
/// and a two-rung gap could pass for a different failure.
///
/// Hand-written rather than derived from the real type on purpose: a shadow
/// struct generated from the current one would grow with it and stop being
/// stale, which is the whole point of the fixture.
#[derive(Serialize)]
struct StaleScenePayload {
    schema_version: u32,
    label: String,
    level_bytes: Vec<u8>,
    classes: Vec<(Uuid, Vec<u8>)>,
    pcgs: Vec<(Uuid, Vec<u8>)>,
    skeletons: Vec<(Uuid, Vec<u8>)>,
    clips: Vec<(Uuid, Vec<u8>)>,
    machines: Vec<(Uuid, Vec<u8>)>,
    biome_sets: Vec<(Uuid, Vec<u8>)>,
    tick_hz: u32,
    windowed: bool,
    voxels: Vec<(Uuid, Vec<u8>)>,
    terrains: Vec<(Uuid, Vec<u8>)>,
    fractures: Vec<(Uuid, Vec<u8>)>,
    /// v7's one addition, at the tail. Present here because this fixture is a
    /// **v7** payload now — v8 is the current wire.
    meshes: Vec<(Uuid, Vec<u8>)>,
}

/// The message wrapper, with `LoadScene` at **its real discriminant**.
///
/// bincode indexes an enum by position, so `Load` has to be here — unit-shaped,
/// never sent — for `LoadScene` to land on index 1 the way the real wire does.
#[derive(Serialize)]
// The variants are deliberately lopsided: `Load` is a placeholder that exists
// only to hold index 0, so `LoadScene` lands on the discriminant the real wire
// uses. Boxing it would change nothing on the wire and would obscure that.
#[allow(clippy::large_enum_variant)]
enum StaleEditorToPlayer {
    #[allow(dead_code)]
    Load,
    LoadScene(StaleScenePayload),
}

fn stale_payload() -> StaleScenePayload {
    StaleScenePayload {
        // A DECREMENTED version, as the ledger entry asks for. It is never read:
        // the frame does not survive its own decode, which is the finding.
        schema_version: inf_runtime::pie::SCENE_PAYLOAD_VERSION - 1,
        label: "Stale".into(),
        level_bytes: Vec::new(),
        classes: Vec::new(),
        pcgs: Vec::new(),
        skeletons: Vec::new(),
        clips: Vec::new(),
        machines: Vec::new(),
        biome_sets: Vec::new(),
        tick_hz: 60,
        windowed: false,
        voxels: Vec::new(),
        terrains: Vec::new(),
        fractures: Vec::new(),
        meshes: Vec::new(),
    }
}

/// Spawn `inf-player --pie` and consume its `Ready` handshake.
fn spawn_ready() -> (Child, BufReader<ChildStdout>) {
    let mut child = Command::new(player_bin())
        .arg("--pie")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("the player spawns");
    let stdout = child.stdout.take().expect("stdout piped");
    let mut reader = BufReader::new(stdout);
    let ready: PlayerToEditor = read_msg(&mut reader).expect("the player says Ready");
    assert!(
        matches!(ready, PlayerToEditor::Ready { .. }),
        "handshake was {ready:?}"
    );
    (child, reader)
}

/// Wait for the child, killing it if it outlives `secs` — a hang is a failure
/// with a message rather than a test that never returns.
fn wait_bounded(child: &mut Child, secs: u64) -> std::process::ExitStatus {
    let deadline = std::time::Instant::now() + Duration::from_secs(secs);
    loop {
        if let Some(status) = child.try_wait().expect("try_wait") {
            return status;
        }
        if std::time::Instant::now() > deadline {
            let _ = child.kill();
            panic!("the player did not exit within {secs}s of the frame it could not read");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// **THE ARM the P24.1 ledger entry asked for.** A payload from an older build
/// draws a `PlayerToEditor::Error` naming the schema, and a non-zero exit.
#[test]
fn a_stale_payload_is_refused_by_name_rather_than_exiting_cleanly() {
    let (mut child, mut reader) = spawn_ready();
    {
        let mut stdin = child.stdin.take().expect("stdin piped");
        write_msg(&mut stdin, &StaleEditorToPlayer::LoadScene(stale_payload()))
            .expect("the stale frame is written");
        stdin.flush().ok();
        // Dropped here: the editor is not going away, but a real one would keep
        // writing and the player must answer before it reads anything else.
    }

    let ev: PlayerToEditor = read_msg(&mut reader).expect(
        "the player closed its stream without saying anything — this is the \
         pre-P24.4 behaviour the whole entry is about",
    );
    let PlayerToEditor::Error { message } = ev else {
        panic!("expected an Error naming the schema, got {ev:?}");
    };
    assert!(
        message.to_lowercase().contains("schema"),
        "the refusal does not name the SCHEMA, so an author reading it learns \
         nothing about why: {message}"
    );
    // …and it is READABLE. The B1 law, which this repository has paid for twice:
    // a scripted edit to a Rust string literal eats the `\` continuation and
    // leaves the indentation in the message. `contains("schema")` is perfectly
    // happy with that, so the assertion has to be its own (P24.4 audit F1).
    assert!(
        !message.contains("  "),
        "the refusal carries a run of spaces — the continuation was eaten: \
         {message:?}"
    );

    let status = wait_bounded(&mut child, 10);
    assert!(
        !status.success(),
        "the player exited 0 after refusing a frame it could not read — which is \
         exactly the silent 'clean exit' this arm exists to retire"
    );
}

/// **The control**: a payload this build DOES speak is loaded, not refused.
///
/// Without it, a player that answered `Error` to everything would satisfy the
/// arm above perfectly.
#[test]
fn a_current_payload_is_loaded_and_not_refused() {
    let (mut child, mut reader) = spawn_ready();
    let mut stdin = child.stdin.take().expect("stdin piped");
    let payload = ScenePayload::new(
        "Current",
        inf_editor_core::scene::serialize::encode(
            &inf_editor_core::scene::serialize::to_scene_file(
                &inf_editor_core::scene::SceneDoc::new(),
            ),
        )
        .expect("an empty level encodes"),
        Vec::new(),
        60,
        false,
    );
    write_msg(&mut stdin, &EditorToPlayer::LoadScene(Box::new(payload)))
        .expect("the current frame is written");
    stdin.flush().ok();

    let ev: PlayerToEditor = read_msg(&mut reader).expect("the player answers");
    assert!(
        matches!(ev, PlayerToEditor::Loaded { .. }),
        "a payload of this build's own version was not loaded: {ev:?}"
    );

    // …and a `Stop`ped session still exits ZERO. The third arm, in the same
    // session: "the player always fails" would satisfy both of the others.
    //
    // Note what this does NOT cover, which is why the fourth arm exists: `Stop`
    // leaves the loop through `Control::Exit` and never reaches
    // `report_stream_end`, so the `UnexpectedEof` classification is untouched by
    // it.
    write_msg(&mut stdin, &EditorToPlayer::Stop).ok();
    drop(stdin);
    let status = wait_bounded(&mut child, 10);
    assert!(
        status.success(),
        "a well-formed session no longer exits 0 — the fault path is firing on a \
         clean shutdown"
    );
}

/// **A pipe that simply closes is not a fault** (P24.4 audit F2).
///
/// The other half of the split the fix introduced, and the half nothing was
/// measuring. `report_stream_end` reads a fault queue and answers `SUCCESS` when
/// it is empty; the arm above never reaches it, because `Stop` returns through
/// its own arm. So deleting the `UnexpectedEof` case from the reader thread —
/// making every ordinary editor disconnect a "schema" refusal with a failing
/// exit — left the whole battery green. Measured that way, then this was
/// written; with it, the same severing fails here by name.
///
/// The editor closing the pipe is how a PIE session normally ends, so the
/// regression this catches is not exotic: it is every Stop-by-window-close.
#[test]
fn a_closed_pipe_with_no_fault_exits_zero_and_says_nothing() {
    use std::io::Read;

    let (mut child, mut reader) = spawn_ready();
    // No `Stop`, no frame at all: the editor just went away.
    drop(child.stdin.take().expect("stdin piped"));

    let status = wait_bounded(&mut child, 10);
    assert!(
        status.success(),
        "a closed pipe exited {:?} — an ordinary editor disconnect is being \
         reported as a decode fault",
        status.code()
    );

    // …and it said nothing on the protocol stream. A refusal the editor never
    // asked for is worse than the silent exit this batch retired: it would put a
    // schema error in front of a user who simply pressed Stop.
    let mut rest = Vec::new();
    reader.read_to_end(&mut rest).expect("drain stdout");
    assert!(
        rest.is_empty(),
        "the player wrote {} bytes after a clean close — an `Error` frame on a \
         disconnect that had no fault in it",
        rest.len()
    );
}
