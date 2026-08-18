//! **The `.inf_sm` text sidecar, as a door** (P29.6) — pillar S1's other half.
//!
//! [`inf_anim::text`] makes a machine reviewable; this makes it **editable**.
//! Without a save path the text is a projection an author may read and not
//! change, which is most of the way to the binary `.uasset` §13's opening
//! finding is about: the point of a diffable machine is that the diff is how you
//! author it, on a branch, in a review, from a merge.
//!
//! # The rule: the payload is written FROM the text, never beside it
//!
//! [`save_from_text`] parses, **validates** and re-encodes. It is the same door
//! `sm_save` puts a machine through (P29.2's A1: the State Machine editor could
//! write a file its own reader refuses), and it is deliberately not "write the
//! text and let something else catch up" — a `.inf_sm` whose text and payload
//! disagree is two machines with one name, and the P29.6 gate asserts they never
//! do for the committed sample.
//!
//! # Where the file lives
//!
//! `<name>.inf_sm` → `<name>.inf_sm.txt`. Beside the asset's own
//! `<name>.inf_sm.toml` metadata sidecar rather than inside it, because the two
//! answer different questions: the sidecar is what the *database* knows about the
//! asset (its GUID, its hash, its dependencies) and this is what the *author*
//! wrote.

use std::path::{Path, PathBuf};

use inf_anim::{StateMachine, StateMachineAsset};

/// Why a text-form save or load refused. Every variant carries the underlying
/// door's own message, because that text is what reaches the author.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SmTextError {
    /// The text is not a machine.
    #[error("{0}")]
    Text(String),
    /// It parses and the machine is not usable — the same refusal `sm_save`
    /// makes.
    #[error("{0}")]
    Invalid(String),
    /// The filesystem said no.
    #[error("{0}")]
    Io(String),
}

/// `foo.inf_sm` → `foo.inf_sm.txt`.
pub fn text_path(payload_path: &Path) -> PathBuf {
    let mut s = payload_path.as_os_str().to_os_string();
    s.push(".txt");
    PathBuf::from(s)
}

/// Write the text face beside a payload — what the character wizard does, and
/// what a save through the State Machine editor should do.
pub fn write_text(payload_path: &Path, machine: &StateMachine) -> Result<(), SmTextError> {
    inf_asset::write_atomically(
        &text_path(payload_path),
        inf_anim::to_toml(machine).as_bytes(),
    )
    .map_err(|e| SmTextError::Io(e.to_string()))
}

/// Read the text face beside a payload, if one is there.
pub fn read_text(payload_path: &Path) -> Option<String> {
    std::fs::read_to_string(text_path(payload_path)).ok()
}

/// **Author a machine by editing its text.**
///
/// Parses `text`, validates the machine, re-encodes the `.inf_sm` payload and
/// rewrites the text from the machine that was actually stored — so a file that
/// round-trips to something slightly different (a state referenced by index
/// because two share a name, a float that reads back shorter) ends up saying
/// exactly what was saved. `skeleton` is carried through because it is the
/// asset's binding and not the machine's.
///
/// Answers the machine it stored.
pub fn save_from_text(
    payload_path: &Path,
    text: &str,
    skeleton: Option<[u8; 16]>,
) -> Result<StateMachine, SmTextError> {
    let machine = inf_anim::from_toml(text).map_err(|e| SmTextError::Text(e.to_string()))?;
    machine
        .validate()
        .map_err(|e| SmTextError::Invalid(e.to_string()))?;
    let bytes = inf_asset::encode(&StateMachineAsset::new(machine.clone(), skeleton))
        .map_err(|e| SmTextError::Io(e.to_string()))?;
    inf_asset::write_atomically(payload_path, &bytes)
        .map_err(|e| SmTextError::Io(e.to_string()))?;
    write_text(payload_path, &machine)?;
    Ok(machine)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> StateMachineAsset {
        let m = inf_anim::StateMachine {
            states: vec![
                inf_anim::SmState::clip("idle", [1; 16]),
                inf_anim::SmState::clip("walk", [2; 16]),
            ],
            transitions: vec![inf_anim::SmTransition::on(
                0,
                1,
                0.15,
                "speed",
                inf_anim::CmpOp::Gt,
                0.5,
            )],
            entry: 0,
            params: vec![inf_anim::SmParam::float("speed")],
            profiles: Vec::new(),
        };
        StateMachineAsset::new(m, Some([9; 16]))
    }

    /// **The loop closes**: write the text, edit one line, save, and the payload
    /// beside it is the edited machine.
    #[test]
    fn a_one_line_edit_to_the_text_is_a_saved_machine() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("Locomotion.inf_sm");
        let asset = fixture();
        std::fs::write(&path, inf_asset::encode(&asset).unwrap()).unwrap();
        write_text(&path, &asset.machine).unwrap();

        let text = read_text(&path).expect("the text is beside it");
        assert!(text.contains("duration = 0.15"));
        let edited = text.replace("duration = 0.15", "duration = 0.42");

        let stored = save_from_text(&path, &edited, asset.skeleton).expect("it saves");
        assert_eq!(stored.transitions[0].duration, 0.42);

        // The PAYLOAD is the edited machine, read back through the shipped door.
        let back: StateMachineAsset = inf_asset::decode(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(back.machine, stored);
        assert_eq!(back.skeleton, asset.skeleton, "the binding survived");
        // …and the text was rewritten from what was stored, so the two faces
        // cannot drift apart across a save.
        assert_eq!(
            inf_anim::from_toml(&read_text(&path).unwrap()).unwrap(),
            stored
        );
    }

    /// **A machine the reader would refuse is not written** — P29.2's A1, from
    /// the text side.
    #[test]
    fn text_that_makes_an_unusable_machine_is_refused_and_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("Locomotion.inf_sm");
        let asset = fixture();
        let original = inf_asset::encode(&asset).unwrap();
        std::fs::write(&path, &original).unwrap();
        write_text(&path, &asset.machine).unwrap();

        // A transition with a negative duration — the exact thing `validate`
        // refuses, expressed as one line of text.
        let bad = read_text(&path)
            .unwrap()
            .replace("duration = 0.15", "duration = -1.0");
        let err = save_from_text(&path, &bad, asset.skeleton).expect_err("it refuses");
        assert!(matches!(err, SmTextError::Invalid(_)), "{err:?}");
        assert_eq!(
            std::fs::read(&path).unwrap(),
            original,
            "a refused save rewrote the payload"
        );

        // …and so is text that is not a machine at all.
        let err = save_from_text(&path, "this is not toml", asset.skeleton).expect_err("refuses");
        assert!(matches!(err, SmTextError::Text(_)), "{err:?}");
        assert_eq!(std::fs::read(&path).unwrap(), original);
    }

    /// The text path is beside the payload and beside — not instead of — the
    /// asset's own metadata sidecar.
    #[test]
    fn the_text_sits_beside_the_sidecar_rather_than_in_it() {
        let p = Path::new("Content/Characters/Hero Locomotion.inf_sm");
        assert_eq!(
            text_path(p),
            Path::new("Content/Characters/Hero Locomotion.inf_sm.txt")
        );
        assert_ne!(
            text_path(p),
            Path::new("Content/Characters/Hero Locomotion.inf_sm.toml")
        );
    }
}
