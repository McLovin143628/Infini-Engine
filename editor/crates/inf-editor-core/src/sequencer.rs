//! Sequencer seed — a simple property-track timeline (P11.4).
//!
//! A [`Sequence`] animates **f64 scalar tracks** over time: each
//! [`PropertyTrack`] targets one entity ([`Uuid`] guid) and one reflection
//! property *scalar* (a path like `"Transform.translation.x"`), holding a sorted
//! list of [`Key`] samples with `Step`/`Linear` interpolation. Sampling at a time
//! `t` yields `(target, path, value)` triples that the scrub layer applies through
//! the existing reflection edit path.
//!
//! **v1 scope (deliberately small but real).**
//! - **Scalar tracks only.** Vector fields (`Transform.translation`) are addressed
//!   **per component** — `translation.x`, `translation.y`, `translation.z` — each a
//!   separate scalar track. Color / bool / enum / multi-scalar tracks and
//!   Bezier/eased curves are deferred (documented follow-ups).
//! - **Path grammar:** `<Component>.<field>[.<swizzle>]` where `<Component>` is the
//!   Details **display name** (`registry().type_path_for`), `<field>` is the reflect
//!   field key, and the optional `<swizzle>` (`x|y|z|w` or `r|g|b|a`, index 0..=3)
//!   selects one component of a vector field. Two segments → a scalar (`Number`)
//!   field; three → one component of a `Vec3` field.
//!
//! **Persistence (v1).** Sequences are **project-local** files under
//! `<root>/.infinity/sequences/<slug>.toml`, mirroring the sorting-layer / settings
//! sidecar discipline (deterministic, stable-order TOML, byte-identical re-emit).
//! They are *not* a new asset kind — the `.inf_seq` asset (content-addressed, in the
//! dependency graph) is the full-sequencer follow-up.
//!
//! **Scrub vs. keys.** Scrubbing is a **non-undoable live preview**: it writes
//! sampled values through [`SceneDoc::write_prop_preview`] (bumps the version so the
//! viewport re-syncs, but never dirties the document) and the scrub session restores
//! the captured originals on stop, so previewing never permanently changes the scene.
//! Editing a **key** ([`set_key`] / [`capture_key`]) mutates the *sequence file*, not
//! the scene, so it is non-undoable in v1 (documented). A deliberate "bake this
//! sampled value into the scene" would instead go through the normal undoable
//! `edit_set_prop` path — not wired as a command in the seed.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use inf_ecs::PropValue;

use crate::scene::SceneDoc;

/// How a segment between two keys is interpolated. The **left** key's `interp`
/// governs the segment to its right.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Interp {
    /// Hold the left key's value until the next key.
    Step,
    /// Linearly blend left→right across the segment.
    #[default]
    Linear,
}

/// One keyframe: a time (seconds) and a scalar value, plus the interpolation that
/// governs the segment starting at this key.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Key {
    pub t: f64,
    pub value: f64,
    #[serde(default)]
    pub interp: Interp,
}

/// One scalar property track: an entity guid + a reflection scalar path + its
/// sorted keys.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PropertyTrack {
    /// The animated entity's stable guid.
    pub target: Uuid,
    /// The reflection scalar path, e.g. `"Transform.translation.x"`.
    pub path: String,
    /// Keyframes, kept sorted by `t` with unique times (see [`Sequence::normalize`]).
    #[serde(default)]
    pub keys: Vec<Key>,
}

impl PropertyTrack {
    pub fn new(target: Uuid, path: impl Into<String>) -> Self {
        Self {
            target,
            path: path.into(),
            keys: Vec::new(),
        }
    }

    /// Insert or replace the key at time `t` (times are unique per track). The
    /// `interp` becomes that key's segment mode. Returns the track for chaining.
    pub fn set_key(&mut self, t: f64, value: f64, interp: Interp) {
        match self.keys.iter_mut().find(|k| k.t == t) {
            Some(k) => {
                k.value = value;
                k.interp = interp;
            }
            None => self.keys.push(Key { t, value, interp }),
        }
        self.sort_keys();
    }

    /// Remove the key nearest to `t` within `eps` seconds. Returns whether one was
    /// removed.
    pub fn remove_key_near(&mut self, t: f64, eps: f64) -> bool {
        let Some((i, _)) = self
            .keys
            .iter()
            .enumerate()
            .filter(|(_, k)| (k.t - t).abs() <= eps)
            .min_by(|(_, a), (_, b)| {
                (a.t - t)
                    .abs()
                    .partial_cmp(&(b.t - t).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        else {
            return false;
        };
        self.keys.remove(i);
        true
    }

    fn sort_keys(&mut self) {
        self.keys
            .sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
        // De-duplicate equal times (last authored wins — it was just pushed/sorted
        // stably after the earlier one).
        self.keys.dedup_by(|a, b| a.t == b.t);
    }

    /// Sample this track at time `t`, holding at the endpoints. `None` when the
    /// track has no keys.
    pub fn sample(&self, t: f64) -> Option<f64> {
        let keys = &self.keys;
        let first = keys.first()?;
        let last = keys.last()?;
        if t <= first.t {
            return Some(first.value);
        }
        if t >= last.t {
            return Some(last.value);
        }
        // Find the segment [k0, k1] with k0.t <= t < k1.t. Keys are sorted+unique.
        for pair in keys.windows(2) {
            let (k0, k1) = (&pair[0], &pair[1]);
            if t >= k0.t && t < k1.t {
                return Some(match k0.interp {
                    Interp::Step => k0.value,
                    Interp::Linear => {
                        let span = k1.t - k0.t;
                        if span <= 0.0 {
                            k0.value
                        } else {
                            let f = (t - k0.t) / span;
                            k0.value + f * (k1.value - k0.value)
                        }
                    }
                });
            }
        }
        Some(last.value)
    }
}

/// A simple property-track timeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sequence {
    /// Display name (also the file stem's source; see [`slug`]).
    pub name: String,
    /// Timeline length in seconds (the playhead range).
    pub duration: f64,
    /// Authoring frame rate hint — the retime/snap grid is `1 / fps_hint`.
    #[serde(default = "default_fps")]
    pub fps_hint: u32,
    #[serde(default)]
    pub tracks: Vec<PropertyTrack>,
}

fn default_fps() -> u32 {
    30
}

impl Sequence {
    /// A fresh, empty sequence (5 s at 30 fps).
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            duration: 5.0,
            fps_hint: default_fps(),
            tracks: Vec::new(),
        }
    }

    /// Find a track by `(target, path)`.
    pub fn track(&self, target: Uuid, path: &str) -> Option<&PropertyTrack> {
        self.tracks
            .iter()
            .find(|tr| tr.target == target && tr.path == path)
    }

    /// Find-or-create a track for `(target, path)`, returning a mut reference.
    pub fn track_mut(&mut self, target: Uuid, path: &str) -> &mut PropertyTrack {
        if let Some(i) = self
            .tracks
            .iter()
            .position(|tr| tr.target == target && tr.path == path)
        {
            return &mut self.tracks[i];
        }
        self.tracks.push(PropertyTrack::new(target, path));
        self.tracks.last_mut().expect("just pushed")
    }

    /// Set (or replace) a key on the `(target, path)` track, creating the track if
    /// missing.
    pub fn set_key(&mut self, target: Uuid, path: &str, t: f64, value: f64, interp: Interp) {
        self.track_mut(target, path).set_key(t, value, interp);
    }

    /// Snap a time to the nearest frame on the `1 / fps_hint` grid (the retime /
    /// key-drag rounding). `fps_hint == 0` is treated as "no grid".
    pub fn snap_time(&self, t: f64) -> f64 {
        if self.fps_hint == 0 {
            return t;
        }
        let step = 1.0 / self.fps_hint as f64;
        (t / step).round() * step
    }

    /// Canonicalize for persistence: keep tracks sorted by `(target, path)`, keys
    /// sorted+unique per track, and drop tracks with no keys.
    pub fn normalize(&mut self) {
        for tr in &mut self.tracks {
            tr.sort_keys();
        }
        self.tracks.retain(|tr| !tr.keys.is_empty());
        self.tracks
            .sort_by(|a, b| a.target.cmp(&b.target).then_with(|| a.path.cmp(&b.path)));
    }

    /// Structural validation. Returns the list of problems (empty ⇒ valid). This
    /// is the entity-agnostic check; [`validate_against`] additionally flags tracks
    /// whose `target` is not present in a document.
    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();
        if !self.duration.is_finite() || self.duration <= 0.0 {
            issues.push(format!(
                "duration must be finite and > 0 (got {})",
                self.duration
            ));
        }
        if self.fps_hint == 0 {
            issues.push("fps_hint must be >= 1".to_string());
        }
        for tr in &self.tracks {
            if split_path(&tr.path).is_none() {
                issues.push(format!("track '{}' has an unparseable path", tr.path));
            }
            let mut prev: Option<f64> = None;
            for k in &tr.keys {
                if !k.t.is_finite() || !k.value.is_finite() {
                    issues.push(format!("track '{}' has a non-finite key", tr.path));
                }
                if let Some(p) = prev {
                    if k.t <= p {
                        issues.push(format!(
                            "track '{}' keys are not strictly increasing",
                            tr.path
                        ));
                    }
                }
                prev = Some(k.t);
            }
        }
        issues
    }

    /// Like [`validate`](Self::validate) but also reports tracks whose target
    /// entity is absent from `doc` (the "known-entity optional" check).
    pub fn validate_against(&self, doc: &SceneDoc) -> Vec<String> {
        let mut issues = self.validate();
        for tr in &self.tracks {
            if doc.entity_of(tr.target).is_none() {
                issues.push(format!("track target {} is not in the scene", tr.target));
            }
        }
        issues
    }

    // ── persistence (project-local, deterministic TOML) ──────────────────────

    /// Deterministic TOML for the (normalized) sequence — byte-identical re-emit.
    pub fn to_toml(&self) -> Result<String, String> {
        let mut copy = self.clone();
        copy.normalize();
        toml::to_string_pretty(&copy).map_err(|e| e.to_string())
    }

    /// Write the sequence under `root` (creating `.infinity/sequences/` if needed).
    /// The file is `<slug(name)>.toml`.
    pub fn save(&self, root: &Path) -> Result<(), String> {
        let path = sequence_path(root, &self.name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        std::fs::write(&path, self.to_toml()?).map_err(|e| e.to_string())
    }

    /// Load the sequence named `name` under `root`, if present + parseable.
    pub fn load(root: &Path, name: &str) -> Option<Sequence> {
        let text = std::fs::read_to_string(sequence_path(root, name)).ok()?;
        toml::from_str(&text).ok()
    }
}

/// `<root>/.infinity/sequences`.
pub fn sequences_dir(root: &Path) -> PathBuf {
    root.join(".infinity").join("sequences")
}

/// `<root>/.infinity/sequences/<slug(name)>.toml`.
pub fn sequence_path(root: &Path, name: &str) -> PathBuf {
    sequences_dir(root).join(format!("{}.toml", slug(name)))
}

/// Filename-safe slug: keep ASCII alphanumerics, `-` and `_`; everything else
/// (spaces, punctuation) becomes `_`. Two names that collide on their slug share a
/// file — acceptable for the v1 seed (documented).
pub fn slug(name: &str) -> String {
    let s: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if s.is_empty() {
        "sequence".to_string()
    } else {
        s
    }
}

/// List every sequence under `root`, returning their display names (read from each
/// file's `name` field), sorted. Missing dir ⇒ empty.
pub fn list_sequences(root: &Path) -> Vec<String> {
    let dir = sequences_dir(root);
    let mut names: Vec<String> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("toml") {
                continue;
            }
            if let Ok(text) = std::fs::read_to_string(&path) {
                if let Ok(seq) = toml::from_str::<Sequence>(&text) {
                    names.push(seq.name);
                }
            }
        }
    }
    names.sort();
    names.dedup();
    names
}

/// Delete the sequence named `name` under `root`. Ok even if absent.
pub fn delete_sequence(root: &Path, name: &str) -> Result<(), String> {
    let path = sequence_path(root, name);
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.to_string()),
    }
}

// ── sampling ─────────────────────────────────────────────────────────────────

/// Sample every track at time `t`, yielding `(target, path, value)` for each track
/// that produced a value (has ≥1 key).
pub fn sample(seq: &Sequence, t: f64) -> Vec<(Uuid, String, f64)> {
    seq.tracks
        .iter()
        .filter_map(|tr| tr.sample(t).map(|v| (tr.target, tr.path.clone(), v)))
        .collect()
}

// ── path resolution + scene read/write ───────────────────────────────────────

/// Split a track path into `(component-display, field, swizzle-index?)`.
/// `"Transform.translation.x"` → `("Transform", "translation", Some(0))`;
/// `"Light.intensity"` → `("Light", "intensity", None)`.
pub fn split_path(path: &str) -> Option<(&str, &str, Option<usize>)> {
    let mut it = path.split('.');
    let comp = it.next()?;
    let field = it.next()?;
    if comp.is_empty() || field.is_empty() {
        return None;
    }
    match it.next() {
        None => Some((comp, field, None)),
        Some(sw) => {
            if it.next().is_some() {
                return None; // no deeper than one swizzle in v1
            }
            Some((comp, field, Some(swizzle_index(sw)?)))
        }
    }
}

fn swizzle_index(s: &str) -> Option<usize> {
    match s {
        "x" | "r" => Some(0),
        "y" | "g" => Some(1),
        "z" | "b" => Some(2),
        "w" | "a" => Some(3),
        _ => None,
    }
}

fn scalar_of(value: &PropValue, swizzle: Option<usize>) -> Option<f64> {
    match (value, swizzle) {
        (PropValue::Number(n), None) => Some(*n),
        (PropValue::Vec3(a), Some(i)) if i < 3 => Some(a[i]),
        _ => None,
    }
}

/// Read the CURRENT scene value of a track's scalar (the capture-key / scrub-
/// snapshot primitive). `None` if the path is unparseable, the component/field is
/// absent, or the value is not a scalar we can address.
pub fn read_scalar(doc: &SceneDoc, target: Uuid, path: &str) -> Option<f64> {
    let (comp, field, swizzle) = split_path(path)?;
    let type_path = doc.world().registry().type_path_for(comp)?;
    let comps = doc.entity_props(target);
    let c = comps.iter().find(|c| c.type_path == type_path)?;
    let f = c.fields.iter().find(|f| f.name == field)?;
    scalar_of(&f.value, swizzle)
}

/// Write one scalar into the scene as a **live preview** (non-dirtying): for a
/// swizzled path, the current vector is read, the addressed component replaced, and
/// the whole vector written back. Returns whether it applied.
pub fn write_scalar_preview(doc: &mut SceneDoc, target: Uuid, path: &str, value: f64) -> bool {
    let Some((comp, field, swizzle)) = split_path(path) else {
        return false;
    };
    let Some(type_path) = doc.world().registry().type_path_for(comp) else {
        return false;
    };
    let type_path = type_path.to_string();
    match swizzle {
        None => doc.write_prop_preview(target, &type_path, field, &PropValue::Number(value)),
        Some(i) => {
            if i >= 3 {
                return false;
            }
            // Read the current vector so untouched components are preserved.
            let comps = doc.entity_props(target);
            let Some(mut arr) = comps
                .iter()
                .find(|c| c.type_path == type_path)
                .and_then(|c| c.fields.iter().find(|f| f.name == field))
                .and_then(|f| match &f.value {
                    PropValue::Vec3(a) => Some(*a),
                    _ => None,
                })
            else {
                return false;
            };
            arr[i] = value;
            doc.write_prop_preview(target, &type_path, field, &PropValue::Vec3(arr))
        }
    }
}

// ── scrub session ────────────────────────────────────────────────────────────

/// The captured pre-scrub scene values for one sequence — the restore-on-stop
/// snapshot. Held by the Ring-2 command layer for the lifetime of a scrub gesture.
#[derive(Debug, Clone, Default)]
pub struct ScrubSnapshot {
    /// `(target, path, original_scalar)` for every distinct track that resolved.
    pub saved: Vec<(Uuid, String, f64)>,
}

/// Capture the CURRENT scene value of every distinct `(target, path)` a sequence
/// touches — the snapshot restored when scrubbing stops.
pub fn capture_snapshot(doc: &SceneDoc, seq: &Sequence) -> ScrubSnapshot {
    let mut saved = Vec::new();
    for tr in &seq.tracks {
        if saved
            .iter()
            .any(|(t, p, _)| *t == tr.target && p == &tr.path)
        {
            continue;
        }
        if let Some(v) = read_scalar(doc, tr.target, &tr.path) {
            saved.push((tr.target, tr.path.clone(), v));
        }
    }
    ScrubSnapshot { saved }
}

/// Apply a sequence's sampled values at time `t` as a non-dirtying live preview.
pub fn apply_sequence_at(doc: &mut SceneDoc, seq: &Sequence, t: f64) {
    for (target, path, value) in sample(seq, t) {
        write_scalar_preview(doc, target, &path, value);
    }
}

/// Restore the captured pre-scrub values (non-dirtying), leaving the scene exactly
/// as it was before scrubbing began.
pub fn restore_snapshot(doc: &mut SceneDoc, snapshot: &ScrubSnapshot) {
    for (target, path, value) in &snapshot.saved {
        write_scalar_preview(doc, *target, path, *value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc::SpawnKind;

    fn key(t: f64, value: f64, interp: Interp) -> Key {
        Key { t, value, interp }
    }

    #[test]
    fn sample_holds_before_first_and_after_last() {
        let mut tr = PropertyTrack::new(Uuid::nil(), "Transform.translation.x");
        tr.keys = vec![
            key(1.0, 10.0, Interp::Linear),
            key(3.0, 30.0, Interp::Linear),
        ];
        assert_eq!(tr.sample(0.0), Some(10.0)); // before first → hold
        assert_eq!(tr.sample(1.0), Some(10.0)); // at first
        assert_eq!(tr.sample(3.0), Some(30.0)); // at last
        assert_eq!(tr.sample(5.0), Some(30.0)); // after last → hold
    }

    #[test]
    fn sample_linear_interpolates_midpoint() {
        let mut tr = PropertyTrack::new(Uuid::nil(), "Transform.translation.x");
        tr.keys = vec![
            key(0.0, 0.0, Interp::Linear),
            key(2.0, 20.0, Interp::Linear),
        ];
        assert_eq!(tr.sample(1.0), Some(10.0));
        assert_eq!(tr.sample(0.5), Some(5.0));
    }

    #[test]
    fn sample_step_holds_left_value() {
        let mut tr = PropertyTrack::new(Uuid::nil(), "Transform.translation.x");
        tr.keys = vec![key(0.0, 0.0, Interp::Step), key(2.0, 20.0, Interp::Step)];
        assert_eq!(tr.sample(0.0), Some(0.0));
        assert_eq!(tr.sample(1.9), Some(0.0)); // step holds until next key
        assert_eq!(tr.sample(2.0), Some(20.0));
    }

    #[test]
    fn empty_track_samples_none() {
        let tr = PropertyTrack::new(Uuid::nil(), "Transform.translation.x");
        assert_eq!(tr.sample(1.0), None);
    }

    #[test]
    fn set_key_replaces_at_same_time_and_keeps_sorted() {
        let mut tr = PropertyTrack::new(Uuid::nil(), "Transform.scale.x");
        tr.set_key(2.0, 2.0, Interp::Linear);
        tr.set_key(0.0, 0.0, Interp::Linear);
        tr.set_key(1.0, 1.0, Interp::Linear);
        tr.set_key(1.0, 9.0, Interp::Step); // replace @ t=1
        assert_eq!(tr.keys.len(), 3);
        let times: Vec<f64> = tr.keys.iter().map(|k| k.t).collect();
        assert_eq!(times, vec![0.0, 1.0, 2.0]);
        assert_eq!(tr.keys[1].value, 9.0);
        assert_eq!(tr.keys[1].interp, Interp::Step);
    }

    #[test]
    fn remove_key_near_picks_closest() {
        let mut tr = PropertyTrack::new(Uuid::nil(), "Transform.translation.y");
        tr.set_key(0.0, 0.0, Interp::Linear);
        tr.set_key(1.0, 1.0, Interp::Linear);
        assert!(tr.remove_key_near(0.9, 0.25));
        assert_eq!(tr.keys.len(), 1);
        assert_eq!(tr.keys[0].t, 0.0);
        assert!(!tr.remove_key_near(5.0, 0.25)); // nothing within eps
    }

    #[test]
    fn snap_time_rounds_to_fps_grid() {
        let mut seq = Sequence::new("s");
        seq.fps_hint = 30;
        // 1/30 ≈ 0.0333; 0.05 rounds to 0.0667 (frame 2) not 0.0333 (frame 1.5).
        let snapped = seq.snap_time(0.05);
        assert!((snapped - 2.0 / 30.0).abs() < 1e-9, "got {snapped}");
        assert_eq!(seq.snap_time(0.0), 0.0);
    }

    #[test]
    fn normalize_sorts_tracks_and_drops_empty() {
        let a = Uuid::from_u128(1);
        let b = Uuid::from_u128(2);
        let mut seq = Sequence::new("s");
        seq.tracks = vec![
            PropertyTrack::new(b, "Transform.translation.x"),
            {
                let mut t = PropertyTrack::new(a, "Transform.translation.z");
                t.set_key(0.0, 0.0, Interp::Linear);
                t
            },
            {
                let mut t = PropertyTrack::new(a, "Transform.translation.x");
                t.set_key(0.0, 0.0, Interp::Linear);
                t
            },
        ];
        seq.normalize();
        // The empty (b) track is dropped; the two `a` tracks sort by path.
        assert_eq!(seq.tracks.len(), 2);
        assert_eq!(seq.tracks[0].target, a);
        assert_eq!(seq.tracks[0].path, "Transform.translation.x");
        assert_eq!(seq.tracks[1].path, "Transform.translation.z");
    }

    #[test]
    fn validate_flags_bad_duration_and_path() {
        let mut seq = Sequence::new("s");
        seq.duration = 0.0;
        seq.tracks = vec![{
            let mut t = PropertyTrack::new(Uuid::nil(), "not_a_path");
            t.set_key(0.0, 0.0, Interp::Linear);
            t
        }];
        let issues = seq.validate();
        assert!(issues.iter().any(|i| i.contains("duration")));
        assert!(issues.iter().any(|i| i.contains("unparseable")));
    }

    #[test]
    fn split_path_variants() {
        assert_eq!(
            split_path("Transform.translation.x"),
            Some(("Transform", "translation", Some(0)))
        );
        assert_eq!(
            split_path("Light.intensity"),
            Some(("Light", "intensity", None))
        );
        assert_eq!(split_path("Transform"), None);
        assert_eq!(split_path("Transform.translation.q"), None); // bad swizzle
        assert_eq!(split_path("A.b.c.d"), None); // too deep
    }

    #[test]
    fn toml_round_trip_is_deterministic() {
        let mut seq = Sequence::new("Intro");
        seq.duration = 4.0;
        seq.set_key(
            Uuid::from_u128(7),
            "Transform.translation.x",
            0.0,
            0.0,
            Interp::Linear,
        );
        seq.set_key(
            Uuid::from_u128(7),
            "Transform.translation.x",
            2.0,
            5.0,
            Interp::Step,
        );
        let a = seq.to_toml().unwrap();
        let b = seq.to_toml().unwrap();
        assert_eq!(a, b, "re-emit is byte-identical");
        let back: Sequence = toml::from_str(&a).unwrap();
        let mut norm = seq.clone();
        norm.normalize();
        assert_eq!(back, norm);
    }

    #[test]
    fn save_list_load_delete_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let mut seq = Sequence::new("My Shot 1");
        seq.set_key(
            Uuid::from_u128(3),
            "Transform.translation.y",
            1.0,
            2.0,
            Interp::Linear,
        );
        seq.save(root).unwrap();

        assert_eq!(list_sequences(root), vec!["My Shot 1".to_string()]);
        let back = Sequence::load(root, "My Shot 1").expect("loads");
        assert_eq!(back.name, "My Shot 1");
        assert_eq!(back.tracks.len(), 1);

        delete_sequence(root, "My Shot 1").unwrap();
        assert!(list_sequences(root).is_empty());
        assert!(Sequence::load(root, "My Shot 1").is_none());
    }

    // ── scene-bound tests (scrub / capture / restore) ────────────────────────

    #[test]
    fn read_and_write_scalar_addresses_vector_component() {
        let mut doc = SceneDoc::new();
        let cube = doc.create(SpawnKind::Cube, "Cube", None);
        // Default translation is (0,0,0).
        assert_eq!(
            read_scalar(&doc, cube, "Transform.translation.x"),
            Some(0.0)
        );

        assert!(write_scalar_preview(
            &mut doc,
            cube,
            "Transform.translation.y",
            3.5
        ));
        assert_eq!(
            read_scalar(&doc, cube, "Transform.translation.y"),
            Some(3.5)
        );
        // Only y changed; x and z stay 0.
        assert_eq!(
            read_scalar(&doc, cube, "Transform.translation.x"),
            Some(0.0)
        );
        assert_eq!(
            read_scalar(&doc, cube, "Transform.translation.z"),
            Some(0.0)
        );
    }

    #[test]
    fn scrub_preview_never_dirties_the_document() {
        let mut doc = SceneDoc::new();
        let cube = doc.create(SpawnKind::Cube, "Cube", None);
        doc.mark_saved(); // pretend the scene was just saved (clean)
        assert!(!doc.is_dirty());
        let v0 = doc.version();

        let mut seq = Sequence::new("s");
        seq.set_key(cube, "Transform.translation.x", 0.0, 0.0, Interp::Linear);
        seq.set_key(cube, "Transform.translation.x", 2.0, 10.0, Interp::Linear);

        let snap = capture_snapshot(&doc, &seq);
        apply_sequence_at(&mut doc, &seq, 1.0);
        assert_eq!(
            read_scalar(&doc, cube, "Transform.translation.x"),
            Some(5.0)
        );
        assert!(!doc.is_dirty(), "scrub preview must not dirty the scene");
        assert!(
            doc.version() > v0,
            "but the version bumps so the viewport re-syncs"
        );

        // Stop restores the captured original exactly, still clean.
        restore_snapshot(&mut doc, &snap);
        assert_eq!(
            read_scalar(&doc, cube, "Transform.translation.x"),
            Some(0.0)
        );
        assert!(!doc.is_dirty());
    }

    #[test]
    fn capture_snapshot_dedupes_and_reads_current() {
        let mut doc = SceneDoc::new();
        let cube = doc.create(SpawnKind::Cube, "Cube", None);
        write_scalar_preview(&mut doc, cube, "Transform.translation.x", 7.0);

        let mut seq = Sequence::new("s");
        // Two tracks on the same (target,path) — snapshot must dedupe.
        seq.set_key(cube, "Transform.translation.x", 0.0, 1.0, Interp::Linear);
        let snap = capture_snapshot(&doc, &seq);
        assert_eq!(snap.saved.len(), 1);
        assert_eq!(snap.saved[0].2, 7.0);
    }
}
