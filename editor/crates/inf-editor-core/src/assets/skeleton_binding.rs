//! **The identity tie between an animation asset and the rig it was authored
//! against** (round-2 finding R2.C).
//!
//! # The class
//!
//! `AnimClipAsset::skeleton` and `StateMachineAsset::skeleton` record a
//! **GUID** — "this clip animates that skeleton" — and nothing else. But the
//! coupling between a clip and a rig is **positional**: `Vec3Track::joint` /
//! `QuatTrack::joint` are `u16` indices into `Pose::locals`, which is
//! index-aligned to `Skeleton::joints`. So the pairing is only meaningful while
//! the skeleton's joint *list* is the one the clip was baked against.
//!
//! Re-import the rig with a joint inserted, re-order a hierarchy in the DCC,
//! or generate a fresh rig from a template into the same GUID, and every track
//! index past the edit point now names a different bone — **in range**, so no
//! guard fires, no `Option` is `None`, and no length check disagrees. The
//! character animates: an arm rotates where a spine should, or a finger drives
//! a hip. Every parity gate is blind to it, because both sides read the same
//! two assets and both are wrong together.
//!
//! This is L5.F11's staleness class, and round 2 found instances **4 and 5**:
//! the first three (`DerivedMaterial` with no source digest, and the cloth
//! garment / hair scalp GUID-only bindings that are *also* index-aligned to
//! vertices) are recorded in [`DEFERRED_INSTANCES`] and are not closed here —
//! they challenge recorded bound **B57**'s scope, and a recorded bound is
//! changed deliberately with its own bump.
//!
//! # The tie, and why it is a sidecar entry rather than a schema field
//!
//! The `.inf_vmesh` derivation has solved this once already: it records the
//! **source's content hash** in the derived asset's sidecar `import` table and
//! compares it before trusting the derivation (`assets::vmesh`'s
//! `recorded_source_hash`). A sidecar's `import` table is free-form TOML, so
//! this costs **no schema version** on either payload — which matters, because
//! `AnimClipAsset` and `StateMachineAsset` are both bincode and every schema
//! move on them is a positional break.
//!
//! # An advisory, not a refusal
//!
//! A mismatch means "these two were not authored together", and the right
//! response is to tell the author, not to stop loading their level: the clip is
//! *probably* still mostly right, the fix is a re-import they have to perform
//! anyway, and refusing would take a project offline over a bone the author may
//! have deliberately renamed. The cook-advisory doctrine (P16: *a silent hazard
//! gets an advisory*) is the shape this takes.

use inf_asset::{AssetId, AssetKind, AssetSidecar, ContentHash};

use super::AssetProject;

/// The sidecar `import` key holding the skeleton's content hash, as lowercase
/// hex. Named once; `assets::vmesh` uses the same shape for its own source.
pub const SKELETON_HASH_KEY: &str = "skeleton_content_hash";

/// The instances of this class that are **recorded and not closed**, with what
/// each would need.
///
/// Kept as data rather than prose so it can be asserted non-empty and read by a
/// reviewer: a deferral nobody can find again is the same as a decision nobody
/// made. Each challenges recorded bound **B57**'s scope.
pub const DEFERRED_INSTANCES: &[(&str, &str)] = &[
    (
        "inf_asset::DerivedMaterial",
        "has no source digest at all, so a re-imported texture set is invisible to \
         every consumer; needs the same sidecar entry as this module, at the \
         material derivation's own door",
    ),
    (
        "inf_anim::ClothAsset garment binding",
        "GUID-only while its particle indices are index-aligned to the mesh's \
         VERTICES — a re-imported mesh with one vertex more re-binds every \
         constraint; needs a mesh content hash, not a skeleton one",
    ),
    (
        "inf_anim::HairAsset scalp binding",
        "the same shape as the garment: strand roots are index-aligned to scalp \
         vertices",
    ),
];

/// Record `skeleton`'s content hash in an animation asset's sidecar `import`
/// table.
///
/// Returns the table to hand to `AssetProject::write_asset`. `None` skeleton ⇒
/// `None` table, which is exactly the "skeleton-agnostic clip" case and is not a
/// hazard: a clip bound to nothing cannot be mis-bound.
pub fn import_table(project: &AssetProject, skeleton: Option<AssetId>) -> Option<toml::Table> {
    let hash = skeleton_hash(project, skeleton?)?;
    let mut table = toml::Table::new();
    table.insert(
        SKELETON_HASH_KEY.to_string(),
        toml::Value::String(format!("{:032x}", hash.0)),
    );
    Some(table)
}

/// The content hash a sidecar records for the skeleton its asset was authored
/// against, if it records one.
///
/// A malformed value reads as "nothing recorded", i.e. no advisory — the safe
/// way to be wrong about provenance, and the same rule
/// `vmesh::recorded_source_hash` states for its own key.
pub fn recorded_skeleton_hash(sidecar: &AssetSidecar) -> Option<ContentHash> {
    let hex = sidecar.import.as_ref()?.get(SKELETON_HASH_KEY)?.as_str()?;
    u128::from_str_radix(hex, 16).ok().map(ContentHash)
}

/// The current content hash of skeleton `id`, if the project has it.
fn skeleton_hash(project: &AssetProject, id: AssetId) -> Option<ContentHash> {
    let entry = project.db().get(id)?;
    (entry.kind() == AssetKind::Skeleton).then_some(entry.sidecar.content_hash)
}

/// The skeleton an animation asset depends on, if the project can see one.
///
/// One walk, two readers (round 3): [`advisories`] asks it of the asset it is
/// checking, and the State Machine editor asks it of the **clips** a machine
/// plays — a `.inf_sm` names no skeleton of its own, so the only thing that can
/// say which rig it was authored against is the clips inside it.
///
/// An animation asset has at most one skeleton by construction (every writer in
/// the tree passes exactly one), so "the first Skeleton among the dependencies"
/// is the answer rather than a choice among several.
pub fn skeleton_of(project: &AssetProject, id: AssetId) -> Option<AssetId> {
    let db = project.db();
    db.get(id)?
        .sidecar
        .dependencies
        .iter()
        .copied()
        .find(|d| db.get(*d).is_some_and(|e| e.kind() == AssetKind::Skeleton))
}

/// **The check.** One advisory line per animation asset whose recorded skeleton
/// hash disagrees with the skeleton it points at, in GUID order.
///
/// Empty on a healthy project, and empty on an old one — an asset imported
/// before this key existed records nothing and is silent, which is the only
/// honest answer: nothing on disk says what it was baked against.
pub fn advisories(project: &AssetProject) -> Vec<String> {
    let db = project.db();
    let mut out = Vec::new();
    let mut rows: Vec<_> = db
        .iter()
        .filter(|e| matches!(e.kind(), AssetKind::AnimClip | AssetKind::StateMachine))
        .collect();
    rows.sort_by_key(|e| e.sidecar.guid);
    for entry in rows {
        let Some(recorded) = recorded_skeleton_hash(&entry.sidecar) else {
            continue; // imported before the key existed — nothing to compare
        };
        // The skeleton it depends on. An animation asset has at most one.
        let Some(skel) = skeleton_of(project, entry.id()) else {
            continue;
        };
        let Some(current) = skeleton_hash(project, skel) else {
            continue;
        };
        if current != recorded {
            out.push(format!(
                "`{}` was authored against a different version of its skeleton \
                 (recorded {:032x}, current {:032x}). Its track indices are \
                 POSITIONS in the rig's joint list, so any joint inserted, \
                 removed or re-ordered since re-binds every track past the edit — \
                 in range, so nothing refuses it, and the character animates the \
                 wrong bones. Re-import the animation against the current rig.",
                entry.name, recorded.0, current.0
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_anim::{AnimClip, AnimClipAsset, Joint, JointTransform, Skeleton, SkeletonAsset};

    fn joint(name: &str, parent: Option<u16>) -> Joint {
        Joint {
            name: name.into(),
            parent,
            inverse_bind: glam::Mat4::IDENTITY.to_cols_array(),
            local_bind: JointTransform::IDENTITY,
        }
    }

    /// **Round-2 finding R2.C.** A clip's coupling to its rig is POSITIONAL —
    /// `track.joint` is a `u16` index into `Pose::locals` — while the binding it
    /// records is a GUID. A rig re-imported with a joint inserted re-binds every
    /// track past the edit point, in range, with nothing to refuse it.
    #[test]
    fn a_reimported_skeleton_makes_its_clips_say_so() {
        let dir = tempfile::tempdir().unwrap();
        let mut project = AssetProject::open(dir.path()).unwrap();
        let content = project.content_dir("Anim").unwrap();

        let two = Skeleton::new(vec![joint("root", None), joint("spine", Some(0))]).unwrap();
        let skel_id = project
            .write_asset(
                &content,
                "Rig",
                &SkeletonAsset::new(two),
                None,
                vec![],
                None,
            )
            .unwrap();

        let table = import_table(&project, Some(skel_id));
        assert!(
            table.is_some(),
            "no hash was recorded, so every assertion below is vacuous"
        );
        let clip_id = project
            .write_asset(
                &content,
                "Walk",
                &AnimClipAsset::new(
                    AnimClip::new("walk", Vec::new()),
                    Some(*skel_id.uuid().as_bytes()),
                ),
                None,
                vec![skel_id],
                table,
            )
            .unwrap();

        // Authored together: silent.
        assert!(
            advisories(&project).is_empty(),
            "a freshly imported pair raised an advisory"
        );

        // The rig is re-imported with a joint INSERTED in the middle — the exact
        // edit that renumbers everything after it.
        let three = Skeleton::new(vec![
            joint("root", None),
            joint("pelvis", Some(0)),
            joint("spine", Some(1)),
        ])
        .unwrap();
        project
            .rewrite_payload(skel_id, &SkeletonAsset::new(three), vec![])
            .expect("the rig re-imports in place");

        let found = advisories(&project);
        assert_eq!(
            found.len(),
            1,
            "the re-imported rig said nothing: {found:?}"
        );
        assert!(
            found[0].contains("Walk") && found[0].contains("POSITIONS"),
            "the advisory must name the asset and the mechanism: {}",
            found[0]
        );
        let _ = clip_id;

        // An asset with no recorded hash — everything imported before this key
        // existed — stays silent rather than raising a false alarm.
        project
            .write_asset(
                &content,
                "Legacy",
                &AnimClipAsset::new(
                    AnimClip::new("walk", Vec::new()),
                    Some(*skel_id.uuid().as_bytes()),
                ),
                None,
                vec![skel_id],
                None,
            )
            .unwrap();
        assert_eq!(
            advisories(&project).len(),
            1,
            "an asset with nothing recorded raised an advisory about nothing"
        );
    }

    /// The deferred instances are recorded rather than implied.
    #[test]
    fn the_deferred_instances_carry_their_reasons() {
        assert_eq!(DEFERRED_INSTANCES.len(), 3);
        for (what, why) in DEFERRED_INSTANCES {
            assert!(!what.is_empty() && why.len() > 60, "{what}: {why}");
        }
    }
}
