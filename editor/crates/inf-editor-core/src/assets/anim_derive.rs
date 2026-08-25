//! **Import-time animation derivation, at the asset door** (P29.5, pillar S2).
//!
//! [`inf_anim::derive`] is the rule; this is the two places it is applied:
//!
//! * at **import**, inside the glTF orchestrator's clip stage, so a clip is
//!   derived before it is ever written and there is no window in which a
//!   `.inf_anim` in this project has not been measured;
//! * on **demand**, from the panel's re-derive button, over a clip that is
//!   already on disk — which is why the rule is idempotent.
//!
//! # It rewrites the clip rather than deriving a sibling
//!
//! `assets::vmesh` derives a *second* asset (`.inf_vmesh`) from a `.inf_mesh`,
//! and the shape is deliberately not copied here. A meshlet DAG is a different
//! representation of the same geometry and a consumer chooses between them; a
//! root-motion track is not a representation of a clip, it is **part** of one,
//! and `.inf_anim` v2 has had the fields since P29.2 for exactly this. A sibling
//! asset would mean every reader — both hosts' pose steps, the cook, the packer,
//! the blend-space preview — had to learn to join two ids, and the one that
//! forgot would silently read an underived clip.
//!
//! # Failures are advisories, never a failed import
//!
//! A clip that cannot be derived is still a clip. A rig with no root joint, a
//! zero-length pose, a clip bound to no skeleton at all — each costs that clip
//! its derived channels and nothing else, and says so in the import's advisory
//! list. `vmesh`'s stage does the same thing for the same reason.

use inf_anim::{AnimClip, AnimClipAsset, DeriveOptions, DeriveReport, SkeletonAsset};
use inf_asset::{AssetId, AssetKind};

use super::AssetProject;

/// How far a clip's derived **net rise** has to be, in metres, before the import
/// says out loud that it might be a traversal clip.
///
/// A quarter of a metre: taller than any gait's residual drift and shorter than
/// the shortest thing anybody mantles. The advisory exists because the vertical
/// policy is the one derivation decision that cannot be read off the clip —
/// [`inf_anim::VerticalPolicy`] explains why — and a silent wrong answer here
/// puts a mantle's rise in the pose instead of on the track, which looks like
/// the character sinking into the ledge. The P16 rule: a hazard nothing can
/// detect automatically gets an **advisory**, not a guess.
pub const TRAVERSAL_HINT_RISE_M: f32 = 0.25;

/// What happened to one clip.
#[derive(Clone, Debug, PartialEq)]
pub enum ClipDerivation {
    /// Derived, with what it found.
    Derived(Box<DeriveReport>),
    /// Not derived, and why — a **value**, carried into the import's advisories.
    Skipped(String),
}

impl ClipDerivation {
    /// The report, when there is one.
    pub fn report(&self) -> Option<&DeriveReport> {
        match self {
            ClipDerivation::Derived(r) => Some(r),
            ClipDerivation::Skipped(_) => None,
        }
    }

    /// The reason it was skipped, when it was.
    pub fn skipped(&self) -> Option<&str> {
        match self {
            ClipDerivation::Skipped(why) => Some(why),
            ClipDerivation::Derived(_) => None,
        }
    }
}

/// **Derive one clip in place**, answering what happened.
///
/// The one door both callers below go through, so an imported clip and a
/// re-derived one are measured by the same code with the same defaults.
pub fn derive_in_place(
    name: &str,
    clip: &mut AnimClip,
    skeleton: Option<&SkeletonAsset>,
    opts: &DeriveOptions,
) -> ClipDerivation {
    let Some(rig) = skeleton else {
        return ClipDerivation::Skipped(format!(
            "clip `{name}` is bound to no skeleton, so its root joint is unknown and nothing was derived"
        ));
    };
    match inf_anim::derive_clip(clip, rig, opts) {
        Ok((out, report)) => {
            *clip = out;
            ClipDerivation::Derived(Box::new(report))
        }
        Err(e) => ClipDerivation::Skipped(format!("clip `{name}` was not derived: {e}")),
    }
}

/// The advisory lines one derivation is worth, if any.
///
/// Two kinds, and both are things an author can act on: a skip says what was not
/// measured, and a large net rise says the vertical policy may be the wrong one
/// for this clip.
pub fn advisories(name: &str, d: &ClipDerivation) -> Vec<String> {
    match d {
        ClipDerivation::Skipped(why) => vec![why.clone()],
        ClipDerivation::Derived(r) => {
            let rise = r.translation[1];
            if rise.abs() > TRAVERSAL_HINT_RISE_M {
                vec![format!(
                    "clip `{name}` rises {rise:.2} m over its length; if it is a traversal clip (a mantle, a vault, a climb) re-derive it with the traversal preset so the whole rise reaches the root-motion track instead of only its net drift"
                )]
            } else {
                Vec::new()
            }
        }
    }
}

/// **Re-derive a clip that is already on disk**, and write it back.
///
/// The panel's button. It rewrites the same asset in place — same GUID, same
/// dependencies, same sidecar `import` table (`None` keeps it, which matters
/// here: `skeleton_binding`'s rig hash is exactly the provenance a re-derive
/// must not silence).
///
/// Idempotent, because [`inf_anim::derive_clip`] is: pressing the button twice
/// produces the same bytes as pressing it once, which is the property that makes
/// it safe to put on a button at all.
pub fn rederive_asset(
    project: &mut AssetProject,
    id: AssetId,
    opts: &DeriveOptions,
) -> Result<ClipDerivation, String> {
    let entry = project
        .db()
        .get(id)
        .ok_or_else(|| format!("no asset {id}"))?;
    if entry.kind() != AssetKind::AnimClip {
        return Err(format!("{} is not an animation clip", entry.name));
    }
    let name = entry.name.clone();
    let mut asset = project
        .load_payload::<AnimClipAsset>(id)
        .map_err(|e| e.to_string())?;
    let rig = skeleton_for(project, id, &asset);
    let outcome = derive_in_place(&name, &mut asset.clip, rig.as_ref(), opts);
    if outcome.report().is_some() {
        let deps: Vec<AssetId> = project
            .db()
            .get(id)
            .map(|e| e.sidecar.dependencies.clone())
            .unwrap_or_default();
        project
            .rewrite_payload(id, &asset, deps)
            .map_err(|e| e.to_string())?;
    }
    Ok(outcome)
}

/// The rig a clip is bound to, resolved the way every other reader resolves it:
/// the payload's own `skeleton` GUID first, then the dependency edge.
///
/// Both, because they are written by different producers — the glTF importer
/// records the GUID, and a hand-built asset may carry only the edge — and a
/// resolver that knew about one of them would derive half the clips in a project
/// and skip the rest with a message about missing skeletons.
pub fn skeleton_for(
    project: &AssetProject,
    id: AssetId,
    asset: &AnimClipAsset,
) -> Option<SkeletonAsset> {
    let direct = asset
        .skeleton
        .map(|b| AssetId::from(uuid::Uuid::from_bytes(b)));
    let candidates = direct.into_iter().chain(
        project
            .db()
            .get(id)
            .map(|e| e.sidecar.dependencies.clone())
            .unwrap_or_default(),
    );
    for cand in candidates {
        if project.db().get(cand).map(|e| e.kind()) != Some(AssetKind::Skeleton) {
            continue;
        }
        if let Ok(rig) = project.load_payload::<SkeletonAsset>(cand) {
            return Some(rig);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_anim::{BodyParams, BodyPlan};

    fn rig() -> SkeletonAsset {
        inf_anim::build_template(BodyPlan::Biped, &BodyParams::default()).unwrap()
    }

    /// The door derives, and it derives the same thing twice.
    #[test]
    fn the_import_door_derives_and_is_idempotent() {
        let sk = rig();
        let set = inf_anim::build_locomotion(BodyPlan::Biped, &sk, &Default::default()).unwrap();
        let mut clip = set.walk.clone();
        let first = derive_in_place("Walk", &mut clip, Some(&sk), &DeriveOptions::default());
        assert!(first.report().is_some(), "{first:?}");
        let once = clip.clone();
        let second = derive_in_place("Walk", &mut clip, Some(&sk), &DeriveOptions::default());
        assert_eq!(first, second);
        assert_eq!(once, clip, "a re-derive moved the clip");
    }

    /// A clip with no rig is skipped **by name**, not silently.
    #[test]
    fn a_clip_with_no_rig_is_skipped_with_a_reason() {
        let mut clip = inf_anim::AnimClip::new("Orphan", Vec::new());
        let d = derive_in_place("Orphan", &mut clip, None, &DeriveOptions::default());
        let why = d.skipped().expect("a reason");
        assert!(why.contains("Orphan"), "{why}");
        assert!(why.contains("skeleton"), "{why}");
        assert_eq!(advisories("Orphan", &d), vec![why.to_string()]);
    }

    /// A clip that rises says so, and one that does not stays quiet — the
    /// advisory has to be able to *not* fire or it is noise.
    #[test]
    fn a_rising_clip_earns_a_traversal_advisory_and_a_gait_does_not() {
        let sk = rig();
        let set = inf_anim::build_locomotion(BodyPlan::Biped, &sk, &Default::default()).unwrap();
        let mut walk = set.walk.clone();
        let d = derive_in_place("Walk", &mut walk, Some(&sk), &DeriveOptions::default());
        assert!(
            advisories("Walk", &d).is_empty(),
            "a walk cycle must not be advised: {:?}",
            advisories("Walk", &d)
        );

        // A clip whose root climbs a metre over its length.
        let mut climb = inf_anim::AnimClip::new("Climb", Vec::new());
        let root = inf_anim::root_joint_index(&sk.skeleton).unwrap() as u16;
        inf_anim::derive::set_root_translation(
            &mut climb,
            root,
            vec![0.0, 1.0],
            vec![[0.0, 0.0, 0.0], [0.0, 1.0, 0.5]],
        );
        climb.duration = 1.0;
        let d = derive_in_place("Climb", &mut climb, Some(&sk), &DeriveOptions::default());
        let lines = advisories("Climb", &d);
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert!(lines[0].contains("traversal"), "{}", lines[0]);
        assert!(lines[0].contains("1.00 m"), "{}", lines[0]);
    }
}
