//! Skeleton templates (P24.1): the editor door onto `inf_anim::template`.
//!
//! One command — [`skel_create_template`] — writes a generated `.inf_skel` into
//! the open project through the same `AssetProject::write_asset` every other
//! create path uses, so a templated rig is an ordinary asset the moment it
//! exists: it appears in the Content Drawer, it thumbnails, it can be dragged
//! onto a `SkeletalMesh`, and deleting it warns about referrers.
//!
//! # The v1 surface, and what it is not
//!
//! The **create door**, not a skeleton *editor*. The proportions a caller can set
//! are the two that change what kind of creature this is — the plan and its size
//! (plus the leg count for an arbitrary N-pedal) — and everything else comes from
//! [`BodyParams::default`]. Editing every proportion belongs to the P24.5 "New
//! Character from Template" wizard, and a joint-level skeleton panel to P24.3;
//! shipping a half-wizard now would be a surface both would have to replace.
//!
//! What matters is that the *asset* is complete: canonical humanoid joint names,
//! standard sockets, and the IK joint limits P24.2 consumes are all in the
//! payload from the first generation, so nothing downstream has to special-case a
//! template-made rig.

use inf_anim::template::{build_template, BodyParams, BodyPlan};
use tauri::{AppHandle, State};

use super::assets::{emit_changed, AssetState};

/// The content sub-folder generated skeletons land in.
const SKELETON_FOLDER: &str = "Skeletons";

/// Parse the wire name of a body plan.
///
/// A **string**, not a ts-rs enum, deliberately: the plan set is the one thing in
/// this API that P24.3 (modular rigging — "append limbs, tails and extras") is
/// expected to grow, and a stringly-typed name that fails loudly with the list of
/// what it does know is kinder to a frontend built against an older backend than
/// a generated union that silently loses a variant.
fn parse_plan(plan: &str, legs: Option<u16>) -> Result<BodyPlan, String> {
    match plan {
        "biped" => Ok(BodyPlan::Biped),
        "quadruped" => Ok(BodyPlan::Quadruped),
        "hexapod" => Ok(BodyPlan::Hexapod),
        "npedal" => Ok(BodyPlan::Npedal {
            legs: legs.ok_or("an `npedal` template needs a leg count")?,
        }),
        other => Err(format!(
            "unknown body plan `{other}` (expected biped, quadruped, hexapod or npedal)"
        )),
    }
}

/// The default name for a generated rig.
fn default_name(plan: BodyPlan) -> String {
    match plan {
        BodyPlan::Biped => "Biped Rig".into(),
        BodyPlan::Quadruped => "Quadruped Rig".into(),
        BodyPlan::Hexapod => "Hexapod Rig".into(),
        BodyPlan::Npedal { legs } => format!("{legs}-Legged Rig"),
    }
}

/// Generate a template skeleton and write it as a `.inf_skel` under
/// `Content/Skeletons`. Returns the new asset GUID.
///
/// `height_m` overrides the default standing height; everything else uses
/// [`BodyParams::default`] (see the module docs for why the v1 surface stops
/// there). A degenerate request is refused as a **message**, not a panic — the
/// generator's own [`TemplateError`](inf_anim::template::TemplateError) names the
/// offending parameter, and that text is what reaches the user.
#[tauri::command]
pub async fn skel_create_template(
    app: AppHandle,
    plan: String,
    legs: Option<u16>,
    height_m: Option<f64>,
    name: Option<String>,
    state: State<'_, AssetState>,
) -> Result<String, String> {
    let plan = parse_plan(&plan, legs)?;
    let params = BodyParams {
        height_m: height_m.unwrap_or(BodyParams::default().height_m),
        ..BodyParams::default()
    };
    // Generated (and validated) BEFORE the project is touched, so a refusal never
    // leaves a half-written asset behind.
    let asset = build_template(plan, &params).map_err(|e| e.to_string())?;
    let name = name.unwrap_or_else(|| default_name(plan));

    let id = state.with_project(|p| {
        let dir = p.content_dir(SKELETON_FOLDER).map_err(|e| e.to_string())?;
        p.write_asset(&dir, &name, &asset, None, vec![], None)
            .map_err(|e| e.to_string())
    })?;
    emit_changed(&app, &state);
    Ok(id.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every plan the Content Drawer offers parses, and an unknown one is a
    /// message naming the alternatives rather than a fallback to "biped".
    #[test]
    fn plan_names_parse_and_unknown_ones_are_refused() {
        assert_eq!(parse_plan("biped", None).unwrap(), BodyPlan::Biped);
        assert_eq!(parse_plan("quadruped", None).unwrap(), BodyPlan::Quadruped);
        assert_eq!(parse_plan("hexapod", None).unwrap(), BodyPlan::Hexapod);
        assert_eq!(
            parse_plan("npedal", Some(8)).unwrap(),
            BodyPlan::Npedal { legs: 8 }
        );
        let err = parse_plan("octopus", None).unwrap_err();
        assert!(err.contains("biped"), "{err}");
        assert!(parse_plan("npedal", None).is_err(), "npedal needs a count");
    }

    /// The refusal a degenerate request produces is the generator's own message,
    /// so it names the parameter — the whole point of `TemplateError` being a
    /// value.
    #[test]
    fn a_degenerate_request_is_refused_by_name() {
        let err = build_template(
            BodyPlan::Biped,
            &BodyParams {
                height_m: 0.0,
                ..BodyParams::default()
            },
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("height_m"), "{err}");
    }
}
