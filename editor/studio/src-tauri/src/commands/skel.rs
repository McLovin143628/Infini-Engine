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
//!
//! # P24.3: the create door grew an EDITOR
//!
//! The paragraph above says "a joint-level skeleton panel [belongs] to P24.3",
//! and the rest of this module is it: `skel_open` .. `skel_save`, over
//! `inf_editor_core::skel::SkelSession`. `skel_create_template` is unchanged and
//! still the *create* door — it mints a new asset; the editor opens one that
//! exists.

use std::collections::BTreeMap;
use std::sync::Mutex;

use inf_anim::template::{build_template, BodyParams, BodyPlan};
use inf_anim::{JointLimit, JointTransform, SkeletonAsset};
use inf_asset::{AssetId, AssetKind};
use inf_editor_core::ipc::{SkelApplyDto, SkelDocDto, SkelJointDto, SkelSocketDto};
use inf_editor_core::skel::{RenameVerdict, SkelError, SkelSession};
use tauri::{AppHandle, Emitter, State};

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

// ── the Skeleton Editor (P24.3) ─────────────────────────────────────────────
//
// The panel's Ring-2 door. `inf_editor_core::skel::SkelSession` holds every rule
// (see its module docs); everything here is state, a DTO projection, and one
// `Result<_, String>` per author action, per the typed-IPC law.
//
// **The asset IS the document.** `DccState` keys on `"dcc:<assetId>"` because a
// mesh document owns a journal the asset does not; a skeleton session owns only
// the asset and its snapshot history, so the key is the asset GUID itself and
// re-opening a rig returns the live session rather than silently forking it.
//
// **Save lives partly in Ring 1 and this is the honest split.** `save_mesh_session`
// is in Ring 1 because a `#[tauri::command]` cannot be driven from a test and the
// mesh save has real logic (a derived `.inf_vmesh` to rebuild or remove). A
// skeleton has **no derived artifact** — nothing is computed from it at cook time
// — so its save is one `rewrite_payload` call with no ordering to get wrong, and
// the only rule worth stating (`mark_saved` runs after the write, never before)
// is enforced by `SkelSession` itself, where a test can reach it.

/// Every open skeleton document, keyed by asset GUID.
#[derive(Default)]
pub struct SkelState {
    inner: Mutex<BTreeMap<String, SkelSession>>,
}

impl SkelState {
    fn with<R>(
        &self,
        f: impl FnOnce(&mut BTreeMap<String, SkelSession>) -> Result<R, String>,
    ) -> Result<R, String> {
        let mut store = self.inner.lock().map_err(|e| e.to_string())?;
        f(&mut store)
    }
}

fn doc_of<'a>(
    store: &'a mut BTreeMap<String, SkelSession>,
    id: &str,
) -> Result<&'a mut SkelSession, String> {
    store
        .get_mut(id)
        .ok_or_else(|| format!("no open skeleton document `{id}`"))
}

fn joint_dto(asset: &SkeletonAsset, index: usize) -> SkelJointDto {
    let j = &asset.skeleton.joints()[index];
    let limit = asset.limit(index as u16);
    let twin = inf_anim::mirrored_joint_name(&j.name);
    let mirror = twin.as_deref().and_then(|t| asset.skeleton.index_of(t));
    SkelJointDto {
        index: index as u16,
        name: j.name.clone(),
        parent: j.parent,
        translation: j.local_bind.translation,
        rotation: j.local_bind.rotation,
        scale: j.local_bind.scale,
        canonical: inf_anim::humanoid_joint_names().contains(&j.name.as_str()),
        mirror,
        sided_without_twin: twin.is_some() && mirror.is_none(),
        limit_min_deg: limit.map(|l| l.min_deg),
        limit_max_deg: limit.map(|l| l.max_deg),
    }
}

fn doc_dto(id: &str, name: &str, s: &SkelSession) -> SkelDocDto {
    let a = s.asset();
    SkelDocDto {
        id: id.to_string(),
        name: name.to_string(),
        joints: (0..a.skeleton.len()).map(|i| joint_dto(a, i)).collect(),
        sockets: a
            .sockets
            .iter()
            .map(|k| SkelSocketDto {
                name: k.name.clone(),
                joint: k.joint,
                translation: k.local_offset.translation,
                rotation: k.local_offset.rotation,
                scale: k.local_offset.scale,
            })
            .collect(),
        dirty: s.is_dirty(),
        can_undo: s.can_undo(),
        can_redo: s.can_redo(),
        unmatched_sided: s.unmatched_sided(),
    }
}

/// The verdict text for a rename, or `None` when it cost nothing.
fn rename_warning(v: &RenameVerdict) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();
    if v.left_humanoid_set {
        parts.push(format!(
            "`{}` was one of the canonical humanoid joints — retargeting onto \
             another rig will no longer pair it",
            v.previous
        ));
    }
    if v.broke_mirror_pair {
        parts.push(format!(
            "`{}` had a left/right twin and the new name does not — mirroring \
             across this rig will now weight the copy to the wrong side",
            v.previous
        ));
    }
    (!parts.is_empty()).then(|| parts.join("; "))
}

/// An asset's display name, for the document header.
fn asset_name(assets: &AssetState, id: AssetId) -> Result<String, String> {
    assets.with_project(|p| {
        p.db()
            .get(id)
            .map(|e| e.name.clone())
            .ok_or_else(|| format!("no asset {id}"))
    })
}

/// Open a `.inf_skel` for editing. Idempotent: re-opening an already-open rig
/// returns the live document rather than discarding its history.
#[tauri::command]
pub async fn skel_open(
    asset_id: String,
    state: State<'_, SkelState>,
    assets: State<'_, AssetState>,
) -> Result<SkelDocDto, String> {
    let id: AssetId = asset_id.parse().map_err(|e| format!("bad asset id: {e}"))?;
    let name = asset_name(&assets, id)?;
    let open = state.with(|store| Ok(store.contains_key(&asset_id)))?;
    if !open {
        let asset = assets.with_project(|p| {
            let entry = p.db().get(id).ok_or_else(|| format!("no asset {id}"))?;
            if entry.kind() != AssetKind::Skeleton {
                return Err(format!("{} is not a skeleton asset", entry.name));
            }
            p.load_payload::<SkeletonAsset>(id)
                .map_err(|e| e.to_string())
        })?;
        state.with(|store| {
            store.insert(asset_id.clone(), SkelSession::new(asset));
            Ok(())
        })?;
    }
    state.with(|store| Ok(doc_dto(&asset_id, &name, doc_of(store, &asset_id)?)))
}

/// Free a document and its history. Idempotent, and takes the **asset** id — the
/// same key `skel_open` takes — so a panel that unmounts before its open resolves
/// can still send it (the `dcc_close` leak this symmetry exists to prevent).
#[tauri::command]
pub async fn skel_close(asset_id: String, state: State<'_, SkelState>) -> Result<(), String> {
    state.with(|store| {
        store.remove(&asset_id);
        Ok(())
    })
}

/// Every open document.
#[tauri::command]
pub async fn skel_list(
    state: State<'_, SkelState>,
    assets: State<'_, AssetState>,
) -> Result<Vec<SkelDocDto>, String> {
    let ids: Vec<String> = state.with(|store| Ok(store.keys().cloned().collect()))?;
    let mut out = Vec::with_capacity(ids.len());
    for id in ids {
        let guid: AssetId = match id.parse() {
            Ok(g) => g,
            Err(_) => continue,
        };
        let name = asset_name(&assets, guid).unwrap_or_else(|_| id.clone());
        out.push(state.with(|store| Ok(doc_dto(&id, &name, doc_of(store, &id)?)))?);
    }
    Ok(out)
}

/// **Apply one edit to a session and build the reply** — the funnel's whole
/// rule, as a pure function.
///
/// # Why this is separate from [`edit`] (G2)
///
/// The snapshot-and-restore used to live inside the `#[tauri::command]` wrapper,
/// and the audit measured what that cost: **severing `*s = before;` produced
/// zero test failures.** `inf-studio` has no `tests/` directory, the wrapper
/// needs a `State<'_, SkelState>` and an `AppHandle` that no workspace test can
/// construct, and the Ring-1 property test drives `SkelSession` directly — so the
/// funnel's headline claim was unreachable from every gate in the repository.
///
/// Splitting the rule out of the plumbing fixes that without pretending the
/// plumbing is testable: this function takes a plain `&mut SkelSession` and
/// returns the DTO, so `a_refusal_through_the_funnel_restores_the_session` below
/// drives the real code path, and severing the restore fails it by name. What
/// stays in [`edit`] is the lock, the name lookup and the event emit — which are
/// the parts a test genuinely cannot reach, and which own no rule.
///
/// The property: **whatever `f` did on its way to refusing is undone**, including
/// a checkpoint it had already pushed. The whole session is restored, not just
/// the asset, so the undo stacks stay honest.
fn apply_edit(
    session: &mut SkelSession,
    id: &str,
    name: &str,
    f: impl FnOnce(&mut SkelSession) -> Result<Option<String>, SkelError>,
) -> SkelApplyDto {
    let before = session.clone();
    match f(session) {
        Ok(warning) => SkelApplyDto {
            ok: true,
            refusal: None,
            warning,
            doc: doc_dto(id, name, session),
        },
        Err(e) => {
            *session = before;
            SkelApplyDto {
                ok: false,
                refusal: Some(e.to_string()),
                warning: None,
                doc: doc_dto(id, name, session),
            }
        }
    }
}

/// Run one edit and answer with the new document, or the refusal.
///
/// The lock, the name lookup and the `skel://sync` emit; the rule itself is
/// [`apply_edit`], which is where a test can reach it.
///
/// **Every mutating command routes through this, `skel_save` included**, so the
/// emit cannot be forgotten on one of them. That routing is a *convention*
/// enforced by review — there is no compile-time exhaustiveness here, and the
/// earlier claim that a new mutator would be a build error was wrong (a
/// mutate-then-refuse mutator compiles clean). What IS enforced is that any
/// mutator which does route through gets the property, and that `SkelSession`'s
/// own refusal paths leave it untouched (`every_refusal_leaves_the_session_untouched`).
async fn edit(
    app: &AppHandle,
    id: &str,
    state: &State<'_, SkelState>,
    assets: &State<'_, AssetState>,
    f: impl FnOnce(&mut SkelSession) -> Result<Option<String>, SkelError>,
) -> Result<SkelApplyDto, String> {
    let guid: AssetId = id.parse().map_err(|e| format!("bad asset id: {e}"))?;
    let name = asset_name(assets, guid)?;
    let out = state.with(|store| Ok(apply_edit(doc_of(store, id)?, id, &name, f)))?;
    let _ = app.emit("skel://sync", id);
    Ok(out)
}

#[tauri::command]
pub async fn skel_rename_joint(
    app: AppHandle,
    id: String,
    joint: u16,
    name: String,
    state: State<'_, SkelState>,
    assets: State<'_, AssetState>,
) -> Result<SkelApplyDto, String> {
    edit(&app, &id, &state, &assets, |s| {
        s.rename_joint(joint, &name).map(|v| rename_warning(&v))
    })
    .await
}

/// Move a joint's **rest** transform. Every inverse bind is recomputed — see
/// `SkelSession::set_joint_local` for why that is not optional.
#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn skel_set_joint_transform(
    app: AppHandle,
    id: String,
    joint: u16,
    translation: [f32; 3],
    rotation: [f32; 4],
    scale: [f32; 3],
    state: State<'_, SkelState>,
    assets: State<'_, AssetState>,
) -> Result<SkelApplyDto, String> {
    edit(&app, &id, &state, &assets, |s| {
        s.set_joint_local(
            joint,
            JointTransform {
                translation,
                rotation,
                scale,
            },
        )
        .map(|()| None)
    })
    .await
}

/// Set or clear a joint's rotation limit. Passing either bound as `null`
/// **clears the row** — absent means unlimited, and a full-range row is a
/// different (authored) statement.
#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn skel_set_limit(
    app: AppHandle,
    id: String,
    joint: u16,
    min_deg: Option<[f32; 3]>,
    max_deg: Option<[f32; 3]>,
    state: State<'_, SkelState>,
    assets: State<'_, AssetState>,
) -> Result<SkelApplyDto, String> {
    edit(&app, &id, &state, &assets, |s| {
        let limit = match (min_deg, max_deg) {
            (Some(min_deg), Some(max_deg)) => Some(JointLimit {
                joint,
                min_deg,
                max_deg,
            }),
            _ => None,
        };
        s.set_limit(joint, limit).map(|()| None)
    })
    .await
}

#[tauri::command]
pub async fn skel_add_socket(
    app: AppHandle,
    id: String,
    name: String,
    joint: u16,
    state: State<'_, SkelState>,
    assets: State<'_, AssetState>,
) -> Result<SkelApplyDto, String> {
    edit(&app, &id, &state, &assets, |s| {
        s.add_socket(&name, joint).map(|()| None)
    })
    .await
}

#[tauri::command]
pub async fn skel_remove_socket(
    app: AppHandle,
    id: String,
    name: String,
    state: State<'_, SkelState>,
    assets: State<'_, AssetState>,
) -> Result<SkelApplyDto, String> {
    edit(&app, &id, &state, &assets, |s| {
        s.remove_socket(&name).map(|()| None)
    })
    .await
}

#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn skel_place_socket(
    app: AppHandle,
    id: String,
    name: String,
    joint: u16,
    translation: [f32; 3],
    rotation: [f32; 4],
    scale: [f32; 3],
    state: State<'_, SkelState>,
    assets: State<'_, AssetState>,
) -> Result<SkelApplyDto, String> {
    edit(&app, &id, &state, &assets, |s| {
        s.place_socket(
            &name,
            joint,
            JointTransform {
                translation,
                rotation,
                scale,
            },
        )
        .map(|()| None)
    })
    .await
}

/// **Replace** this rig with a freshly generated template — undoable, so an
/// author exploring proportions gets their hand-built rig back.
#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn skel_instantiate_template(
    app: AppHandle,
    id: String,
    plan: String,
    legs: Option<u16>,
    height_m: Option<f64>,
    state: State<'_, SkelState>,
    assets: State<'_, AssetState>,
) -> Result<SkelApplyDto, String> {
    let plan = parse_plan(&plan, legs)?;
    let params = BodyParams {
        height_m: height_m.unwrap_or(BodyParams::default().height_m),
        ..BodyParams::default()
    };
    edit(&app, &id, &state, &assets, |s| {
        s.instantiate_template(plan, &params).map(|()| None)
    })
    .await
}

/// **Append another rig onto this one** at `attach` — the modular-rigging door
/// (`inf_anim::merge_skeletons`). A socket- or joint-name collision refuses, by
/// name, and leaves the rig untouched.
#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn skel_merge_part(
    app: AppHandle,
    id: String,
    part_asset: String,
    attach: u16,
    state: State<'_, SkelState>,
    assets: State<'_, AssetState>,
) -> Result<SkelApplyDto, String> {
    let part_id: AssetId = part_asset
        .parse()
        .map_err(|e| format!("bad asset id: {e}"))?;
    let part = assets.with_project(|p| {
        p.load_payload::<SkeletonAsset>(part_id)
            .map_err(|e| e.to_string())
    })?;
    edit(&app, &id, &state, &assets, |s| {
        s.merge_part(&part, attach).map(|offset| {
            Some(format!(
                "the part's joints start at index {offset}; a mesh skinned to it \
                 needs its weight table shifted by the same amount"
            ))
        })
    })
    .await
}

#[tauri::command]
pub async fn skel_undo(
    app: AppHandle,
    id: String,
    state: State<'_, SkelState>,
    assets: State<'_, AssetState>,
) -> Result<SkelApplyDto, String> {
    edit(&app, &id, &state, &assets, |s| {
        s.undo();
        Ok(None)
    })
    .await
}

#[tauri::command]
pub async fn skel_redo(
    app: AppHandle,
    id: String,
    state: State<'_, SkelState>,
    assets: State<'_, AssetState>,
) -> Result<SkelApplyDto, String> {
    edit(&app, &id, &state, &assets, |s| {
        s.redo();
        Ok(None)
    })
    .await
}

/// **The `SkeletonAsset` writer this engine did not have.**
///
/// Goes through `AssetProject::rewrite_payload` — the generic door every
/// data-asset editor saves through — so the content hash and the sidecar move
/// with the bytes and the Content Drawer sees the change. A skeleton references
/// no other asset, so the dependency list is empty and stays that way.
///
/// `mark_saved` runs **only after** the write returns, so a failed write leaves
/// the document dirty (the `save_mesh_session` disk-truth rule). There is no
/// derived-artifact arm because a skeleton has no derived artifact: the
/// `SaveError::Derived`/`Torn` states the mesh save exists to prevent have no
/// analogue here.
#[tauri::command]
pub async fn skel_save(
    app: AppHandle,
    id: String,
    state: State<'_, SkelState>,
    assets: State<'_, AssetState>,
) -> Result<SkelApplyDto, String> {
    let guid: AssetId = id.parse().map_err(|e| format!("bad asset id: {e}"))?;
    let payload = state.with(|store| Ok(doc_of(store, &id)?.asset().clone()))?;
    // The write happens OUTSIDE the store lock and before the funnel, because it
    // is I/O and the funnel holds a mutex. Its verdict is then what the funnel's
    // closure returns, so a failed write takes the refusal path like any other —
    // and `mark_saved` runs only on the success arm, which is the disk-truth rule
    // (`SkelSession::mark_saved`).
    let written = assets.with_project(|p| {
        p.rewrite_payload(guid, &payload, vec![])
            .map_err(|e| e.to_string())
    });
    let ok = written.is_ok();
    let out = edit(&app, &id, &state, &assets, |s| match written {
        Ok(()) => {
            s.mark_saved();
            Ok(None)
        }
        Err(e) => Err(SkelError::Write(e)),
    })
    .await?;
    if ok {
        emit_changed(&app, &assets);
    }
    Ok(out)
}

/// **Auto-fit a template to a mesh** — the door `inf_dcc::fit_template` has not
/// had since P24.2.
///
/// The fit was built, tested and reachable only from its own unit tests: no
/// Ring-2 command, no `ipc.ts` wrapper, so an author could not run it. The hop
/// that was missing is `CanonicalMesh` → triangles, and it is now
/// `inf_editor_core::dcc::triangle_soup`, verified against a hand-computed
/// closest-point query rather than against the BVH's own opinion.
///
/// The fitted rig **replaces** the open one as one undoable edit, and the report
/// comes back with it: a fit that placed most joints outside the mesh is one an
/// author needs to see rather than infer from the viewport.
#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn skel_fit_to_mesh(
    app: AppHandle,
    id: String,
    mesh_asset: String,
    plan: String,
    legs: Option<u16>,
    height_m: Option<f64>,
    state: State<'_, SkelState>,
    assets: State<'_, AssetState>,
) -> Result<SkelApplyDto, String> {
    let plan = parse_plan(&plan, legs)?;
    let mesh_id: AssetId = mesh_asset
        .parse()
        .map_err(|e| format!("bad asset id: {e}"))?;
    let payload = assets.with_project(|p| {
        let entry = p
            .db()
            .get(mesh_id)
            .ok_or_else(|| format!("no asset {mesh_id}"))?;
        if entry.kind() != AssetKind::Mesh {
            return Err(format!("{} is not a mesh asset", entry.name));
        }
        p.load_payload::<inf_mesh::MeshAsset>(mesh_id)
            .map_err(|e| e.to_string())
    })?;
    // Everything that can fail does so BEFORE the session is touched, so a
    // refusal never leaves a half-fitted rig behind.
    let imported = inf_dcc::from_mesh_asset(&payload).map_err(|e| e.to_string())?;
    let geo = inf_editor_core::dcc::tessellate(&imported.mesh);
    let bvh = inf_dcc::Bvh::new(inf_editor_core::dcc::triangle_soup(&geo));
    let opts = inf_dcc::FitOptions {
        plan,
        params: BodyParams {
            height_m: height_m.unwrap_or(BodyParams::default().height_m),
            ..BodyParams::default()
        },
        ..inf_dcc::FitOptions::default()
    };
    let fitted = inf_dcc::fit_template(&bvh, &opts);

    edit(&app, &id, &state, &assets, |s| match fitted {
        Ok((asset, report)) => {
            s.replace_asset(asset);
            Ok(Some(format!(
                "fitted to a {:.2} m mesh: {} of {} joints landed inside it, symmetry {:.2}, {} refinement steps rejected",
                report.height_m,
                report.joints_inside,
                report.joints,
                report.symmetry_score,
                report.steps_rejected
            )))
        }
        // The fit's own refusal, carried through as a VALUE — it names the
        // degenerate measurement, which is the whole point of `FitError`.
        Err(e) => Err(SkelError::Template(e.to_string())),
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn biped() -> SkeletonAsset {
        build_template(BodyPlan::Biped, &BodyParams::default()).unwrap()
    }

    /// **A refusal through the funnel restores the session** (G2).
    ///
    /// The arm the audit found missing: severing `*s = before;` used to produce
    /// zero failures anywhere in the workspace. This drives `apply_edit` — the
    /// real code path the Tauri wrapper calls — with a closure that **mutates and
    /// then refuses**, which is exactly the case precondition-checking inside
    /// `SkelSession` cannot cover and the only reason the funnel owns a snapshot
    /// at all.
    #[test]
    fn a_refusal_through_the_funnel_restores_the_session() {
        let mut s = SkelSession::new(biped());
        s.add_socket("keepsake", 0).expect("seed");
        let before = s.asset().clone();
        let (undo, redo, dirty) = (s.can_undo(), s.can_redo(), s.is_dirty());

        let dto = apply_edit(&mut s, "id", "Rig", |sess| {
            // Mutate FIRST, then refuse — a mutator that validates halfway
            // through. Nothing in `SkelSession` protects against this; the
            // funnel's snapshot is the only thing that does.
            sess.add_socket("half_applied", 1)?;
            sess.rename_joint(0, "renamed_before_refusing")?;
            Err(SkelError::Write("the disk went away".into()))
        });

        assert!(!dto.ok);
        assert_eq!(dto.refusal.as_deref(), Some("the disk went away"));
        assert_eq!(
            s.asset(),
            &before,
            "the funnel did not restore the session after a mutator refused partway through"
        );
        assert!(
            !s.asset().sockets.iter().any(|k| k.name == "half_applied"),
            "a socket added on the way to a refusal survived"
        );
        assert_eq!(s.can_undo(), undo, "the refusal left an undo step behind");
        assert_eq!(s.can_redo(), redo, "the refusal dropped the redo branch");
        assert_eq!(s.is_dirty(), dirty, "the refusal changed the dirty flag");
        // …and the DTO it returned describes the restored document, not the
        // half-applied one — a panel adopting it must not show the mutation.
        assert!(!dto.doc.sockets.iter().any(|k| k.name == "half_applied"));
        assert_eq!(dto.doc.joints[0].name, before.skeleton.joints()[0].name);
    }

    /// …and a SUCCESS is not rolled back — the anti-vacuity twin, without which
    /// a funnel that restored unconditionally would pass the test above.
    #[test]
    fn a_successful_edit_through_the_funnel_is_kept() {
        let mut s = SkelSession::new(biped());
        let dto = apply_edit(&mut s, "id", "Rig", |sess| {
            sess.add_socket("scabbard", 1)?;
            Ok(Some("a warning".into()))
        });
        assert!(dto.ok && dto.refusal.is_none());
        assert_eq!(dto.warning.as_deref(), Some("a warning"));
        assert!(s.asset().sockets.iter().any(|k| k.name == "scabbard"));
        assert!(dto.doc.sockets.iter().any(|k| k.name == "scabbard"));
        assert!(s.is_dirty() && s.can_undo());
    }

    /// **The DTO projection tells the panel the three things it cannot derive**:
    /// which joints are canonical, which have a twin, and which have a limit.
    ///
    /// Asserted on a real template rather than a hand-built rig, so the flags are
    /// measured against what the generator actually emits.
    #[test]
    fn the_joint_projection_carries_the_flags_the_panel_needs() {
        let a = biped();
        let by = |n: &str| {
            let i = a
                .skeleton
                .index_of(n)
                .unwrap_or_else(|| panic!("{n} missing"));
            joint_dto(&a, i as usize)
        };

        let arm = by("upper_arm_l");
        assert!(arm.canonical, "upper_arm_l is one of the nineteen");
        assert_eq!(
            arm.mirror,
            a.skeleton.index_of("upper_arm_r"),
            "the twin was not resolved"
        );
        assert!(!arm.sided_without_twin);
        assert!(arm.limit_min_deg.is_none(), "a shoulder is unlimited");

        // A knee carries its hinge, as BOTH bounds — a projection that dropped
        // one would leave the panel unable to tell a hinge from a free joint.
        let knee = by("lower_leg_r");
        assert_eq!(knee.limit_min_deg.map(|d| d[0]), Some(-150.0));
        assert_eq!(knee.limit_max_deg.map(|d| d[0]), Some(0.0));

        // `pelvis` is deliberately NOT one of the nineteen (template.rs says so).
        assert!(!by("pelvis").canonical);
        assert!(!by("pelvis").sided_without_twin, "it names no side");

        // …and the rest transform really is the joint's, not a default.
        assert_eq!(
            arm.translation,
            a.skeleton.joints()[arm.index as usize]
                .local_bind
                .translation
        );
    }

    /// A rig with one side of a pair missing is projected as `sidedWithoutTwin`
    /// AND appears in the document's `unmatchedSided` list — the anti-vacuity arm
    /// for both, since a flag that is always `false` would pass the test above.
    #[test]
    fn a_half_sided_rig_is_flagged_on_the_joint_and_on_the_document() {
        let mut s = SkelSession::new(biped());
        let i = s.asset().skeleton.index_of("hand_r").unwrap();
        s.rename_joint(i, "grabber").expect("renames");
        let doc = doc_dto("id", "Rig", &s);
        let hand_l = doc
            .joints
            .iter()
            .find(|j| j.name == "hand_l")
            .expect("hand_l is still there");
        assert!(
            hand_l.sided_without_twin,
            "hand_l lost its twin and the projection did not say so"
        );
        assert!(hand_l.mirror.is_none());
        assert!(doc.unmatched_sided.contains(&"hand_l".to_string()));
        assert!(doc.dirty && doc.can_undo && !doc.can_redo);
    }

    /// The rename warning names BOTH costs separately, and a harmless rename
    /// produces none — so the panel never shows an empty warning banner.
    #[test]
    fn the_rename_warning_says_what_it_cost_or_nothing() {
        let none = rename_warning(&RenameVerdict {
            left_humanoid_set: false,
            broke_mirror_pair: false,
            previous: "pelvis".into(),
        });
        assert_eq!(none, None, "a harmless rename must warn about nothing");

        let both = rename_warning(&RenameVerdict {
            left_humanoid_set: true,
            broke_mirror_pair: true,
            previous: "upper_arm_l".into(),
        })
        .expect("a costly rename warns");
        assert!(both.contains("upper_arm_l"));
        assert!(both.contains("retargeting"), "{both}");
        assert!(both.contains("mirroring"), "{both}");

        // Each cost alone reports only itself.
        let one = rename_warning(&RenameVerdict {
            left_humanoid_set: true,
            broke_mirror_pair: false,
            previous: "hips".into(),
        })
        .unwrap();
        assert!(
            one.contains("retargeting") && !one.contains("mirroring"),
            "{one}"
        );
    }

    /// The socket projection round-trips a placed socket's joint and offset —
    /// the half a name-only DTO would lose.
    #[test]
    fn the_socket_projection_carries_the_joint_and_the_offset() {
        let mut s = SkelSession::new(biped());
        s.add_socket("scabbard", 0).unwrap();
        let off = JointTransform::from_trs(
            glam::Vec3::new(0.1, -0.2, 0.0),
            glam::Quat::IDENTITY,
            glam::Vec3::ONE,
        );
        s.place_socket("scabbard", 3, off).unwrap();
        let doc = doc_dto("id", "Rig", &s);
        let sock = doc
            .sockets
            .iter()
            .find(|k| k.name == "scabbard")
            .expect("projected");
        assert_eq!(sock.joint, 3);
        assert_eq!(sock.translation, off.translation);
    }

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
