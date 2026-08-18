//! **New Character from Template** (P24.5) — the wizard's Ring-1 half.
//!
//! *Pick a plan → shape it → auto-rig → a default locomotion set wired to a
//! state machine.* The anti-pain headline, and it is a **composition** of doors
//! that already existed rather than new simulation:
//!
//! | step | door | landed |
//! |---|---|---|
//! | pick a plan | [`inf_anim::BodyPlan`] | P24.1 |
//! | shape it | [`inf_anim::BodyParams`] → [`inf_anim::build_template`] | P24.1 |
//! | auto-rig to a mesh | [`inf_dcc::fit_template`] | P24.2 |
//! | solve weights | [`inf_dcc::solve_heat_weights`] | P24.2 |
//! | clips + machine | [`inf_anim::build_locomotion`] / [`inf_anim::locomotion_machine`] | P24.5 |
//! | the assets | [`AssetProject::write_asset`] | P4 |
//! | the actor | [`crate::scene::SceneDoc::edit_create_character`] | P24.5 |
//!
//! The two new things are the last row of that table and the row above it. The
//! rest is wiring, and saying so is the point: a wizard that reimplemented any
//! of the middle rows would be a second spelling of a rule that already has one.
//!
//! # Everything is generated BEFORE anything is written
//!
//! [`build_character`] validates the plan, builds the rig, fits it, solves the
//! weights and generates all three clips *in memory*, and only then starts
//! writing assets — the `skel_create_template` rule, applied to a job that
//! writes six assets instead of one.
//!
//! Generating first is **not enough on its own**, and the difference is what
//! [`roll_back`] is for: the sixth `write_asset` can still fail on a full disk,
//! a path the platform refuses or a sidecar that cannot be created, long after
//! the first five have landed. Measured before it was fixed — a blocked fourth
//! write left three registered assets and a payload with no sidecar, which the
//! content watcher promotes under a *fresh* GUID. So a failed write takes back
//! everything this call wrote, and
//! `a_write_that_fails_halfway_takes_back_what_it_wrote` checks that against the
//! **filesystem** rather than against the build's own verdict (the P23 lesson
//! that made `SaveError::Torn` reachable at all).
//!
//! # The body
//!
//! Two paths, and which one runs is decided by whether the author brought a mesh:
//!
//! * **They did** — the rig is *fitted* to it ([`inf_dcc::fit_template`]),
//!   the mesh is bound to that rig and its weights are solved
//!   ([`inf_dcc::solve_heat_weights`]), and the result is written as a **new**
//!   `.inf_mesh` beside the character. The author's own asset is never rewritten:
//!   a wizard that silently reskinned an imported mesh would be destroying the
//!   thing it was given.
//! * **They did not** — [`block_body_mesh`] builds a blocky mannequin, one box
//!   per bone, each box rigid to the bone it wraps. It is not a model; it is the
//!   difference between a character you can see walking in PIE and an empty
//!   transform, and it is deliberately obvious about being a placeholder. The
//!   alternative considered and rejected was `SkeletalMesh { mesh: None }`, which
//!   draws the renderer's placeholder **cube** — one cube for a whole creature,
//!   which tells an author nothing about whether their rig moves.
//!
//! # Units
//!
//! Metres and seconds throughout (the units doctrine). The only degrees in this
//! module are [`inf_anim::GaitParams`]' angles and [`inf_anim::JointLimit`]'s,
//! which are the authoring convention.

use std::path::Path;

use inf_anim::locomotion::{build_locomotion, locomotion_machine, GaitParams};
use inf_anim::{AnimClipAsset, BodyParams, BodyPlan, SkeletonAsset, StateMachineAsset};
use inf_asset::{AssetId, AssetKind, AssetPayload};
use inf_mesh::{MeshAsset, MeshVertex, SubMesh, VertexSkin};

use crate::assets::AssetProject;

/// The content sub-folder a generated character's assets land in.
pub const CHARACTER_FOLDER: &str = "Characters";

/// How wide a mannequin box is, as a fraction of the bone it wraps.
const BOX_RADIUS_OF_BONE: f64 = 0.22;
/// …clamped into this fraction of the whole rig's height, so a 5 cm finger bone
/// is not invisible and a 1 m spine segment is not a barrel.
const BOX_RADIUS_MIN_OF_HEIGHT: f64 = 0.012;
const BOX_RADIUS_MAX_OF_HEIGHT: f64 = 0.070;

/// Why a character refused to generate.
///
/// Every variant carries the *underlying* door's own message rather than a
/// rewritten one, because those messages name the offending parameter, joint or
/// asset and that text is what reaches the author.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum CharacterError {
    /// The template generator refused the proportions.
    #[error("{0}")]
    Template(String),
    /// The locomotion generator refused the gait, or the rig.
    #[error("{0}")]
    Locomotion(String),
    /// The auto-fit refused the mesh.
    #[error("{0}")]
    Fit(String),
    /// The weight solve refused.
    #[error("{0}")]
    Weights(String),
    /// The named mesh asset is missing, of the wrong kind, or unreadable.
    #[error("{0}")]
    Mesh(String),
    /// An asset could not be written.
    #[error("{0}")]
    Write(String),
    /// A name that would produce no file.
    #[error("a character needs a name")]
    EmptyName,
}

/// Everything the wizard collects. One struct, so a preview and a build are the
/// **same** input and a preview cannot describe a character the build would not
/// make.
#[derive(Debug, Clone, PartialEq)]
pub struct CharacterSpec {
    /// Base name for every generated asset (`"Hero"` → `Hero.inf_skel`,
    /// `Hero Walk.inf_anim`, `Hero Locomotion.inf_sm`, …).
    pub name: String,
    pub plan: BodyPlan,
    pub params: BodyParams,
    pub gait: GaitParams,
    /// An existing `.inf_mesh` to fit the rig to and skin. `None` builds a blocky
    /// mannequin from the rig instead (see the module docs).
    pub mesh: Option<AssetId>,
}

impl Default for CharacterSpec {
    fn default() -> Self {
        Self {
            name: "Character".to_string(),
            plan: BodyPlan::Biped,
            params: BodyParams::default(),
            gait: GaitParams::default(),
            mesh: None,
        }
    }
}

/// One joint, as the wizard's live preview draws it. The same three numbers the
/// Skeleton Editor's SVG diagram projects from, deliberately — the wizard's
/// preview and the editor's are the same picture of the same rig.
#[derive(Debug, Clone, PartialEq)]
pub struct PreviewJoint {
    pub name: String,
    pub parent: Option<u16>,
    /// Local rest translation, metres.
    pub translation: [f32; 3],
}

/// What the wizard can say about a spec **without writing anything**.
///
/// Recomputed on every slider drag, so it is the whole feedback loop of the
/// "shape it" step: the rig really is generated, the clips really are generated,
/// and the numbers below are read off them rather than predicted.
#[derive(Debug, Clone, PartialEq)]
pub struct CharacterPreview {
    pub joints: Vec<PreviewJoint>,
    pub sockets: Vec<String>,
    /// How many joints carry a rotation limit — the IK input P24.1 emits.
    pub limits: usize,
    /// The rig's actual extent along Y in the bind pose (metres). Not
    /// `params.height_m`: a fitted rig's height comes from the mesh, and reading
    /// it back is how an author sees that the fit worked.
    pub height_m: f64,
    /// Each driven leg's `(name, length_m, gait phase)`.
    pub legs: Vec<(String, f64, f64)>,
    /// The generated cycles' durations, seconds: idle, walk, run.
    pub durations_s: [f32; 3],
    /// The speeds the machine's thresholds are derived from (m/s).
    pub walk_speed_m_s: f64,
    pub run_speed_m_s: f64,
    pub walk_threshold_m_s: f64,
    pub run_threshold_m_s: f64,
    /// Vertices and triangles of the mannequin body the build would generate.
    /// `None` when the spec brings its own mesh.
    pub body: Option<(usize, usize)>,
}

/// What an auto-fit did, when one ran.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FitSummary {
    pub height_m: f64,
    pub joints_inside: usize,
    pub joints: usize,
    pub symmetry_score: f64,
}

/// What the weight solve did, when one ran.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeightSummary {
    /// Vertices whose weights the solve wrote.
    pub assigned: usize,
    /// Vertices no bone could see — they keep "all of joint 0", which is the
    /// state an author needs told about rather than left to notice.
    pub unreached: usize,
    pub worst_residual: f64,
}

/// The assets a build produced, and what happened on the way.
#[derive(Debug, Clone, PartialEq)]
pub struct CharacterBuild {
    pub skeleton: AssetId,
    pub mesh: AssetId,
    pub idle: AssetId,
    pub walk: AssetId,
    pub run: AssetId,
    pub machine: AssetId,
    /// Whether `mesh` is the generated mannequin (`true`) or a skinned copy of
    /// the author's own mesh (`false`).
    pub mannequin: bool,
    pub fit: Option<FitSummary>,
    pub weights: Option<WeightSummary>,
    /// Things that happened and are not refusals — an unreached region, a fit
    /// that placed joints outside the mesh. The panel shows them; nothing here
    /// is a failure.
    pub warnings: Vec<String>,
}

/// **Fit a template rig to a model** — `MeshAsset` → kernel → tessellation →
/// BVH → [`inf_dcc::fit_template`], as one door.
///
/// P24.3 opened this hop inside the Ring-2 `skel_fit_to_mesh` command; the wizard
/// needs the identical four lines, and two spellings of "how you turn an asset
/// into something the fit can see" is the shape the P22 *one door for three
/// paths* law is about. So it lives here and the Skeleton Editor's command calls
/// it.
///
/// The **whole** [`inf_dcc::FitReport`] comes back rather than a projection,
/// because the Skeleton Editor's readout names `steps_rejected` and the wizard's
/// does not; a shared door that returned only the intersection would make the
/// panel poorer to make the wizard tidier.
pub fn fit_rig_to_mesh(
    plan: BodyPlan,
    params: &BodyParams,
    payload: &MeshAsset,
) -> Result<(SkeletonAsset, inf_dcc::FitReport), CharacterError> {
    fit_rig_to_bvh(&fit_bvh(payload)?, plan, params)
}

/// The **expensive half** of [`fit_rig_to_mesh`], split out (P29.5).
///
/// Kernel → tessellation → BVH. It depends on the *model* and on nothing the
/// wizard's sliders touch, which is the whole reason the split exists: a
/// proportion drag re-fits, and re-fitting is cheap; rebuilding this per
/// keystroke is what the ROADMAP's F5 item was about.
pub fn fit_bvh(payload: &MeshAsset) -> Result<inf_dcc::Bvh, CharacterError> {
    let imported =
        inf_dcc::from_mesh_asset(payload).map_err(|e| CharacterError::Mesh(e.to_string()))?;
    let geo = crate::dcc::tessellate(&imported.mesh);
    Ok(inf_dcc::Bvh::new(crate::dcc::triangle_soup(&geo)))
}

/// The **cheap half**: the fit itself, against a BVH somebody else built.
pub fn fit_rig_to_bvh(
    bvh: &inf_dcc::Bvh,
    plan: BodyPlan,
    params: &BodyParams,
) -> Result<(SkeletonAsset, inf_dcc::FitReport), CharacterError> {
    let opts = inf_dcc::FitOptions {
        plan,
        params: *params,
        ..inf_dcc::FitOptions::default()
    };
    inf_dcc::fit_template(bvh, &opts).map_err(|e| CharacterError::Fit(e.to_string()))
}

/// The rig a spec produces, plus what fitting it cost. Shared by the preview and
/// the build so the two cannot disagree about what rig they are describing.
fn rig_for(
    spec: &CharacterSpec,
    fit_source: Option<&MeshAsset>,
) -> Result<(SkeletonAsset, Option<FitSummary>), CharacterError> {
    let Some(payload) = fit_source else {
        let asset = inf_anim::build_template(spec.plan, &spec.params)
            .map_err(|e| CharacterError::Template(e.to_string()))?;
        return Ok((asset, None));
    };
    let (asset, report) = fit_rig_to_mesh(spec.plan, &spec.params, payload)?;
    Ok((
        asset,
        Some(FitSummary {
            height_m: report.height_m,
            joints_inside: report.joints_inside,
            joints: report.joints,
            symmetry_score: report.symmetry_score,
        }),
    ))
}

/// The rig's bind-pose extent along Y, metres — the span between its lowest and
/// highest **joint**.
///
/// **Not the creature's standing height**, and the difference is not noise: a
/// template rig's topmost joint is `head`, which
/// [`inf_anim::build_template`] places at `head_height_ratio × height_m`, so a
/// 1.75 m biped measures **1.6275** here. The skull above that joint is geometry
/// and geometry has no joints. A fitted rig measures whatever the fit put in the
/// mesh, which is the number worth reading — it is how an author sees the fit
/// took the model's proportions rather than the spec's.
///
/// Pinned as that identity rather than with a tolerance by
/// `the_previewed_height_is_the_joint_span_and_not_the_requested_height`; the
/// panel labels the row accordingly, because "Height: 1.63" beside a field the
/// author typed 1.75 into reads as a bug.
fn rig_height_m(rig: &SkeletonAsset) -> f64 {
    let mut mats: Vec<glam::Mat4> = Vec::with_capacity(rig.skeleton.len());
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for j in rig.skeleton.joints() {
        let local = j.local_bind.to_mat4();
        let m = match j.parent {
            Some(p) => mats[p as usize] * local,
            None => local,
        };
        let y = m.w_axis.y;
        lo = lo.min(y);
        hi = hi.max(y);
        mats.push(m);
    }
    if lo.is_finite() && hi.is_finite() {
        (hi - lo) as f64
    } else {
        0.0
    }
}

/// Describe what [`build_character`] would produce, without touching the disk.
///
/// `fit_source` is the decoded `.inf_mesh` when the spec names one. Taking the
/// payload rather than an [`AssetProject`] is a **ring** decision — this function
/// has no business holding a database — and it is worth saying plainly that it is
/// not a performance one, because the first version of this sentence claimed it
/// was.
///
/// # What a slider drag actually costs, stated rather than implied
///
/// The panel re-previews on every edit and nothing debounces it, so on the
/// **fitted** path one keystroke is: `AssetProject::load_payload` (a `read` plus
/// a decode of the whole mesh, uncached), then [`fit_rig_to_mesh`] — a kernel
/// build, a tessellation, a BVH and the fit itself — and then all three clips.
/// The mannequin path is cheap and is the default; the fitted path is linear in
/// the author's model and will be felt on a real one.
///
/// **The cold path, kept**: one preview with nothing remembered. The build path
/// and every test use it, and it is [`CharacterPreviewSession::preview`] with a
/// session that is thrown away — one rule, two lifetimes.
pub fn preview_character(
    spec: &CharacterSpec,
    fit_source: Option<&MeshAsset>,
) -> Result<CharacterPreview, CharacterError> {
    CharacterPreviewSession::default().preview(spec, fit_source.map(|m| (m, 0)))
}

/// **The wizard's kept-warm preview** (P29.5, ROADMAP §13's F5 item; the P23.2a
/// `PreviewSession` law applied to a rig instead of to a GPU).
///
/// # What was actually slow, and it was not the fit
///
/// The panel re-previews on **every edit** and nothing debounces it, so before
/// this a single keystroke on the fitted path was: a `read` plus a full decode of
/// the author's model (Ring 2), a half-edge kernel build, a tessellation, a BVH
/// **build**, the fit, and then all three locomotion clips generated from
/// scratch — and, on the mannequin path, a whole block body mesh generated so
/// that two integers could be counted off it.
///
/// Of those, exactly one depends on the slider being dragged. A session
/// remembers the other three:
///
/// * the **BVH**, keyed by the caller's model stamp — a proportion drag re-fits
///   against a BVH that is already built;
/// * the **locomotion set**'s summary, keyed by `(plan, params, gait)` — a
///   *proportion* drag regenerates it (the clips are derived from the rig) but a
///   re-preview at the same spec does not, and neither does a drag that only
///   moves the model;
/// * the **body counts**, keyed by `(plan, params)` — a *gait* drag does not
///   rebuild a mesh whose vertex count cannot have changed.
///
/// # The stamp is the caller's, and that is deliberate
///
/// A session cannot hash a `MeshAsset` cheaply enough to be worth it (that is the
/// decode it exists to avoid paying twice), so the caller supplies a stamp that
/// identifies the model — Ring 2 passes the asset's content hash, which is the
/// same key `EditorRenderAssets` re-keys on. `0` means "no stamp", and a session
/// handed `0` rebuilds every time rather than guessing: `PreviewSession`'s own
/// rule, where `0` is the built-in sphere and not a caller's geometry.
///
/// # Counters, not claims
///
/// [`builds`](Self::builds) reports how much work was actually done. It exists so
/// the warm path is asserted rather than described — a cache that silently stopped
/// hitting would keep every number in the preview correct.
#[derive(Default)]
pub struct CharacterPreviewSession {
    bvh: Option<(u64, inf_dcc::Bvh)>,
    loco: Option<(LocoKey, LocoSummary)>,
    body: Option<((BodyPlan, BodyParams), (usize, usize))>,
    builds: PreviewBuilds,
}

/// What a [`CharacterPreviewSession`] rebuilt, cumulatively.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PreviewBuilds {
    /// BVHs built from a decoded model.
    pub bvh: usize,
    /// Locomotion sets generated (three clips each).
    pub locomotion: usize,
    /// Mannequin body meshes generated, to count their vertices.
    pub body: usize,
    /// Previews answered — the denominator.
    pub previews: usize,
}

/// What the locomotion cache is keyed on: everything `build_locomotion` reads.
///
/// The last element is the **fit source's identity**, and it is an `Option`
/// rather than a bare stamp (P29.5 audit, A8): `None` is the mannequin path,
/// whose rig comes from `build_template`, and `Some(s)` is a fitted one, whose
/// rig comes from a BVH. A bare `0` for both would let an unstamped fitted
/// preview seed the key a *mannequin* preview then hits, and answer a mannequin
/// with the fitted model's leg lengths — which is exactly the "two unidentified
/// models are indistinguishable" hazard the stamp exists for, met from the
/// other side.
type LocoKey = (
    BodyPlan,
    BodyParams,
    inf_anim::locomotion::GaitParams,
    Option<u64>,
);

/// The part of a [`inf_anim::LocomotionSet`] a preview reports — kept instead of
/// the set, because the set holds three whole clips and the preview reads six
/// numbers and a leg list off them.
#[derive(Clone, Debug, PartialEq)]
struct LocoSummary {
    legs: Vec<(String, f64, f64)>,
    durations_s: [f32; 3],
    walk_speed_m_s: f64,
    run_speed_m_s: f64,
    walk_threshold_m_s: f64,
    run_threshold_m_s: f64,
}

impl CharacterPreviewSession {
    /// A session with nothing remembered.
    pub fn new() -> Self {
        Self::default()
    }

    /// What this session has rebuilt so far.
    pub fn builds(&self) -> PreviewBuilds {
        self.builds
    }

    /// Describe what [`build_character`] would produce, reusing whatever of the
    /// last answer still applies.
    ///
    /// `fit_source` is `(the decoded model, a stamp identifying it)`. See the
    /// type's docs for what the stamp is for and why `0` disables the cache.
    pub fn preview(
        &mut self,
        spec: &CharacterSpec,
        fit_source: Option<(&MeshAsset, u64)>,
    ) -> Result<CharacterPreview, CharacterError> {
        self.builds.previews += 1;
        let stamp = fit_source.map(|(_, s)| s).unwrap_or(0);
        // An unstamped model is an **unidentified** one, and two different models
        // with no stamp are indistinguishable here. Rather than hit a cache that
        // might be describing somebody else's mesh, forget: the cold path (and
        // every test that calls `preview_character`) takes this branch, which is
        // why the free function is still exactly the old behaviour.
        if fit_source.is_some() && stamp == 0 {
            self.bvh = None;
            self.loco = None;
        }
        let rig = match fit_source {
            None => inf_anim::build_template(spec.plan, &spec.params)
                .map_err(|e| CharacterError::Template(e.to_string()))?,
            Some((payload, _)) => {
                let bvh = self.bvh_for(payload, stamp)?;
                fit_rig_to_bvh(bvh, spec.plan, &spec.params)?.0
            }
        };
        let loco = self
            .loco_for(spec, &rig, fit_source.map(|(_, s)| s))?
            .clone();
        let body = match fit_source {
            Some(_) => None,
            None => Some(self.body_for(spec, &rig)),
        };
        Ok(CharacterPreview {
            joints: rig
                .skeleton
                .joints()
                .iter()
                .map(|j| PreviewJoint {
                    name: j.name.clone(),
                    parent: j.parent,
                    translation: j.local_bind.translation,
                })
                .collect(),
            sockets: rig.sockets.iter().map(|s| s.name.clone()).collect(),
            limits: rig.limits.len(),
            height_m: rig_height_m(&rig),
            legs: loco.legs,
            durations_s: loco.durations_s,
            walk_speed_m_s: loco.walk_speed_m_s,
            run_speed_m_s: loco.run_speed_m_s,
            walk_threshold_m_s: loco.walk_threshold_m_s,
            run_threshold_m_s: loco.run_threshold_m_s,
            body,
        })
    }

    /// The BVH for `stamp`, building it only when the stamp has moved.
    ///
    /// A **borrow-returning** accessor rather than a clone: an `inf_dcc::Bvh` over
    /// an author's model is the largest thing in this file, and cloning it per
    /// keystroke would replace one cost with another.
    fn bvh_for(
        &mut self,
        payload: &MeshAsset,
        stamp: u64,
    ) -> Result<&inf_dcc::Bvh, CharacterError> {
        let hit = matches!(&self.bvh, Some((s, _)) if *s == stamp && stamp != 0);
        if !hit {
            self.builds.bvh += 1;
            self.bvh = Some((stamp, fit_bvh(payload)?));
        }
        Ok(&self.bvh.as_ref().expect("just built or hit").1)
    }

    fn loco_for(
        &mut self,
        spec: &CharacterSpec,
        rig: &SkeletonAsset,
        fit: Option<u64>,
    ) -> Result<&LocoSummary, CharacterError> {
        let key: LocoKey = (spec.plan, spec.params, spec.gait, fit);
        let hit = matches!(&self.loco, Some((k, _)) if *k == key);
        if !hit {
            self.builds.locomotion += 1;
            let set = build_locomotion(spec.plan, rig, &spec.gait)
                .map_err(|e| CharacterError::Locomotion(e.to_string()))?;
            let summary = LocoSummary {
                legs: set
                    .legs
                    .iter()
                    .map(|l| (l.name.clone(), l.length_m, l.phase))
                    .collect(),
                durations_s: [set.idle.duration, set.walk.duration, set.run.duration],
                walk_speed_m_s: set.walk_speed_m_s,
                run_speed_m_s: set.run_speed_m_s,
                walk_threshold_m_s: set.walk_threshold_m_s(),
                run_threshold_m_s: set.run_threshold_m_s(),
            };
            self.loco = Some((key, summary));
        }
        Ok(&self.loco.as_ref().expect("just built or hit").1)
    }

    fn body_for(&mut self, spec: &CharacterSpec, rig: &SkeletonAsset) -> (usize, usize) {
        let key = (spec.plan, spec.params);
        if let Some((k, counts)) = &self.body {
            if *k == key {
                return *counts;
            }
        }
        self.builds.body += 1;
        let m = block_body_mesh(rig);
        let verts: usize = m.submeshes.iter().map(|s| s.vertices.len()).sum();
        let tris: usize = m.submeshes.iter().map(|s| s.triangle_count()).sum();
        self.body = Some((key, (verts, tris)));
        (verts, tris)
    }
}

/// Take back the assets a failed build had already written.
///
/// Newest first, and with `force`: the dependency edges point *into* this set —
/// the machine names the clips, the clips and the body name the rig — so a
/// reference-guarded delete would refuse the rig and leave half of a half-built
/// character behind, which is the state this exists to prevent.
///
/// Failures are ignored on purpose. This runs on a path that is *already*
/// failing, and the caller is about to report the write error that started it;
/// a second io error reported over the top of the first would replace the
/// diagnosis with its own consequence.
fn roll_back(project: &mut AssetProject, written: &[AssetId]) {
    for id in written.iter().rev() {
        let _ = project.delete(*id, true);
    }
}

/// One write, remembered — so [`roll_back`] can undo it if a later one fails.
///
/// `import` is the sidecar `import` table, and it is a parameter rather than
/// `None` because of round 3: `skeleton_binding` exists to catch a clip whose
/// rig has been re-imported under it, and it compares a hash **only assets that
/// recorded one** carry. Its single producer was the glTF importer, so the
/// engine's own generator of skeleton-bound animation — this wizard, which
/// writes three clips and a machine against a rig it has just written — was
/// invisible to it. The rig is re-writable in place (`write_asset` on the same
/// path routes to `rewrite_payload`, which keeps the GUID), so the staleness is
/// a live path and not a hypothetical one.
fn write_one<T: AssetPayload>(
    project: &mut AssetProject,
    written: &mut Vec<AssetId>,
    dir: &Path,
    name: &str,
    payload: &T,
    dependencies: Vec<AssetId>,
    import: Option<toml::Table>,
) -> Result<AssetId, CharacterError> {
    match project.write_asset(dir, name, payload, None, dependencies, import) {
        Ok(id) => {
            written.push(id);
            Ok(id)
        }
        Err(e) => {
            roll_back(project, written);
            Err(CharacterError::Write(e.to_string()))
        }
    }
}

/// Generate a character's six assets into `project` and return their ids.
///
/// Writing order is a dependency order and not a convenience: the skeleton is
/// written first because the mesh's skin binding and every clip's `skeleton`
/// field name its GUID, and the machine is written last because it names all
/// four of the others.
///
/// **All six or none of them.** Every write goes through [`write_one`], which
/// hands a failure to [`roll_back`] before returning it — see the module docs on
/// why generating everything up front is necessary and not sufficient.
pub fn build_character(
    project: &mut AssetProject,
    spec: &CharacterSpec,
) -> Result<CharacterBuild, CharacterError> {
    let name = spec.name.trim();
    if name.is_empty() {
        return Err(CharacterError::EmptyName);
    }

    // ── everything that can refuse, before anything is written ─────────────
    let source: Option<MeshAsset> = match spec.mesh {
        Some(id) => Some(load_mesh(project, id)?),
        None => None,
    };
    let (rig, fit) = rig_for(spec, source.as_ref())?;
    let set = build_locomotion(spec.plan, &rig, &spec.gait)
        .map_err(|e| CharacterError::Locomotion(e.to_string()))?;

    let mut warnings: Vec<String> = Vec::new();
    if let Some(f) = fit {
        if f.joints_inside < f.joints {
            warnings.push(format!(
                "{} of {} joints landed outside the mesh — check the proportions \
                 against the model's pose",
                f.joints - f.joints_inside,
                f.joints
            ));
        }
    }

    let dir = project
        .content_dir(CHARACTER_FOLDER)
        .map_err(|e| CharacterError::Write(e.to_string()))?;

    // Every id this call has put in the database, for `roll_back`.
    let mut written: Vec<AssetId> = Vec::new();

    // ── the rig ────────────────────────────────────────────────────────────
    let skeleton = write_one(project, &mut written, &dir, name, &rig, vec![], None)?;
    // The rig is on disk now, so its content hash exists to be recorded. Every
    // asset below whose track indices are POSITIONS in that rig's joint list
    // records it (round 3).
    let bound = crate::assets::skeleton_binding::import_table(project, Some(skeleton));
    debug_assert!(
        bound.is_some(),
        "the rig was just written, so its hash must be recordable"
    );

    // ── the body ───────────────────────────────────────────────────────────
    let (body, weights, mannequin) = match source.as_ref() {
        Some(payload) => match skinned_copy(payload, &rig, skeleton) {
            Ok((asset, summary, export)) => {
                if summary.unreached > 0 {
                    warnings.push(format!(
                        "{} vertices could not see any bone and kept the root's weights \
                         — paint them, or check that the rig is inside the mesh",
                        summary.unreached
                    ));
                }
                warnings.extend(export_advisories(&export));
                (asset, Some(summary), false)
            }
            // The rig is already on disk, and the skin solve is the one door in
            // the write section that can still refuse.
            Err(e) => {
                roll_back(project, &written);
                return Err(e);
            }
        },
        None => (block_body_mesh(&rig), None, true),
    };
    let mesh = write_one(
        project,
        &mut written,
        &dir,
        &format!("{name} Body"),
        &body,
        vec![skeleton],
        // **Not the body.** Its skin stream is index-aligned to the rig's joints
        // and it is the same staleness class — but `skeleton_binding` scans
        // `AnimClip | StateMachine` only, so recording a hash here would write a
        // key nothing reads. It is the garment/scalp shape, and it belongs with
        // the two `DEFERRED_INSTANCES` that already name it.
        None,
    )?;

    // ── the clips ──────────────────────────────────────────────────────────
    let skel_bytes = *skeleton.0.as_bytes();
    let clip = |project: &mut AssetProject,
                written: &mut Vec<AssetId>,
                suffix: &str,
                clip: &inf_anim::AnimClip|
     -> Result<AssetId, CharacterError> {
        let payload = AnimClipAsset::new(clip.clone(), Some(skel_bytes));
        write_one(
            project,
            written,
            &dir,
            &format!("{name} {suffix}"),
            &payload,
            vec![skeleton],
            bound.clone(),
        )
    };
    let idle = clip(project, &mut written, "Idle", &set.idle)?;
    let walk = clip(project, &mut written, "Walk", &set.walk)?;
    let run = clip(project, &mut written, "Run", &set.run)?;

    // ── the machine ────────────────────────────────────────────────────────
    let machine_asset = StateMachineAsset::new(
        locomotion_machine(
            &set,
            *idle.0.as_bytes(),
            *walk.0.as_bytes(),
            *run.0.as_bytes(),
        ),
        Some(skel_bytes),
    );
    let machine = write_one(
        project,
        &mut written,
        &dir,
        &format!("{name} Locomotion"),
        &machine_asset,
        vec![skeleton, idle, walk, run],
        bound,
    )?;

    Ok(CharacterBuild {
        skeleton,
        mesh,
        idle,
        walk,
        run,
        machine,
        mannequin,
        fit,
        weights,
        warnings,
    })
}

fn load_mesh(project: &AssetProject, id: AssetId) -> Result<MeshAsset, CharacterError> {
    let entry = project
        .db()
        .get(id)
        .ok_or_else(|| CharacterError::Mesh(format!("no asset {id}")))?;
    if entry.kind() != AssetKind::Mesh {
        return Err(CharacterError::Mesh(format!(
            "{} is not a mesh asset",
            entry.name
        )));
    }
    project
        .load_payload::<MeshAsset>(id)
        .map_err(|e| CharacterError::Mesh(e.to_string()))
}

/// What the writer had to do to the author's mesh, in words an author can act
/// on — empty when it had to do nothing.
///
/// [`inf_dcc::ExportReport`] is **advisory**, and its own docs say so field by
/// field: the writer counts these rather than refusing, because refusing would
/// mean an author who opened a bad file cannot save their work at all. The
/// P23.6 save path surfaces them; the wizard is the *second* caller that writes
/// a committed `.inf_mesh` out of this door, and it discarded the whole report —
/// so a NaN that was already in someone's glTF reached a shipped body with
/// nothing said about it.
///
/// Every field is mapped, including the ones that are structurally zero on this
/// path, because "which of these can happen here" is a property of today's code
/// and not of the rule: `from_mesh_asset` welds exactly and the wizard runs no
/// geometry op, so `coincident_vertices` and `reused_diagonals` are zero *by
/// construction* today and would stop being the day a cleanup op joins the
/// chain. Measured on the fixture: a well-formed mesh reports nothing at all, a
/// mesh with no UVs reports all 144 of its vertices.
fn export_advisories(r: &inf_dcc::ExportReport) -> Vec<String> {
    let mut out = Vec::new();
    if r.non_finite_written > 0 {
        out.push(format!(
            "{} vertices of the body arrived from the source mesh with a \
             non-finite position, normal or UV, and were written as zeroes — the \
             body is readable and those vertices are collapsed at the origin. \
             Fix them in the source model and re-run the wizard",
            r.non_finite_written
        ));
    }
    if r.non_unit_normals_written > 0 {
        out.push(format!(
            "{} vertices of the body carry a normal that is not unit length, from \
             the source mesh — lighting on those faces will be wrong",
            r.non_unit_normals_written
        ));
    }
    if r.fallback_tangents > 0 {
        out.push(format!(
            "{} vertices of the body took a fallback tangent because no triangle \
             around them carries a usable UV gradient — unwrap the source mesh \
             before using a normal map on it",
            r.fallback_tangents
        ));
    }
    if r.coincident_vertices > 0 || r.reused_diagonals > 0 {
        out.push(format!(
            "the body was written with {} coincident vertices and {} repeated \
             triangulation diagonals — it may not read back as the same mesh",
            r.coincident_vertices, r.reused_diagonals
        ));
    }
    if r.fan_fallbacks > 0 {
        out.push(format!(
            "{} faces of the source mesh had no ear and were fanned — they are \
             self-intersecting or collinear",
            r.fan_fallbacks
        ));
    }
    out
}

/// Bind `payload` to `rig`, solve its weights, and hand back a **new** skinned
/// mesh — the author's asset is untouched.
///
/// The [`inf_dcc::ExportReport`] comes back with it rather than being dropped:
/// see [`export_advisories`].
fn skinned_copy(
    payload: &MeshAsset,
    rig: &SkeletonAsset,
    skeleton: AssetId,
) -> Result<(MeshAsset, WeightSummary, inf_dcc::ExportReport), CharacterError> {
    let imported =
        inf_dcc::from_mesh_asset(payload).map_err(|e| CharacterError::Mesh(e.to_string()))?;
    let mut mesh = imported.mesh;
    let geo = crate::dcc::tessellate(&mesh);
    let bvh = inf_dcc::Bvh::new(crate::dcc::triangle_soup(&geo));

    // The bind comes first: the solve refuses an unbound mesh by design (a joint
    // index means nothing without a skeleton to index), and re-binding an already
    // bound mesh is the supported way a rig change lands.
    inf_dcc::ops::apply(
        &mut mesh,
        &inf_dcc::Op::BindSkin {
            skeleton: Some(*skeleton.0.as_bytes()),
            joints: rig.skeleton.len() as u32,
        },
    )
    .map_err(|e| CharacterError::Weights(e.to_string()))?;

    let (op, report) = inf_dcc::solve_heat_weights(&mesh, &bvh, &rig.skeleton)
        .map_err(|e| CharacterError::Weights(e.to_string()))?;
    if let Some(op) = op {
        inf_dcc::ops::apply(&mut mesh, &op).map_err(|e| CharacterError::Weights(e.to_string()))?;
    }
    let (asset, export) = inf_dcc::to_mesh_asset(&mesh, &inf_dcc::ExportOptions::default());
    Ok((
        asset,
        WeightSummary {
            assigned: report.assigned,
            unreached: report.unreached,
            worst_residual: report.worst_residual,
        },
        export,
    ))
}

// ── the mannequin ───────────────────────────────────────────────────────────

/// A blocky body for `rig`: one box per bone, each box **rigid** to the joint at
/// its top.
///
/// # Why rigid and not solved
///
/// A box built around a bone is *already* the answer a weight solve would
/// converge to for that box, and running a heat diffusion over geometry this
/// generator authored would be asking a solver to rediscover the thing it was
/// told. Rigid also makes the mannequin legible as a *diagnostic*: every box
/// moves with exactly one bone, so a bone that is not moving is visibly not
/// moving, which is the whole reason to draw one.
///
/// # Determinism
///
/// Joint order is the skeleton's, box corners are generated in a fixed order,
/// and the only non-arithmetic operation is `sqrt` (inside `normalize`), which
/// IEEE-754 specifies exactly. No trigonometry: the box's frame is built from a
/// cross product against a reference axis chosen by comparison, not by an angle.
pub fn block_body_mesh(rig: &SkeletonAsset) -> MeshAsset {
    let joints = rig.skeleton.joints();
    // Bind-pose globals, composed down the chain (the joint list is topological
    // by `Skeleton::new`'s own invariant).
    let mut mats: Vec<glam::Mat4> = Vec::with_capacity(joints.len());
    for j in joints {
        let local = j.local_bind.to_mat4();
        let m = match j.parent {
            Some(p) => mats[p as usize] * local,
            None => local,
        };
        mats.push(m);
    }
    let globals: Vec<glam::Vec3> = mats.iter().map(|m| m.w_axis.truncate()).collect();
    let height = rig_height_m(rig).max(1.0e-3) as f32;
    let radius_min = height * BOX_RADIUS_MIN_OF_HEIGHT as f32;
    let radius_max = height * BOX_RADIUS_MAX_OF_HEIGHT as f32;

    let mut vertices: Vec<MeshVertex> = Vec::new();
    let mut skin: Vec<VertexSkin> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for (index, joint) in joints.iter().enumerate() {
        // A bone is a joint and its PARENT: the box wraps the segment between
        // them and rides the parent, which is the bone that moves it.
        let Some(parent) = joint.parent else {
            continue;
        };
        let a = globals[parent as usize];
        let b = globals[index];
        let axis = b - a;
        let len = axis.length();
        if !(len.is_finite() && len > 1.0e-4) {
            continue;
        }
        let dir = axis / len;
        // A reference that is never parallel to `dir`, chosen by comparison so
        // no angle is involved.
        let reference = if dir.y.abs() < 0.9 {
            glam::Vec3::Y
        } else {
            glam::Vec3::X
        };
        let right = dir.cross(reference).normalize();
        let up = right.cross(dir);
        let radius = (len * BOX_RADIUS_OF_BONE as f32).clamp(radius_min, radius_max);
        let centre = (a + b) * 0.5;

        let corner =
            |base: glam::Vec3, i: f32, j: f32| base + right * (i * radius) + up * (j * radius);
        let a00 = corner(a, -1.0, -1.0);
        let a10 = corner(a, 1.0, -1.0);
        let a11 = corner(a, 1.0, 1.0);
        let a01 = corner(a, -1.0, 1.0);
        let b00 = corner(b, -1.0, -1.0);
        let b10 = corner(b, 1.0, -1.0);
        let b11 = corner(b, 1.0, 1.0);
        let b01 = corner(b, -1.0, 1.0);

        for quad in [
            [a00, a01, a11, a10],
            [b00, b10, b11, b01],
            [a00, a10, b10, b00],
            [a10, a11, b11, b10],
            [a11, a01, b01, b11],
            [a01, a00, b00, b01],
        ] {
            push_quad(&mut vertices, &mut indices, quad, centre);
        }
        // 6 faces × 4 corners, all riding the parent joint.
        let rigid = VertexSkin {
            joints: [parent, 0, 0, 0],
            weights: [1.0, 0.0, 0.0, 0.0],
        };
        skin.resize(vertices.len(), rigid);
    }

    MeshAsset::new(
        vec![SubMesh {
            name: "body".to_string(),
            vertices,
            indices,
            material_slot: None,
            skin,
        }],
        vec![],
    )
}

/// Append one quad as two triangles, **wound so its normal points away from
/// `centre`**.
///
/// The winding is decided by measurement rather than by getting the corner order
/// right six times: the normal is taken from the winding, and if it faces the box
/// interior the quad is reversed. A face wound inward is invisible under backface
/// culling and correct under lighting, which is exactly the kind of defect that
/// survives review.
fn push_quad(
    vertices: &mut Vec<MeshVertex>,
    indices: &mut Vec<u32>,
    quad: [glam::Vec3; 4],
    centre: glam::Vec3,
) {
    let mut q = quad;
    let mut normal = (q[1] - q[0]).cross(q[2] - q[0]);
    let centroid = (q[0] + q[1] + q[2] + q[3]) * 0.25;
    if normal.dot(centroid - centre) < 0.0 {
        q.reverse();
        normal = (q[1] - q[0]).cross(q[2] - q[0]);
    }
    let normal = if normal.length_squared() > 0.0 {
        normal.normalize()
    } else {
        glam::Vec3::Y
    };
    let base = vertices.len() as u32;
    for (k, p) in q.iter().enumerate() {
        vertices.push(MeshVertex {
            position: p.to_array(),
            normal: normal.to_array(),
            uv: [(k == 1 || k == 2) as u8 as f32, (k >= 2) as u8 as f32],
            ..Default::default()
        });
    }
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn biped() -> SkeletonAsset {
        inf_anim::build_template(BodyPlan::Biped, &BodyParams::default()).unwrap()
    }

    fn project(dir: &std::path::Path) -> AssetProject {
        AssetProject::open(dir.join("Content")).expect("open project")
    }

    /// **The warm path answers the same preview** — the first thing a cache has
    /// to prove, before anything about how fast it is.
    #[test]
    fn a_warm_session_answers_exactly_what_the_cold_path_does() {
        let mut session = CharacterPreviewSession::new();
        for h in [0.9f64, 1.75, 1.75, 2.6] {
            for cadence in [1.0f64, 1.6] {
                let spec = CharacterSpec {
                    params: BodyParams {
                        height_m: h,
                        ..BodyParams::default()
                    },
                    gait: inf_anim::locomotion::GaitParams {
                        walk_cadence_hz: cadence,
                        ..Default::default()
                    },
                    ..CharacterSpec::default()
                };
                assert_eq!(
                    session.preview(&spec, None).unwrap(),
                    preview_character(&spec, None).unwrap(),
                    "the warm preview disagreed with the cold one at {h} m / {cadence} Hz"
                );
            }
        }
        // …and the fitted path, where the BVH is the thing being remembered.
        let mesh = tube_man();
        let mut fitted = CharacterPreviewSession::new();
        for h in [1.6f64, 1.8, 1.8] {
            let spec = CharacterSpec {
                params: BodyParams {
                    height_m: h,
                    ..BodyParams::default()
                },
                mesh: Some(AssetId::new()),
                ..CharacterSpec::default()
            };
            assert_eq!(
                fitted.preview(&spec, Some((&mesh, 0xABCD))).unwrap(),
                preview_character(&spec, Some(&mesh)).unwrap(),
                "the warm fitted preview disagreed at {h} m"
            );
        }
    }

    /// **The warm path is warm, counted rather than claimed** (P29.5, ROADMAP
    /// §13's F5 item).
    ///
    /// Counters and not a clock, because a clock measures this machine and a
    /// counter measures the mechanism. The panel re-previews on every edit, so
    /// the numbers below are what one wizard session costs: dragging a
    /// **proportion** slider must not rebuild the model's BVH, and dragging a
    /// **gait** slider must not rebuild either the BVH or the body mesh.
    ///
    /// The control is the cold path, whose counters are one-per-preview by
    /// construction — a session that stopped hitting would keep every number in
    /// the preview correct and fail exactly here.
    #[test]
    fn a_warm_session_rebuilds_only_what_the_edit_changed() {
        let mesh = tube_man();
        let stamp = 0x5EED_u64;
        let mut s = CharacterPreviewSession::new();

        // Twenty proportion edits — one BVH, twenty fits.
        for k in 0..20 {
            let spec = CharacterSpec {
                params: BodyParams {
                    height_m: 1.5 + 0.01 * k as f64,
                    ..BodyParams::default()
                },
                mesh: Some(AssetId::new()),
                ..CharacterSpec::default()
            };
            s.preview(&spec, Some((&mesh, stamp))).unwrap();
        }
        let after_proportions = s.builds();
        assert_eq!(after_proportions.previews, 20);
        assert_eq!(
            after_proportions.bvh, 1,
            "a proportion drag rebuilt the model's BVH: {after_proportions:?}"
        );
        assert_eq!(
            after_proportions.locomotion, 20,
            "a proportion really does change the rig, so the clips really are regenerated"
        );

        // Twenty gait edits at one fixed proportion — still one BVH, and now the
        // rig has stopped moving too, so nothing but the clips is regenerated.
        let base = BodyParams {
            height_m: 1.75,
            ..BodyParams::default()
        };
        for k in 0..20 {
            let spec = CharacterSpec {
                params: base,
                gait: inf_anim::locomotion::GaitParams {
                    walk_cadence_hz: 1.0 + 0.01 * k as f64,
                    ..Default::default()
                },
                mesh: Some(AssetId::new()),
                ..CharacterSpec::default()
            };
            s.preview(&spec, Some((&mesh, stamp))).unwrap();
        }
        let after = s.builds();
        assert_eq!(after.previews, 40);
        assert_eq!(after.bvh, 1, "forty previews, one BVH: {after:?}");

        // Re-previewing the SAME spec costs nothing at all.
        let spec = CharacterSpec {
            params: base,
            mesh: Some(AssetId::new()),
            ..CharacterSpec::default()
        };
        s.preview(&spec, Some((&mesh, stamp))).unwrap();
        let before_repeat = s.builds();
        s.preview(&spec, Some((&mesh, stamp))).unwrap();
        let repeated = s.builds();
        assert_eq!(repeated.bvh, before_repeat.bvh);
        assert_eq!(
            repeated.locomotion, before_repeat.locomotion,
            "an identical re-preview regenerated the clips: {repeated:?}"
        );

        // The mannequin path: a gait drag must not rebuild the body mesh.
        let mut m = CharacterPreviewSession::new();
        for k in 0..20 {
            let spec = CharacterSpec {
                params: base,
                gait: inf_anim::locomotion::GaitParams {
                    walk_cadence_hz: 1.0 + 0.01 * k as f64,
                    ..Default::default()
                },
                ..CharacterSpec::default()
            };
            m.preview(&spec, None).unwrap();
        }
        assert_eq!(
            m.builds().body,
            1,
            "a gait drag rebuilt the mannequin body: {:?}",
            m.builds()
        );

        // **The control.** A fresh session per preview — which is exactly what
        // the panel did before this existed — rebuilds everything, every time.
        let mut cold = PreviewBuilds::default();
        for k in 0..20 {
            let spec = CharacterSpec {
                params: BodyParams {
                    height_m: 1.5 + 0.01 * k as f64,
                    ..BodyParams::default()
                },
                mesh: Some(AssetId::new()),
                ..CharacterSpec::default()
            };
            let mut once = CharacterPreviewSession::new();
            once.preview(&spec, Some((&mesh, stamp))).unwrap();
            let b = once.builds();
            cold.bvh += b.bvh;
            cold.locomotion += b.locomotion;
            cold.previews += b.previews;
        }
        assert_eq!(
            cold.bvh, 20,
            "the control must pay for a BVH every keystroke: {cold:?}"
        );
    }

    /// A model with **no stamp** is not cached, because two unidentified models
    /// are indistinguishable — and the free `preview_character` is that path, so
    /// it is exactly its pre-P29.5 self.
    #[test]
    fn an_unstamped_model_is_never_cached() {
        let a = tube_man();
        let mut s = CharacterPreviewSession::new();
        let spec = CharacterSpec {
            mesh: Some(AssetId::new()),
            ..CharacterSpec::default()
        };
        s.preview(&spec, Some((&a, 0))).unwrap();
        s.preview(&spec, Some((&a, 0))).unwrap();
        assert_eq!(
            s.builds().bvh,
            2,
            "an unstamped model must be rebuilt: {:?}",
            s.builds()
        );
        // A stamp that MOVES also rebuilds — the case that matters when the
        // author edits the model in the Model Editor while the wizard is open.
        let mut t = CharacterPreviewSession::new();
        t.preview(&spec, Some((&a, 1))).unwrap();
        t.preview(&spec, Some((&a, 1))).unwrap();
        t.preview(&spec, Some((&a, 2))).unwrap();
        assert_eq!(t.builds().bvh, 2, "{:?}", t.builds());

        // **And an unstamped FITTED preview must not seed the key a MANNEQUIN
        // preview then hits** (P29.5 audit, A8). The two rigs come from
        // different producers -- `fit_template` against a BVH, `build_template`
        // against nothing -- so a cache that could not tell them apart would
        // answer a mannequin with the author's model's leg lengths.
        let spec = CharacterSpec {
            mesh: Some(AssetId::new()),
            ..CharacterSpec::default()
        };
        let bare = CharacterSpec {
            mesh: None,
            ..CharacterSpec::default()
        };
        let mut u = CharacterPreviewSession::new();
        u.preview(&spec, Some((&a, 0))).unwrap();
        let mixed = u.preview(&bare, None).unwrap();
        assert_eq!(
            mixed,
            preview_character(&bare, None).unwrap(),
            "a mannequin preview answered with the fitted model's locomotion"
        );
    }

    #[test]
    fn a_preview_describes_the_rig_the_build_would_make() {
        let spec = CharacterSpec::default();
        let p = preview_character(&spec, None).unwrap();
        assert_eq!(p.joints.len(), biped().skeleton.len());
        assert_eq!(p.joints[0].name, "hips");
        assert_eq!(p.legs.len(), 2);
        assert!(p.limits >= 4, "knees and elbows carry hinges");
        assert!(p.walk_threshold_m_s < p.run_threshold_m_s);
        let (verts, tris) = p.body.expect("a mannequin is previewed");
        assert!(verts > 100 && tris > 50, "{verts} verts / {tris} tris");
    }

    /// **The previewed height is the JOINT SPAN, and it is not what was asked
    /// for** (audit F4).
    ///
    /// The first version of this asserted `(height_m - 1.75).abs() < 0.2`, which
    /// is a tolerance wide enough to hide the whole relationship: the answer is
    /// systematically 7 % low, because the topmost joint of a template rig is
    /// `head` at `head_height_ratio × height_m` and the skull above it carries no
    /// joint. Pinned as that identity, on three heights, so the number is
    /// explained rather than approximated — and so a generator that started
    /// emitting a joint above the head fails here instead of drifting inside a
    /// band.
    #[test]
    fn the_previewed_height_is_the_joint_span_and_not_the_requested_height() {
        for h in [0.9, 1.75, 2.6] {
            let spec = CharacterSpec {
                params: BodyParams {
                    height_m: h,
                    ..BodyParams::default()
                },
                ..CharacterSpec::default()
            };
            let p = preview_character(&spec, None).unwrap();
            let want = h * BodyParams::default().head_height_ratio;
            assert!(
                (p.height_m - want).abs() < 1.0e-4,
                "a {h} m spec previews {} m; the head joint sits at {want} m",
                p.height_m
            );
            assert!(
                p.height_m < h,
                "the joint span cannot reach the standing height"
            );
        }
    }

    /// **A proportion reaches the preview.** The "shape it" step is only a step
    /// if moving a slider moves the answer.
    #[test]
    fn shaping_the_body_changes_what_the_preview_reports() {
        let short = preview_character(&CharacterSpec::default(), None).unwrap();
        let tall = preview_character(
            &CharacterSpec {
                params: BodyParams {
                    height_m: 2.6,
                    ..BodyParams::default()
                },
                ..CharacterSpec::default()
            },
            None,
        )
        .unwrap();
        assert!(tall.height_m > short.height_m * 1.3);
        assert!(tall.walk_speed_m_s > short.walk_speed_m_s);
        assert_ne!(tall.joints, short.joints);
    }

    #[test]
    fn the_mannequin_rides_real_bones_and_has_no_degenerate_triangle() {
        let rig = biped();
        let body = block_body_mesh(&rig);
        let sub = &body.submeshes[0];
        assert_eq!(
            sub.skin.len(),
            sub.vertices.len(),
            "every vertex is skinned"
        );
        // Every box rides ONE joint at full weight, and never joint 0 by
        // accident: a mannequin whose every vertex named the root would animate
        // as a single rigid lump and look almost right.
        let mut used: std::collections::BTreeSet<u16> = Default::default();
        for w in &sub.skin {
            assert_eq!(w.weights, [1.0, 0.0, 0.0, 0.0]);
            used.insert(w.joints[0]);
        }
        assert!(
            used.len() > 8,
            "the mannequin rides only {} joints of {}",
            used.len(),
            rig.skeleton.len()
        );
        // No zero-area triangle: a box built on a degenerate frame would still
        // count vertices and draw nothing.
        for tri in sub.indices.chunks_exact(3) {
            let p = |i: u32| glam::Vec3::from_array(sub.vertices[i as usize].position);
            let area = (p(tri[1]) - p(tri[0]))
                .cross(p(tri[2]) - p(tri[0]))
                .length();
            assert!(area > 1.0e-9, "degenerate triangle at {tri:?}");
        }
        // Deterministic to the byte.
        assert_eq!(
            inf_asset::encode(&body).unwrap(),
            inf_asset::encode(&block_body_mesh(&rig)).unwrap()
        );
    }

    /// **Every mannequin face points OUT.** Wound by measurement, checked by
    /// measurement — a box whose faces face inward is invisible under culling.
    #[test]
    fn the_mannequin_faces_point_away_from_their_own_bone() {
        let body = block_body_mesh(&biped());
        let sub = &body.submeshes[0];
        for tri in sub.indices.chunks_exact(3) {
            let v = |i: u32| &sub.vertices[i as usize];
            let p = |i: u32| glam::Vec3::from_array(v(i).position);
            let geometric = (p(tri[1]) - p(tri[0]))
                .cross(p(tri[2]) - p(tri[0]))
                .normalize();
            let stored = glam::Vec3::from_array(v(tri[0]).normal);
            assert!(
                geometric.dot(stored) > 0.99,
                "the stored normal disagrees with the winding: {geometric} vs {stored}"
            );
        }
    }

    #[test]
    fn a_build_writes_six_assets_and_wires_them_together() {
        let tmp = tempfile::tempdir().unwrap();
        let mut p = project(tmp.path());
        let out = build_character(&mut p, &CharacterSpec::default()).expect("builds");
        assert!(out.mannequin);
        assert!(out.fit.is_none() && out.weights.is_none());

        // Every id resolves, at the kind it claims.
        for (id, kind) in [
            (out.skeleton, AssetKind::Skeleton),
            (out.mesh, AssetKind::Mesh),
            (out.idle, AssetKind::AnimClip),
            (out.walk, AssetKind::AnimClip),
            (out.run, AssetKind::AnimClip),
            (out.machine, AssetKind::StateMachine),
        ] {
            assert_eq!(p.db().get(id).expect("registered").kind(), kind);
        }

        // The machine names the three clips that were just written — the wiring
        // the whole wizard exists to do, read back off the payload rather than
        // assumed from the call.
        let sm: StateMachineAsset = p.load_payload(out.machine).unwrap();
        assert_eq!(sm.skeleton, Some(*out.skeleton.0.as_bytes()));
        let refs: Vec<Uuid> = sm
            .machine
            .states
            .iter()
            .map(|s| match &s.motion {
                inf_anim::Motion::Clip(c) => Uuid::from_bytes(*c),
                _ => panic!("a generated state plays a single clip"),
            })
            .collect();
        assert_eq!(refs, vec![out.idle.0, out.walk.0, out.run.0]);

        // …and the dependency edges are real, so a delete warns and a cook ships
        // them.
        assert!(p.db().referenced_by(out.idle).contains(&out.machine));
        assert!(p.db().referenced_by(out.skeleton).contains(&out.mesh));
    }

    #[test]
    fn a_quadruped_builds_too_and_differs_from_the_biped() {
        let tmp = tempfile::tempdir().unwrap();
        let mut p = project(tmp.path());
        let biped = build_character(&mut p, &CharacterSpec::default()).unwrap();
        let quad = build_character(
            &mut p,
            &CharacterSpec {
                name: "Beast".into(),
                plan: BodyPlan::Quadruped,
                ..CharacterSpec::default()
            },
        )
        .unwrap();
        let a: SkeletonAsset = p.load_payload(biped.skeleton).unwrap();
        let b: SkeletonAsset = p.load_payload(quad.skeleton).unwrap();
        let feet = |s: &SkeletonAsset| {
            s.skeleton
                .joints()
                .iter()
                .filter(|j| j.name.starts_with("foot_"))
                .count()
        };
        assert_eq!((feet(&a), feet(&b)), (2, 4), "two legs against four");
        // …and the quadruped has NO arms, which is the other half of the plan.
        assert!(a.skeleton.index_of("hand_l").is_some());
        assert!(b.skeleton.index_of("hand_l").is_none());
        let aw: AnimClipAsset = p.load_payload(biped.walk).unwrap();
        let bw: AnimClipAsset = p.load_payload(quad.walk).unwrap();
        assert_ne!(aw.clip, bw.clip, "two plans, two walks");
    }

    /// **The last row of the table**: the built assets become an actor that is
    /// wearing them, in one undo step.
    #[test]
    fn the_built_character_becomes_one_undoable_actor() {
        use crate::scene::SceneDoc;
        use inf_ecs::components::{AnimStateMachine, SkeletalMesh};

        let tmp = tempfile::tempdir().unwrap();
        let mut p = project(tmp.path());
        let out = build_character(&mut p, &CharacterSpec::default()).unwrap();

        let mut doc = SceneDoc::new();
        let before = doc.order().len();
        let guid = doc.edit_create_character(
            "Hero",
            out.skeleton.0,
            out.mesh.0,
            out.machine.0,
            glam::DVec3::new(1.0, 0.0, -2.0),
        );
        let e = doc.world().entity_of(guid).expect("spawned");
        let sk = doc.world().world().get::<SkeletalMesh>(e).expect("rigged");
        assert_eq!(
            (sk.skeleton, sk.mesh),
            (Some(out.skeleton.0), Some(out.mesh.0))
        );
        assert_eq!(
            doc.world()
                .world()
                .get::<AnimStateMachine>(e)
                .expect("machine assigned")
                .sm,
            Some(out.machine.0)
        );

        // ONE step, and it takes the components with it — a wizard that spawned
        // and then assigned would leave a rigless entity behind on undo.
        doc.undo();
        assert_eq!(doc.order().len(), before);
        assert!(doc.world().entity_of(guid).is_none());
        doc.redo();
        let e = doc.world().entity_of(guid).expect("redone");
        assert_eq!(
            doc.world()
                .world()
                .get::<AnimStateMachine>(e)
                .expect("the machine came back with it")
                .sm,
            Some(out.machine.0)
        );
    }

    /// A 1.8 m "tube man" — the fixture `inf_dcc::autofit`'s own gate uses,
    /// rebuilt as a `MeshAsset` because that is what an author's imported model
    /// is and what the wizard is handed.
    fn tube_man() -> MeshAsset {
        let mut vertices: Vec<MeshVertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut part = |min: [f32; 3], max: [f32; 3]| {
            let lo = glam::Vec3::from_array(min);
            let hi = glam::Vec3::from_array(max);
            let c = (lo + hi) * 0.5;
            let p = |x: bool, y: bool, z: bool| {
                glam::Vec3::new(
                    if x { hi.x } else { lo.x },
                    if y { hi.y } else { lo.y },
                    if z { hi.z } else { lo.z },
                )
            };
            for quad in [
                [
                    p(false, false, false),
                    p(true, false, false),
                    p(true, true, false),
                    p(false, true, false),
                ],
                [
                    p(false, false, true),
                    p(true, false, true),
                    p(true, true, true),
                    p(false, true, true),
                ],
                [
                    p(false, false, false),
                    p(false, true, false),
                    p(false, true, true),
                    p(false, false, true),
                ],
                [
                    p(true, false, false),
                    p(true, true, false),
                    p(true, true, true),
                    p(true, false, true),
                ],
                [
                    p(false, false, false),
                    p(true, false, false),
                    p(true, false, true),
                    p(false, false, true),
                ],
                [
                    p(false, true, false),
                    p(true, true, false),
                    p(true, true, true),
                    p(false, true, true),
                ],
            ] {
                push_quad(&mut vertices, &mut indices, quad, c);
            }
        };
        part([-0.20, 0.90, -0.12], [0.20, 1.50, 0.12]); // torso
        part([-0.10, 1.48, -0.10], [0.10, 1.80, 0.10]); // head + neck
        part([-0.30, 0.66, -0.08], [-0.16, 1.48, 0.08]); // left arm
        part([0.16, 0.66, -0.08], [0.30, 1.48, 0.08]); // right arm
        part([-0.18, 0.0, -0.09], [-0.02, 0.98, 0.09]); // left leg
        part([0.02, 0.0, -0.09], [0.18, 0.98, 0.09]); // right leg
        MeshAsset::new(
            vec![SubMesh {
                name: "tube".into(),
                vertices,
                indices,
                material_slot: None,
                skin: vec![],
            }],
            vec![],
        )
    }

    /// **Round 3: the wizard's own output was invisible to R2.C's advisory.**
    ///
    /// `skeleton_binding` compares a recorded rig hash against the rig an
    /// animation asset points at, and it can only compare a hash somebody
    /// wrote. Its one producer was the glTF importer — so the engine's own
    /// generator of skeleton-bound animation wrote three clips and a machine
    /// with an empty `import` table, and re-generating the rig into the same
    /// GUID renumbered every track index with nothing anywhere saying so.
    ///
    /// Asserted as the WORLD (the advisory fires) and not as "a table was
    /// passed": the healthy control has to be silent first, or the second half
    /// proves nothing.
    #[test]
    fn a_wizard_character_notices_its_rig_being_regenerated_under_it() {
        let tmp = tempfile::tempdir().unwrap();
        let mut p = project(tmp.path());
        let out = build_character(&mut p, &CharacterSpec::default()).expect("a mannequin builds");

        assert!(
            crate::assets::skeleton_binding::advisories(&p).is_empty(),
            "a freshly generated character raised an advisory about itself"
        );

        // The rig is regenerated with DIFFERENT proportions into the same GUID
        // — the wizard's own re-run, and what `rewrite_payload` is for. The
        // joint list changes shape, so every clip baked against the old one is
        // now indexed against a different rig.
        let taller = inf_anim::build_template(
            BodyPlan::Biped,
            &BodyParams {
                height_m: 2.4,
                ..BodyParams::default()
            },
        )
        .unwrap();
        p.rewrite_payload(out.skeleton, &taller, vec![])
            .expect("the rig re-generates in place");

        let found = crate::assets::skeleton_binding::advisories(&p);
        assert_eq!(
            found.len(),
            4,
            "the three clips and the machine must each say so, got {found:?}"
        );
        assert!(
            found.iter().all(|a| a.contains("POSITIONS")),
            "every advisory must name the mechanism: {found:?}"
        );
        // The BODY is not in the set: its skin stream is index-aligned to the
        // same joints, and that instance is the deferred one — recorded in
        // `DEFERRED_INSTANCES`, not silently half-closed here.
        assert!(
            !found.iter().any(|a| a.contains("Body")),
            "the body mesh is the deferred instance, not a fifth advisory: {found:?}"
        );
    }

    /// **The imported-mesh path, end to end**: the rig is fitted to the model,
    /// the model is bound and solved, and the author's own asset is left alone.
    #[test]
    fn a_supplied_mesh_is_fitted_skinned_and_never_rewritten() {
        let tmp = tempfile::tempdir().unwrap();
        let mut p = project(tmp.path());
        let source = tube_man();
        let dir = p.content_dir("Meshes").unwrap();
        let mesh_id = p
            .write_asset(&dir, "TubeMan", &source, None, vec![], None)
            .unwrap();

        let out = build_character(
            &mut p,
            &CharacterSpec {
                name: "Fitted".into(),
                mesh: Some(mesh_id),
                ..CharacterSpec::default()
            },
        )
        .expect("the tube man is a character");

        assert!(!out.mannequin);
        let fit = out.fit.expect("a fit ran");
        assert!(
            (fit.height_m - 1.80).abs() < 0.01,
            "the rig took the MESH's height, not the spec's: {fit:?}"
        );
        let weights = out.weights.expect("a solve ran");
        assert!(
            weights.assigned > 0,
            "the solve wrote no weights: {weights:?}"
        );

        // The written body is a DIFFERENT asset from the author's, and it is
        // skinned where the author's was rigid.
        assert_ne!(out.mesh, mesh_id);
        let untouched: MeshAsset = p.load_payload(mesh_id).unwrap();
        assert_eq!(untouched, source, "the author's own mesh was rewritten");
        let skinned: MeshAsset = p.load_payload(out.mesh).unwrap();
        assert!(skinned.submeshes.iter().all(|s| s.is_skinned()));
        // …and the weights name more than one bone, which is what says a solve
        // happened rather than a bind (a bare bind leaves every vertex rigid to
        // joint 0 and would satisfy `is_skinned` perfectly).
        let named: std::collections::BTreeSet<u16> = skinned
            .submeshes
            .iter()
            .flat_map(|s| s.skin.iter())
            .flat_map(|w| {
                w.joints
                    .iter()
                    .zip(w.weights.iter())
                    .filter(|(_, weight)| **weight > 0.0)
                    .map(|(j, _)| *j)
                    .collect::<Vec<_>>()
            })
            .collect();
        assert!(
            named.len() > 4,
            "the skinned copy rides {} joints — a bare bind, not a solve",
            named.len()
        );
    }

    /// **The writer's advisories reach the author** (audit F3).
    ///
    /// A source mesh with no usable UVs writes a body whose every tangent is the
    /// `[1,0,0,1]` fallback. That is not a refusal — the character is fine to
    /// look at and wrong to put a normal map on — so it is a warning, and the
    /// whole `ExportReport` used to be dropped on the floor at `let (asset,
    /// _export) = to_mesh_asset(..)`.
    ///
    /// The well-formed half is the anti-vacuity: a check that fires on every
    /// mesh is not a check, and would train an author to ignore the panel's
    /// Warnings section.
    #[test]
    fn a_source_mesh_with_no_uvs_says_so_and_a_good_one_does_not() {
        let tmp = tempfile::tempdir().unwrap();
        let mut p = project(tmp.path());
        let dir = p.content_dir("Meshes").unwrap();

        let good = tube_man();
        let mut bare = good.clone();
        for v in &mut bare.submeshes[0].vertices {
            v.uv = [0.0, 0.0];
        }
        let good_id = p
            .write_asset(&dir, "Good", &good, None, vec![], None)
            .unwrap();
        let bare_id = p
            .write_asset(&dir, "Bare", &bare, None, vec![], None)
            .unwrap();

        let build = |p: &mut AssetProject, name: &str, mesh: AssetId| {
            build_character(
                p,
                &CharacterSpec {
                    name: name.into(),
                    mesh: Some(mesh),
                    ..CharacterSpec::default()
                },
            )
            .expect("builds")
        };

        let bad = build(&mut p, "Bare", bare_id);
        assert!(
            bad.warnings.iter().any(|w| w.contains("fallback tangent")),
            "a body with no usable UVs said nothing: {:?}",
            bad.warnings
        );

        let ok = build(&mut p, "Good", good_id);
        assert!(
            !ok.warnings.iter().any(|w| w.contains("fallback tangent")),
            "a well-formed mesh was warned about anyway: {:?}",
            ok.warnings
        );
    }

    /// Every field of the report maps to a sentence, checked on a value rather
    /// than on whatever the fixture happens to produce — `non_finite_written`
    /// needs a NaN in someone's glTF, and manufacturing one to prove a `format!`
    /// runs would be a worse test than reading the mapping directly.
    #[test]
    fn every_export_advisory_has_words() {
        assert!(export_advisories(&inf_dcc::ExportReport::default()).is_empty());
        let all = inf_dcc::ExportReport {
            non_finite_written: 1,
            non_unit_normals_written: 2,
            fallback_tangents: 3,
            coincident_vertices: 4,
            reused_diagonals: 5,
            fan_fallbacks: 6,
            ..inf_dcc::ExportReport::default()
        };
        let out = export_advisories(&all);
        assert_eq!(out.len(), 5, "{out:?}");
        for (n, needle) in [
            (1, "non-finite"),
            (2, "not unit length"),
            (3, "fallback tangent"),
            (4, "coincident"),
            (6, "fanned"),
        ] {
            assert!(
                out.iter()
                    .any(|w| w.contains(needle) && w.contains(&n.to_string())),
                "no advisory names `{needle}` with its count: {out:?}"
            );
        }
    }

    #[test]
    fn an_empty_name_is_refused_before_anything_is_written() {
        let tmp = tempfile::tempdir().unwrap();
        let mut p = project(tmp.path());
        assert_eq!(
            build_character(
                &mut p,
                &CharacterSpec {
                    name: "   ".into(),
                    ..CharacterSpec::default()
                }
            ),
            Err(CharacterError::EmptyName)
        );
        assert_eq!(p.db().len(), 0, "a refusal left assets behind");
    }

    /// **A TORN BUILD, checked against the filesystem.**
    ///
    /// The pre-write refusals above cannot see this: they leave the content root
    /// clean because nothing has been written *yet*. The case that matters is the
    /// one where five assets are already on disk and the sixth cannot be —
    /// induced here by putting a **directory** where the walk clip's sidecar has
    /// to go, so the payload lands and `AssetSidecar::save` fails.
    ///
    /// Measured before it was fixed: the build left **three** registered assets
    /// and a fourth payload with **no sidecar** — which the content watcher
    /// promotes under a freshly-minted GUID, so the author is left with orphans
    /// they cannot tell from their own work. The verdict is asserted against
    /// `read_dir`, not against the build's return value, because a build that
    /// *said* it had cleaned up was exactly what the first version did.
    #[test]
    fn a_write_that_fails_halfway_takes_back_what_it_wrote() {
        let tmp = tempfile::tempdir().unwrap();
        let mut p = project(tmp.path());
        let dir = p.content_dir(CHARACTER_FOLDER).unwrap();
        let blocker = dir.join("Hero_Walk.inf_anim.toml");
        std::fs::create_dir_all(&blocker).unwrap();

        let err = build_character(
            &mut p,
            &CharacterSpec {
                name: "Hero".into(),
                ..CharacterSpec::default()
            },
        )
        .expect_err("the blocked sidecar refuses the walk clip");
        assert!(matches!(err, CharacterError::Write(_)), "{err:?}");

        assert_eq!(p.db().len(), 0, "a torn build left assets registered");
        let left: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
            .filter(|n| n != "Hero_Walk.inf_anim.toml")
            .collect();
        assert!(
            left.is_empty(),
            "a torn build left files behind: {left:?} — an orphaned payload is \
             re-registered by the watcher under a GUID nothing references"
        );

        // …and the folder still works: the same name builds cleanly once the
        // blocker is gone, which is what says the rollback removed files rather
        // than leaving the directory in a state the next write trips over.
        std::fs::remove_dir(&blocker).unwrap();
        let out = build_character(
            &mut p,
            &CharacterSpec {
                name: "Hero".into(),
                ..CharacterSpec::default()
            },
        )
        .expect("the second build succeeds");
        assert_eq!(p.db().len(), 6);
        let sm: StateMachineAsset = p.load_payload(out.machine).unwrap();
        assert_eq!(sm.machine.states.len(), 3);
    }

    /// A refusal from a door *inside* the build leaves the content root clean —
    /// the rule the module docs claim, driven through a gait the generator
    /// rejects rather than through the name check above (which refuses before it
    /// has done anything at all).
    #[test]
    fn a_refused_gait_writes_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        let mut p = project(tmp.path());
        let err = build_character(
            &mut p,
            &CharacterSpec {
                gait: GaitParams {
                    walk_cadence_hz: 0.0,
                    ..GaitParams::default()
                },
                ..CharacterSpec::default()
            },
        )
        .expect_err("a zero cadence is refused");
        assert!(matches!(err, CharacterError::Locomotion(_)), "{err:?}");
        assert_eq!(p.db().len(), 0);
    }
}
