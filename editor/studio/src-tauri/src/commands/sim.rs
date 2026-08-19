//! Simulate commands (P8.4): drive the in-editor [`SimSession`] over the shared
//! [`SceneDoc`] so a Blueprint + 2D-physics scene plays in the viewport.
//!
//! The session lives here behind a `Mutex`; it borrows the **same** `SceneDoc`
//! the viewport renders (`SceneState::doc`), so ticking mutates the live world
//! and the native viewport re-syncs on the bumped version. Enter snapshots the
//! world; Stop restores it exactly.
//!
//! Input path (P8.4.3): `sim_tick` receives the currently-held keys/actions from
//! the frontend each frame (a rAF loop) — the "frontend-focused" input route that
//! always works. When the **native** viewport holds OS focus it forwards
//! unconsumed keys over `viewport://key` (the Phase-2 focus-handoff channel); the
//! frontend merges those into the held set it passes here. Driving the tick from
//! a real frame timer is the human-verified remainder (CI scripts the session
//! directly — see `inf-editor-core`'s `simulate_platformer` test).

use std::sync::Mutex;
use std::time::Instant;

use inf_blueprint::{InterpDebug, LocalId};
use inf_editor_core::samples::bound_actors;
use inf_editor_core::simulate::{SimDebugHit, SimInput, SimSession, SIM_HZ};
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, State};

use super::assets::AssetState;
use super::scene::SceneState;

/// Live Simulate state: the session (absent when stopped) + the last tick's
/// wall-clock instant (for the frame delta fed to the fixed-step accumulator).
#[derive(Default)]
pub struct SimState {
    inner: Mutex<SimInner>,
}

impl SimState {
    /// Whether a Simulate session is live — other command modules (e.g. the
    /// sequencer scrub, P11.4) consult this to avoid two writers fighting over the
    /// shared `SceneDoc`.
    pub fn is_running(&self) -> bool {
        self.inner
            .lock()
            .map(|inner| inner.session.is_some())
            .unwrap_or(false)
    }

    /// **Accumulate a mouse-look delta from the native viewport** (P29.6).
    ///
    /// Raw device counts, drained by the next `sim_tick`. Dropped when no
    /// session is running — a capture can outlive its `sim_stop` by a frame, and
    /// a stale delta arriving into the next session would be a mouse jump nobody
    /// made. A poisoned lock is ignored for the same reason the mixer path
    /// ignores one: losing a frame of mouse-look is not worth taking a session
    /// down.
    pub fn push_look(&self, dx: f32, dy: f32) {
        if let Ok(mut inner) = self.inner.lock() {
            if inner.session.is_some() {
                inner.look.0 += dx;
                inner.look.1 += dy;
            }
        }
    }

    /// Live-apply a mixer to a running Simulate session (E-P9). A no-op when
    /// stopped. Mirrors the mixer load in [`sim_start`] so the Audio Mixer panel's
    /// save is heard immediately in an active Simulate. Poisoned lock is ignored
    /// (the next `sim_start` reloads the mixer from disk regardless).
    pub fn apply_mixer(&self, mixer: inf_audio::MixerConfig) {
        if let Ok(mut inner) = self.inner.lock() {
            if let Some(session) = inner.session.as_mut() {
                session.set_audio_mixer(mixer);
            }
        }
    }
}

struct SimInner {
    session: Option<SimSession>,
    last: Option<Instant>,
    /// **The editor's own resolved input** (P29.6).
    ///
    /// The frontend sends raw `KeyboardEvent.code` values now; this turns them
    /// into the engine's actions and axes through
    /// [`inf_input::default_map`] — the SAME table the shipped player reads, so
    /// a control cannot mean one thing in Simulate and another in a build. It
    /// used to be a set of *action names* mapped in TypeScript, which was a third
    /// copy of the binding table across a language boundary (the campaign's Wave
    /// I defect) and could not carry an analog axis at all.
    input: inf_input::InputState,
    /// Raw mouse-delta counts accumulated from the native viewport since the last
    /// tick, in device units. The viewport captures the pointer while Simulate is
    /// running (the airspace rule: the mouse over the viewport belongs to the
    /// native child window, not to the webview), and this is where its deltas
    /// wait for the next frame.
    look: (f32, f32),
}

impl Default for SimInner {
    fn default() -> Self {
        Self {
            session: None,
            last: None,
            input: inf_input::InputState::new(inf_input::default_map()),
            look: (0.0, 0.0),
        }
    }
}

/// Enter Simulate over the current scene: snapshot the world, bind actors —
/// **preferring the scene's persisted `ActorClass` links** (P9.5), resolving each
/// via the project asset DB, and falling back to the `CharacterController2D`
/// heuristic for scenes with no bindings — and fire BeginPlay.
#[tauri::command]
pub async fn sim_start(
    app: AppHandle,
    scene: State<'_, SceneState>,
    sim: State<'_, SimState>,
    assets: State<'_, AssetState>,
) -> Result<(), String> {
    let mut doc = scene.doc.lock().map_err(|_| "scene lock poisoned")?;
    let actors = bound_actors(&doc, |guid| {
        assets.load_blueprint_class(inf_asset::AssetId(guid))
    });
    // Resolve the scene's referenced anim assets (state machines + root-motion
    // clips) so an `AnimStateMachine` / `RootMotion` entity steps in Simulate
    // exactly as it will in the shipped player (P11.4).
    // P24.1 adds the `.inf_skel` assets and the clips a machine's states play, so
    // the machine drives the DRAWN pose rather than only its own runtime state.
    let (machines, root_clips, skeletons, pose_clips) =
        inf_editor_core::simulate::resolve_anim_assets(&doc, |guid| {
            assets.load_anim_bytes(inf_asset::AssetId(guid))
        });
    // Resolve the scene's referenced `.inf_audio` clips (P12.3) so an `AudioSource`
    // plays the same clip in Simulate as in the shipped player.
    let audio_clips = inf_editor_core::simulate::resolve_audio_assets(&doc, |guid| {
        assets.load_audio_bytes(inf_asset::AssetId(guid))
    });
    // Resolve the scene's referenced `.inf_cloth` garments (P24.4) so a `ClothSim`
    // wearer folds the same coat in Simulate as in the shipped player.
    let cloths = inf_editor_core::simulate::resolve_cloth_assets(&doc, |guid| {
        assets.load_cloth_bytes(inf_asset::AssetId(guid))
    });
    // …and the `.inf_hair` hairstyles (P24.4), same door, same reason.
    let hairs = inf_editor_core::simulate::resolve_hair_assets(&doc, |guid| {
        assets.load_hair_bytes(inf_asset::AssetId(guid))
    });
    // Resolve the scene's referenced `.inf_voxel` volumes (P21.2) so a Blueprint
    // calling `terrain.height_at` over a carved hole reads the cave floor beneath
    // it — the same map the shipped player seeds through `attach_voxel_volumes`.
    // This is the SIM's set, camera-free: the viewport host pages its own against
    // the editor camera, and a fixed step must never be able to see that.
    let mut voxel_volumes = inf_editor_core::simulate::resolve_voxel_volumes(&doc, |guid| {
        assets.load_voxel_bytes(inf_asset::AssetId(guid))
    });
    // …and then the carves that are NOT on disk yet (P21.2 audit). Resolving from
    // the asset alone hands Simulate the last *saved* cave, so pressing Play after
    // digging a tunnel dropped the player onto solid rock. A streamed terrain does
    // not behave that way — `SimSession::enter` snapshots with
    // `ScenePersist::Memory` precisely so unsaved sculpts survive into a session
    // (P16.4b) — and a carve is the same act on the other surface. It cannot ride
    // the snapshot the way a sculpt does (scene schema v19 is frozen, so chunks
    // are not in the document), so it is read out of the shared store the viewport
    // carves into, on the same lock order every other reader uses: **document
    // first, volumes second** (the doc guard is still held here, taken above).
    //
    // Only the store's DIRTY chunks are folded in, which is what keeps the
    // determinism seam on `set_voxel_volumes` intact: dirty is a function of what
    // was dug, never of where the editor camera has paged. Headless/CI has no
    // viewport, so the process store is EMPTY and there is nothing to fold — the
    // resolved map stands as it always did (P23.2a: the store no longer hangs
    // off a viewport, so this reads it directly).
    if let Some(volumes) = app
        .try_state::<crate::commands::SharedStores>()
        .map(|s| s.voxel_volumes())
    {
        match volumes.lock() {
            Ok(store) => {
                let n = inf_editor_core::simulate::overlay_unsaved_carves(
                    &mut voxel_volumes,
                    |entity| store.slot(entity).map(|s| &s.data),
                );
                if n > 0 {
                    tracing::info!("simulate: {n} unsaved voxel chunk(s) carried into Simulate");
                }
            }
            // A poisoned store is a thread that already panicked mid-carve; its
            // chunks are not trustworthy input for a session. Play still starts,
            // on the saved volumes, rather than refusing outright.
            Err(_) => tracing::warn!(
                "simulate: voxel store lock poisoned — Simulate runs on the SAVED \
                 volumes, so unsaved carves will not be there"
            ),
        }
    }
    // **The document's gravity, not a literal** (P29.7). This line used to read
    // `DVec2::ZERO` under a comment about characters applying their own gravity;
    // a dynamic body reads the world's, so Simulate floated what the shipped
    // player dropped. `gravity_of` is the one rule both hosts read a level with.
    let gravity = SimSession::gravity_of(&doc);
    let mut session = SimSession::enter_with_gravity(&mut doc, actors, gravity, SIM_HZ);
    session.set_state_machines(machines);
    for (clip_guid, skeleton, clip) in root_clips {
        session.register_root_motion_clip(clip_guid, skeleton, clip);
    }
    session.set_skeletons(skeletons);
    session.set_pose_clips(pose_clips);
    session.set_audio_clips(audio_clips);
    session.set_cloths(cloths);
    session.set_hairs(hairs);
    session.set_voxel_volumes(voxel_volumes);
    // P22.3: what this level's destructible actors break into. DERIVED here, from
    // each actor's own mesh, by the same `inf_mesh::fracture_mesh` the cook runs
    // with the same authored seed and chunk count — because a `.inf_fracture` is
    // cook output and does not exist in the project's content root at all.
    //
    // **Through the SAME rule the PIE payload uses**, not a second walk that
    // happens to agree: `destructible_mesh_params` decides which of two actors
    // sharing a mesh sets its chunking, in `doc.order()`, and `derive_fracture`
    // applies the cook's own `convex_hull_is_buildable` refusal. The first cut of
    // this seeder used `iter_entities()` — ECS ARCHETYPE order — so adding an
    // `AlwaysLoaded` to one of two walls flipped which one won, and Simulate
    // shattered a wall into 24 chunks that the shipped pack shattered into 8.
    // One function now answers it for all three paths.
    //
    // Camera-free like the voxel map above: the walk reads the world's own
    // `Destructible` + `MeshRef`, never a render store.
    let fracture_params = inf_editor_core::pie::destructible_mesh_params(&doc);
    session.set_fractures(inf_physics::d3::resolve_fracture_states(
        doc.world(),
        |mesh_id| {
            let params = *fracture_params.get(&mesh_id)?;
            let bytes = assets.load_mesh_bytes(inf_asset::AssetId(mesh_id))?;
            let mesh = inf_asset::decode::<inf_mesh::MeshAsset>(&bytes).ok()?;
            inf_editor_core::pie::derive_fracture(&mesh, mesh_id, params).map(std::sync::Arc::new)
        },
    ));
    // Load the project mixer (`<project>/.infinity/mixer.toml`) if present; else the
    // default. The mixer lives at the project root (the parent of Content).
    if let Some(content) = assets.content_root() {
        let root = content.parent().map(|p| p.to_path_buf()).unwrap_or(content);
        if let Ok(mixer) = inf_audio::MixerConfig::load_or_default(&root) {
            session.set_audio_mixer(mixer);
        }
    }
    doc.bump_version_for_runtime();
    drop(doc);

    let mut inner = sim.inner.lock().map_err(|_| "sim lock poisoned")?;
    inner.session = Some(session);
    inner.last = Some(Instant::now());
    // ── P29.6 ── the native viewport captures the mouse for the GAME camera
    //    while a session is live. Backend to backend, `Target::All`, and it must
    //    be paired with the `false` in `sim_stop` or the pointer stays hidden.
    if let Some(vp) = app.try_state::<super::ViewportState>() {
        vp.set_sim_running(true);
    }
    let _ = app.emit("sim://state", true);
    Ok(())
}

/// Advance Simulate by one frame with the currently-held **physical keys**
/// (`KeyboardEvent.code` values, e.g. `["KeyA","Space"]`). No-op when not
/// running.
///
/// **Codes, not action names** (P29.6). The frontend used to map three keys onto
/// three action names in TypeScript, which meant the engine's binding table
/// existed twice in two languages and the second copy knew about three of its
/// seventeen entries. It sends the physical key now and
/// [`inf_input::default_map`] does the mapping — the same table the shipped
/// player uses, so `C` crouches in Simulate because it crouches in a build, and
/// an `input.toml` beside the level would change both.
#[tauri::command]
pub async fn sim_tick(
    app: AppHandle,
    scene: State<'_, SceneState>,
    sim: State<'_, SimState>,
    keys: Vec<String>,
) -> Result<bool, String> {
    let mut inner = sim.inner.lock().map_err(|_| "sim lock poisoned")?;
    if inner.session.is_none() {
        return Ok(false);
    }
    let now = Instant::now();
    let dt = inner
        .last
        .map(|t| now.duration_since(t).as_secs_f64())
        .unwrap_or(1.0 / SIM_HZ)
        // Guard a huge first/hitched frame (the stepper also clamps).
        .clamp(0.0, 0.25);
    inner.last = Some(now);

    let input = resolve_input(&mut inner, &keys, dt);
    let mut doc = scene.doc.lock().map_err(|_| "scene lock poisoned")?;
    let session = inner.session.as_mut().expect("session present");
    session.tick(&mut doc, dt, input);
    doc.bump_version_for_runtime();
    drop(doc);
    overlay_sim_carves(&app, &scene, session);
    publish_sim_fractures(&app, session);
    emit_debug(&app, session);
    Ok(true)
}

/// **Turn this frame's raw device state into the engine's own input** (P29.6).
///
/// Three things arrive here and one thing leaves. In: the physical keys the
/// frontend says are held, the mouse counts the native viewport captured since
/// the last frame, and the frame's own `dt`. Out: a [`SimInput`] carrying the
/// held **actions** and the resolved **axes** — including `look_x`/`look_y` as
/// degrees per second, which is what fills the analog axes P29.3 left empty and
/// what the locomotion camera and `RotationMode::Aiming` both read.
///
/// The key set is diffed rather than assigned, because `InputState` is an
/// event-driven machine: a key that is no longer in the frontend's set must
/// produce a **release**, or a character sprints for ever after one press.
///
/// The mouse accumulator is DRAINED here, so a frame that ran no fixed step does
/// not throw its motion away and a frame that ran three does not deliver it
/// three times — `axis_snapshot` turns the whole frame's counts into a rate and
/// the fixed step integrates that rate, which is the P29.3 rule.
fn resolve_input(inner: &mut SimInner, held: &[String], frame_dt: f64) -> SimInput {
    use std::collections::BTreeSet;
    let want: BTreeSet<&str> = held.iter().map(String::as_str).collect();
    let have: BTreeSet<String> = inner.input.keys_down().map(str::to_string).collect();
    let mut events: Vec<inf_input::InputEvent> = Vec::new();
    for code in want.iter().filter(|c| !have.contains(**c)) {
        events.push(inf_input::InputEvent::Key {
            code: (*code).to_string(),
            pressed: true,
        });
    }
    for code in have.iter().filter(|c| !want.contains(c.as_str())) {
        events.push(inf_input::InputEvent::Key {
            code: code.clone(),
            pressed: false,
        });
    }
    let (dx, dy) = std::mem::take(&mut inner.look);
    if dx != 0.0 || dy != 0.0 {
        events.push(inf_input::InputEvent::MouseMotion { delta: [dx, dy] });
    }
    inner.input.apply(&events);
    let (down, axes) = inner.input.resolved(frame_dt);
    SimInput::with_down(down).with_axes(axes)
}

/// **Mirror the Simulate session's runtime carves into the viewport's voxel
/// store** (P21.4) — the editor twin of `PlayerRenderHost::sync_voxels`' overlay.
///
/// A `voxel.*` node writes the *session's* volume map, which is deliberately not
/// the store the viewport draws from (the store is camera-paged; the sim's map
/// must never be). Without this the editor shows exactly what the shipped player
/// showed before the fold existed: a Blueprint digs, `terrain.height_at` moves,
/// the colliders rebuild — and the screen keeps drawing the rock.
///
/// `sim → render` only, and stamp-gated inside Ring 0
/// ([`inf_voxel::VoxelVolumes::overlay_sim`]): a frame in which nothing was dug
/// copies nothing. A poisoned store is skipped rather than fatal — Simulate keeps
/// running and the picture goes stale, which is strictly better than killing a
/// session over a render detail.
///
/// The terrain half needs no twin here: `SimSession` steps the **document's** own
/// world, so a runtime hole lands in the document and the viewport's existing
/// `overlay_document_edits` already mirrors it.
/// **Publish the Simulate session's fracture states into the viewport** (P22.3)
/// — the destruction twin of [`overlay_sim_carves`] above, and it exists for the
/// same reason.
///
/// A `destruct.*` node writes the *session's* state, which the viewport cannot
/// reach. Without this the editor would show exactly what the shipped player
/// showed before the carve fold existed: a Blueprint breaks a wall, the colliders
/// swap to chunk bodies, `destruct.is_intact` goes false — and the screen keeps
/// drawing the wall.
///
/// A **copy**, not a shared owner, and one direction only: the fixed step's map
/// stays the authority and the viewport reads a snapshot, so nothing the renderer
/// does can reach back into a simulation. `FractureState` is a handful of flags,
/// poses and an `Arc` to the shared chunk set, so a copy is cheap — the geometry
/// is not duplicated.
///
/// Cheap on the off path: a level with no destructible actor publishes an empty
/// map into an already-empty one.
fn publish_sim_fractures(app: &AppHandle, session: &inf_editor_core::simulate::SimSession) {
    // P23.2a: the fracture map is the PROCESS's, not a viewport's, so the
    // publish no longer depends on a window existing. With none attached it
    // writes into a map nobody reads and `sim_exit` clears it — the same
    // outcome the old early return had, without making "is a viewport open"
    // part of what a simulation does.
    let Some(handle) = app
        .try_state::<crate::commands::SharedStores>()
        .map(|s| s.fracture_states())
    else {
        return;
    };
    let Ok(mut states) = handle.lock() else {
        return;
    };
    if states.is_empty() && session.fractures().is_empty() {
        return;
    }
    *states = session.fractures().clone();
}

/// **THE LOCK RULE — never hold the document and the volumes at the same
/// time** (P23.2a audit; stated in full at `scene::scene_save`).
///
/// Two mutexes are reachable from Ring 2: the scene document (`SceneState::doc`,
/// which the viewport thread takes every frame) and the shared carve store
/// (`SharedStores::voxel_volumes`). The rule is *no overlap*, not an acquisition
/// order — the three sites that touch both genuinely differ in which they take
/// first, and that is safe only because none of them holds two.
///
/// This function used to be the one real exception. It held the *store* across
/// the *document*, both live for the whole loop — the classic two-lock deadlock
/// shape, needing only one caller anywhere to take them the other way round at
/// the same moment. It survived because before the P23.2a hoist the store hung
/// off a `ViewportHandle` and was awkward to reach; now it is one `try_state`
/// from any command already holding the document, so the shape had to go rather
/// than be commented.
///
/// The fix is to **snapshot the bindings under the document lock and release it
/// before touching the store**. The snapshot is `(entity, asset)` per volume:
/// the entity→asset binding, so a re-pointed `VoxelVolume.asset` cannot make
/// asset A's chunks land in a slot bound to asset B.
///
/// The trade is stated rather than hidden: the binding is now read a few
/// microseconds before it is used instead of under the same guard. That window
/// is not new — this runs *after* `sim_tick`/`sim_step_fixed` have already
/// dropped the document, so nothing was atomic with the step either way — and
/// `overlay_sim` re-checks the slot's bound asset one layer down, which is what
/// actually enforces the A-vs-B rule.
fn overlay_sim_carves(
    app: &AppHandle,
    scene: &SceneState,
    session: &inf_editor_core::simulate::SimSession,
) {
    let volumes = session.voxel_volumes();
    if volumes.is_empty() {
        return;
    }
    let Some(store) = app
        .try_state::<crate::commands::SharedStores>()
        .map(|s| s.voxel_volumes())
    else {
        return;
    };
    // ── document, and only the document ──────────────────────────────────────
    let bound: Vec<(uuid::Uuid, uuid::Uuid)> = {
        let doc = match scene.doc.lock() {
            Ok(d) => d,
            Err(_) => return,
        };
        volumes
            .keys()
            .filter_map(|entity| {
                doc.entity_of(*entity)
                    .and_then(|e| {
                        doc.world()
                            .world()
                            .get::<inf_ecs::components::VoxelVolume>(e)
                    })
                    .and_then(|v| v.asset)
                    .map(|asset| (*entity, asset))
            })
            .collect()
    };
    // ── …released. Now the store, and only the store ─────────────────────────
    let Ok(mut store) = store.lock() else {
        return;
    };
    for (entity, asset) in bound {
        let Some(data) = volumes.get(&entity) else {
            continue;
        };
        if store.overlay_sim(entity, asset, data) > 0 {
            store.resync(entity);
        }
    }
}

/// Advance Simulate by **exactly one fixed step** with the held keys (B-P4 tier
/// A′) — bypasses the wall-clock accumulator, so Step is a guaranteed single
/// step (fixes the documented `sim_step_fixed` gap). No-op when not running.
#[tauri::command]
pub async fn sim_step_fixed(
    app: AppHandle,
    scene: State<'_, SceneState>,
    sim: State<'_, SimState>,
    keys: Vec<String>,
) -> Result<bool, String> {
    let mut inner = sim.inner.lock().map_err(|_| "sim lock poisoned")?;
    if inner.session.is_none() {
        return Ok(false);
    }
    // A fixed step does not advance wall-clock time; reset `last` so the next
    // free-running tick's delta starts fresh from now.
    inner.last = Some(Instant::now());
    let input = resolve_input(&mut inner, &keys, 1.0 / SIM_HZ);
    let mut doc = scene.doc.lock().map_err(|_| "scene lock poisoned")?;
    let session = inner.session.as_mut().expect("session present");
    session.step_once(&mut doc, input);
    doc.bump_version_for_runtime();
    drop(doc);
    overlay_sim_carves(&app, &scene, session);
    publish_sim_fractures(&app, session);
    emit_debug(&app, session);
    Ok(true)
}

/// Install the debugger config for an actor class (B-P4 tier A′): `breakpoints`
/// (IR `LocalId`s for hand-built classes) + wire `capture`. When a subsequent
/// step hits a breakpoint or captures wires, the backend emits `sim://debug`.
/// No-op (returns `false`) when no session is running.
#[tauri::command]
pub async fn sim_set_debug(
    sim: State<'_, SimState>,
    class_id: String,
    breakpoints: Vec<u32>,
    capture: bool,
) -> Result<bool, String> {
    let mut inner = sim.inner.lock().map_err(|_| "sim lock poisoned")?;
    let Some(session) = inner.session.as_mut() else {
        return Ok(false);
    };
    let debug = InterpDebug {
        breakpoints: breakpoints.into_iter().map(LocalId).collect(),
        capture_wires: capture,
    };
    session.set_debug(class_id, debug);
    Ok(true)
}

/// **Queue a live tuning edit** (P29.5, pillar S4) — a movement tunable, a
/// machine parameter or a machine trigger.
///
/// The rule is `inf_editor_core::tuning`; this is the string↔enum hop and one
/// `State` lookup, per the typed-IPC law. It applies at the top of the **next**
/// fixed step, never inside one.
///
/// `kind` is `"field"`, `"param"`, `"trigger"`, `"vehicle"` or `"camera"`. For
/// `"field"`, `name` is the reflect access path and `type_path` names the
/// component (defaulting to `CharacterMovement`, which is what a movement tuner
/// edits). `keep` chooses whether the value survives Stop — and is ignored by
/// the three kinds that have no document field to keep it on.
///
/// `"camera"` takes no entity, so its `guid` is not read; passing a nil one is
/// the honest call.
///
/// Returns `false` when no session is running, which is a **value**: a tuning
/// panel is live over a session the author can stop at any moment, and a toast
/// for every slider tick after Stop would be the panel's whole behaviour.
#[tauri::command]
pub async fn sim_tune(
    sim: State<'_, SimState>,
    kind: String,
    guid: String,
    name: String,
    value: f64,
    type_path: Option<String>,
    keep: bool,
) -> Result<bool, String> {
    let guid: uuid::Uuid = if kind == "camera" {
        uuid::Uuid::nil()
    } else {
        guid.parse().map_err(|e| format!("bad guid: {e}"))?
    };
    if name.trim().is_empty() {
        return Err("a tune needs a field or parameter name".to_string());
    }
    let tune = match kind.as_str() {
        "field" => inf_editor_core::tuning::Tune::Field {
            guid,
            type_path: type_path
                .unwrap_or_else(|| inf_editor_core::tuning::MOVEMENT_TYPE_PATH.to_string()),
            path: name,
            value: inf_ecs::PropValue::Number(value),
        },
        "param" => inf_editor_core::tuning::Tune::Param { guid, name, value },
        "trigger" => inf_editor_core::tuning::Tune::Trigger { guid, name },
        "vehicle" => inf_editor_core::tuning::Tune::Vehicle { guid, name, value },
        "camera" => inf_editor_core::tuning::Tune::Camera { name, value },
        other => {
            return Err(format!(
                "unknown tune kind `{other}` (expected field, param, trigger, vehicle or camera)"
            ))
        }
    };
    let scope = if keep {
        inf_editor_core::tuning::TuneScope::Keep
    } else {
        inf_editor_core::tuning::TuneScope::Session
    };
    let mut inner = sim.inner.lock().map_err(|_| "sim lock poisoned")?;
    let Some(session) = inner.session.as_mut() else {
        return Ok(false);
    };
    session.tune(tune, scope);
    Ok(true)
}

/// Exit Simulate: restore the pre-play world exactly.
#[tauri::command]
pub async fn sim_stop(
    app: AppHandle,
    scene: State<'_, SceneState>,
    sim: State<'_, SimState>,
) -> Result<(), String> {
    // **The mouse goes back FIRST** (P29.6 audit, A12). This used to be the last
    // statement in the function, below two `?`s on poisonable locks — and the
    // session is taken out of the state before either of them. So a poisoned
    // lock returned `Err` with the session already gone and `SIM_RUNNING` still
    // true in every viewport thread, for the life of the process: every
    // subsequent plain LMB in the viewport would hide the cursor and grab the
    // mouse for a session that does not exist, and `push_look` would discard the
    // deltas in silence. `sim_start` already places its `set_sim_running(true)`
    // after all its fallible work, for the mirror reason.
    if let Some(vp) = app.try_state::<super::ViewportState>() {
        vp.set_sim_running(false);
    }
    let session = {
        let mut inner = sim.inner.lock().map_err(|_| "sim lock poisoned")?;
        inner.last = None;
        inner.session.take()
    };
    if let Some(session) = session {
        let mut doc = scene.doc.lock().map_err(|_| "scene lock poisoned")?;
        session.exit(&mut doc);
        doc.bump_version_for_runtime();
    }
    // P22.3: the rubble dies with the session. Destruction is not persisted (the
    // `destruct.*` kit says so, for the reason the `voxel.*` kit said it first),
    // so leaving the states behind would draw a broken wall over an intact
    // document — the P21.4 "the editor's render store IS the save's staging
    // source" hazard, one component over.
    if let Some(handle) = app
        .try_state::<crate::commands::SharedStores>()
        .map(|s| s.fracture_states())
    {
        if let Ok(mut states) = handle.lock() {
            states.clear();
        }
    }
    let _ = app.emit("sim://state", false);
    Ok(())
}

/// Whether a Simulate session is currently running.
#[tauri::command]
pub async fn sim_is_running(sim: State<'_, SimState>) -> Result<bool, String> {
    Ok(sim
        .inner
        .lock()
        .map_err(|_| "sim lock poisoned")?
        .session
        .is_some())
}

/// One handler's Blueprint debug observation emitted on `sim://debug` (B-P4 tier
/// A′). The wire-inspector JSON shape (mirrors the frontend `SimDebugEvent`).
#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct SimDebugEventDto {
    class_id: String,
    event: String,
    fn_name: String,
    hits: Vec<u32>,
    wires: Vec<(u32, String)>,
}

impl From<SimDebugHit> for SimDebugEventDto {
    fn from(h: SimDebugHit) -> Self {
        Self {
            class_id: h.class_id,
            event: h.event,
            fn_name: h.fn_name,
            hits: h.hits,
            wires: h.wires,
        }
    }
}

/// Drain the session's debug events and, when any were produced, broadcast them
/// on `sim://debug` (the frontend pauses on hits + shows wire values). No-op when
/// no class is being debugged.
fn emit_debug(app: &AppHandle, session: &mut SimSession) {
    let events = session.take_debug_events();
    if events.is_empty() {
        return;
    }
    let dto: Vec<SimDebugEventDto> = events.into_iter().map(Into::into).collect();
    let _ = app.emit("sim://debug", dto);
}

#[cfg(test)]
mod tests {
    /// **The live-tuning panel names movement fields that exist** (P29.5,
    /// pillar S4).
    ///
    /// `LiveTuning.tsx` carries a list of `CharacterMovement` field names as
    /// TypeScript string literals, and every one of them crosses to Rust as a
    /// `bevy_reflect` access path. A misspelling is a **silent no-op**: the tune
    /// is queued, the write returns `false`, the panel says "next step" and the
    /// value never moves — which is exactly the two-copies-of-one-expression
    /// defect the campaign's Wave I found across this same language boundary.
    ///
    /// So the list is read out of the `.tsx` and checked against the component's
    /// own reflected fields. The panel is the copy; the component is the truth.
    #[test]
    fn every_movement_tunable_the_panel_offers_is_a_real_field() {
        const PANEL: &str = include_str!("../../../src/panels/sm/LiveTuning.tsx");

        // The list, extracted from the source rather than restated here — a
        // restatement is a third copy.
        let named: Vec<&str> = PANEL
            .split("field: \"")
            .skip(1)
            .filter_map(|rest| rest.split('"').next())
            .collect();
        assert!(
            named.len() >= 5,
            "found {} tunables in LiveTuning.tsx — this gate is reading the wrong file or the wrong shape",
            named.len()
        );

        // The component's own reflected field names, through the same registry
        // `props::write_field` resolves a tune against.
        let mut world = inf_ecs::EcsWorld::new();
        let e = world
            .world_mut()
            .spawn(inf_ecs::components::CharacterMovement::default())
            .id();
        let props = inf_ecs::props::read_entity(world.world(), world.registry(), e);
        let movement = props
            .iter()
            .find(|p| p.type_path == inf_editor_core::tuning::MOVEMENT_TYPE_PATH)
            .expect("CharacterMovement is a reflected, editable component");
        let fields: std::collections::BTreeSet<&str> =
            movement.fields.iter().map(|f| f.name.as_str()).collect();

        for n in &named {
            assert!(
                fields.contains(n),
                "LiveTuning.tsx offers `{n}`, which is not a field of CharacterMovement: {fields:?}"
            );
        }
        // Anti-vacuity: the field set is a real one, so `contains` is answering.
        assert!(fields.contains("walk_speed_mps"), "{fields:?}");
        assert!(!fields.contains("sprint_speed_mph"));
    }
}
